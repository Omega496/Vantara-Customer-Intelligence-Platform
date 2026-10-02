"""Inference and model prediction service with in-memory model caching."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from api.schemas.prediction import (
    BatchPredictionSummary,
    CustomerFeaturesInput,
    SinglePredictionResponse,
)
from src.explainability.narrative import generate_plain_language_narrative
from src.explainability.shap_explainer import ShapExplainabilityEngine
from src.models.autoencoder import (
    AUTOENCODER_FEATURES,
    SpendingAutoencoder,
    compute_reconstruction_errors,
)
from src.segmentation.clustering import CLUSTERING_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    """Manages loaded ML/DL models in memory and handles low-latency scoring."""

    _instance: Optional["ModelService"] = None

    def __init__(self, artifacts_dir: str = "models_artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing ModelService with device: {self.device}")

        # 1. Load feature names & metadata
        with open(self.artifacts_dir / "feature_names.json", "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        with open(self.artifacts_dir / "anomaly_threshold.json", "r", encoding="utf-8") as f:
            self.anomaly_meta = json.load(f)

        with open(self.artifacts_dir / "segment_profiles.json", "r", encoding="utf-8") as f:
            self.segment_profiles = json.load(f)

        # 2. Load trained models & scalers
        self.churn_model = joblib.load(self.artifacts_dir / "churn_production_model.joblib")
        self.clv_model = joblib.load(self.artifacts_dir / "clv_best_model.joblib")
        self.scaler = joblib.load(self.artifacts_dir / "scaler.joblib")

        self.kmeans = joblib.load(self.artifacts_dir / "kmeans_segmentation.joblib")
        self.seg_scaler = joblib.load(self.artifacts_dir / "segmentation_scaler.joblib")

        self.ae_scaler = joblib.load(self.artifacts_dir / "autoencoder_scaler.joblib")
        self.autoencoder = SpendingAutoencoder(input_dim=len(AUTOENCODER_FEATURES), latent_dim=6).to(self.device)
        self.autoencoder.load_state_dict(
            torch.load(self.artifacts_dir / "spending_autoencoder.pt", map_location=self.device)
        )
        self.autoencoder.eval()

        # 3. Initialize SHAP Explainer
        self.shap_engine = ShapExplainabilityEngine(
            model_path=str(self.artifacts_dir / "churn_production_model.joblib"),
            feature_names=self.feature_names,
        )

        # 4. Load historical RFM profiles for instant retrieval
        self.rfm_lookup: Dict[int, Dict[str, Any]] = {}
        feat_path = Path("data/processed/customer_features.parquet")
        if feat_path.exists():
            try:
                df_feat = pd.read_parquet(feat_path)
                for _, r in df_feat.iterrows():
                    cid = int(r["customer_id"])
                    self.rfm_lookup[cid] = {
                        "recency_days": float(r.get("recency_days", 0.0)),
                        "frequency": float(r.get("frequency", 0.0)),
                        "total_spend": float(round(r.get("total_spend", 0.0), 2)),
                        "avg_order_value": float(round(r.get("avg_order_value", 0.0), 2)),
                        "avg_basket_size": float(round(r.get("avg_basket_size", 0.0), 2)),
                        "tenure_days": float(r.get("tenure_days", 0.0)),
                        "engagement_score": float(round(r.get("engagement_score", 50.0), 2)),
                        "return_rate": float(round(r.get("return_line_rate", 0.0), 4)),
                    }
                logger.info(f"Loaded {len(self.rfm_lookup):,} historical customer RFM profiles into cache.")
            except Exception as e:
                logger.warning(f"Could not load customer RFM profiles: {e}")

        logger.info("ModelService initialized successfully. All models ready for inference.")

    def get_rfm_metrics(self, customer_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves cached historical RFM metrics for a given customer."""
        return self.rfm_lookup.get(customer_id)

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _prepare_feature_row(self, input_data: CustomerFeaturesInput) -> pd.DataFrame:
        """Fills missing derived features and aligns columns to model schema."""
        d = input_data.model_dump()

        # Auto-compute derived fields if not passed
        freq = float(d.get("frequency") or 1.0)
        freq = max(1.0, freq)
        spend = float(d.get("total_spend") or 0.0)

        if d.get("gross_spend") is None:
            d["gross_spend"] = max(spend, 0.0)
        if d.get("avg_order_value") is None:
            d["avg_order_value"] = spend / freq
        if d.get("avg_basket_size") is None:
            total_items = d.get("total_items")
            d["avg_basket_size"] = (float(total_items) / freq) if total_items is not None else 10.0
        if d.get("avg_unique_products_per_order") is None:
            unique_prod = d.get("unique_products_count")
            d["avg_unique_products_per_order"] = (float(unique_prod) / freq) if unique_prod is not None else 3.0
        if d.get("log_total_spend") is None:
            d["log_total_spend"] = float(np.log1p(max(0.0, spend)))
        if d.get("log_frequency") is None:
            d["log_frequency"] = float(np.log1p(freq))
        rec_days = float(d.get("recency_days") or 30.0)
        if d.get("tenure_days") is None:
            d["tenure_days"] = max(rec_days, 30.0)
        if d.get("days_active_span") is None:
            d["days_active_span"] = max(0.0, float(d["tenure_days"]) - rec_days)
        if d.get("inter_purchase_mean") is None:
            d["inter_purchase_mean"] = float(d["tenure_days"]) / max(freq - 1.0, 1.0)
        if d.get("inter_purchase_std") is None:
            d["inter_purchase_std"] = 0.0
        if d.get("is_single_order") is None:
            d["is_single_order"] = 1.0 if freq <= 1.0 else 0.0
        if d.get("engagement_score") is None:
            # Approximation from RFM
            d["engagement_score"] = float(
                np.clip(50.0 + (freq * 3.0) + (spend / 100.0) - (rec_days / 5.0), 0.0, 100.0)
            )

        row_dict = {col: (float(d[col]) if d.get(col) is not None else 0.0) for col in self.feature_names}
        return pd.DataFrame([row_dict], columns=self.feature_names)

    def predict_single(self, input_data: CustomerFeaturesInput) -> SinglePredictionResponse:
        """Executes full multi-model prediction pipeline on a single customer."""
        df_row = self._prepare_feature_row(input_data)

        # 1. Churn Prediction
        churn_prob = float(self.churn_model.predict_proba(df_row)[0, 1])
        is_churn = churn_prob >= 0.50

        if churn_prob >= 0.70:
            risk_tier = "High Risk"
        elif churn_prob >= 0.40:
            risk_tier = "Medium Risk"
        else:
            risk_tier = "Low Risk"

        # 2. CLV Prediction (scaled)
        row_scaled = self.scaler.transform(df_row.values)
        pred_clv = float(round(float(np.clip(self.clv_model.predict(row_scaled)[0], 0.0, None)), 2))

        # 3. Customer Segmentation
        seg_input = df_row[[c for c in CLUSTERING_FEATURES if c in df_row.columns]].copy()
        seg_scaled = self.seg_scaler.transform(seg_input.values)
        cluster_id = int(self.kmeans.predict(seg_scaled)[0])

        cluster_map = {0: "At-Risk / Lapsed", 1: "Loyal Regulars", 2: "VIP Champions", 3: "High-Value Inactive"}
        segment_name = cluster_map.get(cluster_id, f"Cluster {cluster_id}")

        # 4. Anomaly Detection
        ae_input = df_row[[c for c in AUTOENCODER_FEATURES if c in df_row.columns]].copy()
        ae_scaled = self.ae_scaler.transform(ae_input.values)
        recon_err = float(compute_reconstruction_errors(self.autoencoder, ae_scaled, self.device)[0])
        is_anomaly = recon_err >= self.anomaly_meta["threshold_p95"]

        # 5. SHAP Explanation & Plain Language Narrative
        c_id = input_data.customer_id or 99999
        shap_res = self.shap_engine.explain_individual_customer(df_row.iloc[0], customer_id=c_id)
        narrative = generate_plain_language_narrative(
            customer_id=c_id,
            churn_probability=churn_prob,
            top_factors=shap_res["top_factors"],
        )

        return SinglePredictionResponse(
            customer_id=input_data.customer_id,
            churn_probability=round(churn_prob, 4),
            is_churn=is_churn,
            predicted_clv_90d=pred_clv,
            risk_tier=risk_tier,
            segment_name=segment_name,
            is_anomaly=is_anomaly,
            anomaly_score=round(recon_err, 4),
            plain_language_narrative=narrative,
            top_risk_factors=shap_res["top_factors"][:5],
            model_version="v1.0.0",
            scored_at=datetime.utcnow(),
        )

    def predict_batch_df(self, df_batch: pd.DataFrame) -> BatchPredictionSummary:
        """Processes batch DataFrame of customer records."""
        predictions: List[SinglePredictionResponse] = []
        for _, row in df_batch.iterrows():
            item_dict = row.to_dict()
            inp = CustomerFeaturesInput(**item_dict)
            pred = self.predict_single(inp)
            predictions.append(pred)

        churn_rate = float(np.mean([p.churn_probability for p in predictions])) if predictions else 0.0
        mean_clv = float(np.mean([p.predicted_clv_90d for p in predictions])) if predictions else 0.0
        anomalies_count = sum(1 for p in predictions if p.is_anomaly)

        # Generate CSV payload
        csv_rows = [
            "customer_id,churn_probability,is_churn,predicted_clv_90d,risk_tier,segment_name,is_anomaly,anomaly_score"
        ]
        for p in predictions:
            csv_rows.append(
                f"{p.customer_id if p.customer_id is not None else ''},"
                f"{p.churn_probability},{p.is_churn},{p.predicted_clv_90d},"
                f'"{p.risk_tier}","{p.segment_name}",{p.is_anomaly},{p.anomaly_score}'
            )
        csv_payload = "\n".join(csv_rows)

        return BatchPredictionSummary(
            total_records=len(predictions),
            churn_rate=round(churn_rate, 4),
            mean_predicted_clv=round(mean_clv, 2),
            anomalies_detected=anomalies_count,
            download_csv_payload=csv_payload,
            download_url="/predict/batch",
            predictions=predictions,
        )
