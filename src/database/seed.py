"""Database seeding module to populate initial customer cohort, segments, and predictions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session

from src.database.connection import get_engine, init_db
from src.database.models import AnomalyReport, Customer, CustomerSegment, Prediction
from src.models.autoencoder import (
    AUTOENCODER_FEATURES,
    SpendingAutoencoder,
    compute_reconstruction_errors,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def seed_database(limit: Optional[int] = None) -> int:
    """Populates the database with historical customers, assigned segments, predictions, and anomaly scores.

    Args:
        limit: Optional maximum number of customers to seed (default: all ~5,300).

    Returns:
        Number of customer records seeded.
    """
    logger.info("Starting database seeding process...")
    engine = get_engine()
    init_db(engine)

    seg_file = Path("data/processed/customer_segments.parquet")
    if not seg_file.exists():
        raise FileNotFoundError(f"Segmented data file not found at {seg_file}. Please run Milestone 3 first.")

    df = pd.read_parquet(seg_file)
    if limit is not None:
        df = df.head(limit)

    total_records = len(df)
    logger.info(f"Loaded {total_records:,} customer records for database seeding.")

    # 1. Load trained models and metadata
    churn_model = joblib.load("models_artifacts/churn_production_model.joblib")
    clv_model = joblib.load("models_artifacts/clv_best_model.joblib")
    scaler = joblib.load("models_artifacts/scaler.joblib")
    ae_scaler = joblib.load("models_artifacts/autoencoder_scaler.joblib")

    with open("models_artifacts/feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open("models_artifacts/anomaly_threshold.json", "r", encoding="utf-8") as f:
        anomaly_meta = json.load(f)
    thresh_95 = anomaly_meta["threshold_p95"]
    thresh_99 = anomaly_meta["threshold_p99"]

    # 2. Compute model predictions
    X_features = df[feature_names].copy().fillna(0.0)
    churn_probs = churn_model.predict_proba(X_features)[:, 1]

    # Predict CLV (Ridge expects scaled features)
    X_scaled = scaler.transform(X_features)
    predicted_clv = np.clip(clv_model.predict(X_scaled), 0.0, None)

    # Compute Autoencoder Anomaly Reconstruction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_features = [f for f in AUTOENCODER_FEATURES if f in df.columns]
    X_ae = ae_scaler.transform(df[ae_features].fillna(0.0))

    ae_model = SpendingAutoencoder(input_dim=len(ae_features), latent_dim=6).to(device)
    ae_model.load_state_dict(torch.load("models_artifacts/spending_autoencoder.pt", map_location=device))
    recon_errors = compute_reconstruction_errors(ae_model, X_ae, device)

    # 3. Seed into Database
    now = datetime.utcnow()
    customers_to_insert = []
    segments_to_insert = []
    predictions_to_insert = []
    anomalies_to_insert = []

    # Load transaction dates if available
    date_lookup = {}
    tx_file = Path("data/interim/cleaned_transactions.parquet")
    if tx_file.exists():
        df_tx = pd.read_parquet(tx_file, columns=["customer_id", "invoice_date"])
        agg_dates = df_tx.groupby("customer_id")["invoice_date"].agg(first_order="min", last_order="max")
        date_lookup = agg_dates.to_dict(orient="index")

    for i in range(total_records):
        row = df.iloc[i]
        c_id = int(row["customer_id"])
        prob = float(churn_probs[i])
        clv = float(round(predicted_clv[i], 2))
        rec_err = float(round(recon_errors[i], 4))

        if prob >= 0.70:
            tier = "High Risk"
        elif prob >= 0.40:
            tier = "Medium Risk"
        else:
            tier = "Low Risk"

        first_d = date_lookup.get(c_id, {}).get("first_order") if date_lookup else None
        last_d = date_lookup.get(c_id, {}).get("last_order") if date_lookup else None

        customers_to_insert.append(
            Customer(
                customer_id=c_id,
                country=str(row.get("primary_country", "United Kingdom")),
                first_order_date=pd.to_datetime(first_d).to_pydatetime() if pd.notnull(first_d) else None,
                last_order_date=pd.to_datetime(last_d).to_pydatetime() if pd.notnull(last_d) else None,
                is_uk=bool(row.get("is_uk", 1)),
            )
        )

        segments_to_insert.append(
            CustomerSegment(
                customer_id=c_id,
                segment_id=int(row.get("cluster_id", 0)),
                segment_name=str(row.get("segment_name", "Loyal Regulars")),
                engagement_score=float(round(row.get("engagement_score", 50.0), 2)),
                updated_at=now,
            )
        )

        predictions_to_insert.append(
            Prediction(
                customer_id=c_id,
                churn_probability=round(prob, 4),
                is_churn=bool(prob >= 0.50),
                predicted_clv=clv,
                risk_tier=tier,
                model_version="v1.0.0",
                scored_at=now,
            )
        )

        anomalies_to_insert.append(
            AnomalyReport(
                customer_id=c_id,
                reconstruction_error=rec_err,
                is_anomaly_95=bool(rec_err >= thresh_95),
                is_anomaly_99=bool(rec_err >= thresh_99),
                flagged_at=now,
            )
        )

    with Session(engine) as session:
        # Clear existing records
        session.query(AnomalyReport).delete()
        session.query(Prediction).delete()
        session.query(CustomerSegment).delete()
        session.query(Customer).delete()

        # Bulk insert
        session.bulk_save_objects(customers_to_insert)
        session.bulk_save_objects(segments_to_insert)
        session.bulk_save_objects(predictions_to_insert)
        session.bulk_save_objects(anomalies_to_insert)
        session.commit()

    logger.info(f"Successfully seeded database with {total_records:,} customers across all tables.")
    return total_records


if __name__ == "__main__":
    seed_database()
