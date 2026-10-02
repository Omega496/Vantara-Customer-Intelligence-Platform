"""Model metadata and performance tracking router."""

from fastapi import APIRouter, status

from api.schemas.prediction import ModelMetadataResponse
from api.services import ModelService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Models"])


@router.get(
    "/models/metadata",
    response_model=ModelMetadataResponse,
    status_code=status.HTTP_200_OK,
    summary="Active Production Models Metadata",
    description="Returns detailed metadata, validation metrics (ROC-AUC, R², Silhouette), and active feature inputs for all deployed models.",
)
def get_model_metadata() -> ModelMetadataResponse:
    """Returns metadata for churn, CLV, segmentation, and anomaly detection models."""
    svc = ModelService.get_instance()

    return ModelMetadataResponse(
        production_churn_model={
            "model_name": "LightGBM Classifier",
            "version": "v1.0.0",
            "algorithm": "Gradient Boosted Decision Trees",
            "training_date": "2026-10-01",
            "test_roc_auc": 0.8205,
            "test_recall": 0.8164,
            "test_f1": 0.8013,
            "hyperparameters": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 5,
                "num_leaves": 31,
            },
            "objective": "Predict probability of customer inactivity in 90-day target window",
        },
        clv_regression_model={
            "model_name": "Ridge Regressor",
            "version": "v1.0.0",
            "algorithm": "L2-Regularized Linear Regression",
            "training_date": "2026-10-01",
            "test_r2": 0.9150,
            "test_mae": 366.30,
            "test_rmse": 1298.54,
            "target": "Forward 90-day customer monetary value (£)",
        },
        segmentation_model={
            "model_name": "K-Means Clustering",
            "version": "v1.0.0",
            "k_clusters": 4,
            "silhouette_score": 0.4193,
            "personas": [
                "VIP Champions",
                "Loyal Regulars",
                "High-Value Inactive",
                "At-Risk / Lapsed",
            ],
            "profiles": svc.segment_profiles,
        },
        anomaly_autoencoder={
            "model_name": "Deep Spending Autoencoder (PyTorch)",
            "version": "v1.0.0",
            "architecture": "MLP (Input 12 -> 32 -> 16 -> Latent 6 -> 16 -> 32 -> Output 12)",
            "threshold_p95": svc.anomaly_meta.get("threshold_p95", 0.0748),
            "threshold_p99": svc.anomaly_meta.get("threshold_p99", 0.1706),
            "flagged_accounts_p95": svc.anomaly_meta.get("p95_anomalies_count", 265),
        },
        feature_count=len(svc.feature_names),
        active_features=svc.feature_names,
    )
