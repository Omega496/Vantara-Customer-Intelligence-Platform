"""Health check and system status router."""

from datetime import datetime

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.schemas.prediction import HealthResponse
from api.services import ModelService
from src.database.connection import get_db, get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Model Status",
    description="Returns real-time health status, database connectivity, and active ML/DL model readiness.",
)
def check_health(db: Session = Depends(get_db)) -> HealthResponse:
    """Verifies database reachability and in-memory model availability."""
    db_status = "unreachable"
    try:
        # Check active session
        db.execute(text("SELECT 1"))
        engine = get_engine()
        db_status = f"connected ({engine.url.drivername})"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = f"error: {str(e)[:100]}"

    # Verify model service status
    svc = ModelService.get_instance()
    loaded_models = []
    if svc.churn_model is not None:
        loaded_models.append("churn_lightgbm_v1")
    if svc.clv_model is not None:
        loaded_models.append("clv_ridge_regressor_v1")
    if svc.kmeans is not None:
        loaded_models.append("kmeans_clustering_v1")
    if svc.autoencoder is not None:
        loaded_models.append("spending_autoencoder_pytorch_v1")
    if svc.shap_engine is not None:
        loaded_models.append("shap_tree_explainer")

    overall_status = "healthy" if "connected" in db_status and len(loaded_models) >= 4 else "degraded"

    return HealthResponse(
        status=overall_status,
        database=db_status,
        loaded_models=loaded_models,
        timestamp=datetime.utcnow(),
    )
