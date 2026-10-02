"""Prediction endpoints for single-customer and batch CSV scoring."""

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from api.schemas.prediction import (
    BatchPredictionSummary,
    CustomerFeaturesInput,
    SinglePredictionResponse,
)
from api.services import ModelService
from src.database.connection import get_db
from src.database.models import AnomalyReport, Customer, CustomerSegment, Prediction
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Predictions"])


def _persist_prediction_result(
    db: Session,
    pred: SinglePredictionResponse,
    customer_country: str = "United Kingdom",
) -> None:
    """Helper to persist single prediction and anomaly report into database."""
    if pred.customer_id is None:
        return

    c_id = pred.customer_id

    # 1. Upsert Customer master record if not existing
    cust = db.query(Customer).filter(Customer.customer_id == c_id).first()
    if not cust:
        cust = Customer(
            customer_id=c_id,
            country=customer_country,
            is_uk=customer_country == "United Kingdom",
        )
        db.add(cust)
        db.flush()

    # 2. Upsert Customer Segment
    seg = db.query(CustomerSegment).filter(CustomerSegment.customer_id == c_id).first()
    if not seg:
        seg = CustomerSegment(
            customer_id=c_id,
            segment_id=0,
            segment_name=pred.segment_name,
            engagement_score=50.0,
            updated_at=pred.scored_at,
        )
        db.add(seg)
    else:
        seg.segment_name = pred.segment_name
        seg.updated_at = pred.scored_at

    # 3. Insert Prediction Record
    db_pred = Prediction(
        customer_id=c_id,
        churn_probability=pred.churn_probability,
        is_churn=pred.is_churn,
        predicted_clv=pred.predicted_clv_90d,
        risk_tier=pred.risk_tier,
        model_version=pred.model_version,
        scored_at=pred.scored_at,
    )
    db.add(db_pred)

    # 4. Insert Anomaly Report Record
    db_anomaly = AnomalyReport(
        customer_id=c_id,
        reconstruction_error=pred.anomaly_score,
        is_anomaly_95=pred.is_anomaly,
        is_anomaly_99=bool(pred.anomaly_score >= 0.1706),
        flagged_at=pred.scored_at,
    )
    db.add(db_anomaly)


@router.post(
    "/predict/customer",
    response_model=SinglePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Single Customer Behavior",
    description=(
        "Accepts single customer features or raw behavioral metrics. "
        "Returns churn probability, predicted CLV, segment persona, anomaly status, "
        "and a plain-language marketing narrative with top SHAP risk drivers."
    ),
)
def predict_customer(
    features: CustomerFeaturesInput,
    persist: bool = True,
    db: Session = Depends(get_db),
) -> SinglePredictionResponse:
    """Computes multi-model inference for an individual customer."""
    svc = ModelService.get_instance()
    try:
        prediction = svc.predict_single(features)
    except Exception as e:
        logger.error(f"Inference error for customer {features.customer_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(e)}",
        )

    if persist and prediction.customer_id is not None:
        try:
            _persist_prediction_result(db, prediction)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to persist prediction for customer {prediction.customer_id}: {e}")

    return prediction


@router.post(
    "/predict/batch",
    response_model=BatchPredictionSummary,
    status_code=status.HTTP_200_OK,
    summary="Batch Predict via CSV Upload",
    description=(
        "Upload a CSV file containing multiple customer records. "
        "Runs batch inference across all models, persists predictions to the database, "
        "and returns cohort summary statistics along with full individual results and CSV payload."
    ),
)
def predict_batch(
    file: UploadFile = File(..., description="CSV file containing customer feature records"),
    persist: bool = True,
    db: Session = Depends(get_db),
) -> BatchPredictionSummary:
    """Processes uploaded CSV file, scores records in batch, and returns summaries."""
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a CSV format.",
        )

    try:
        contents = file.file.read()
        df_batch = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Error parsing uploaded CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}",
        )

    if df_batch.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file contains no rows.",
        )

    svc = ModelService.get_instance()
    try:
        summary = svc.predict_batch_df(df_batch)
    except Exception as e:
        logger.error(f"Batch prediction failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}",
        )

    if persist:
        try:
            for pred in summary.predictions:
                _persist_prediction_result(db, pred)
            db.commit()
            logger.info(f"Persisted {len(summary.predictions)} batch predictions to database.")
        except Exception as e:
            db.rollback()
            logger.warning(f"Database batch persistence warning: {e}")

    return summary
