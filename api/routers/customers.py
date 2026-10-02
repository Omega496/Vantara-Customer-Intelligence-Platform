"""Customer profiles, segmentation personas, and historical metrics router."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.schemas.customer import (
    CustomerAnomalyRecord,
    CustomerDetailResponse,
    CustomerPredictionRecord,
    CustomerSegmentProfile,
)
from api.services import ModelService
from src.database.connection import get_db
from src.database.models import AnomalyReport, Customer, CustomerSegment, Prediction
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Customers"])


@router.get(
    "/customers/{customer_id}",
    response_model=CustomerDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Customer Details & Predictions",
    description=(
        "Fetches persisted master data, current assigned segment persona, "
        "latest predictions, historical audit records, and historical RFM metrics."
    ),
)
def get_customer_by_id(
    customer_id: int,
    db: Session = Depends(get_db),
) -> CustomerDetailResponse:
    """Retrieves comprehensive customer detail record by customer_id."""
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        logger.warning(f"Customer {customer_id} not found in database.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found.",
        )

    # 1. Fetch Segment
    seg = db.query(CustomerSegment).filter(CustomerSegment.customer_id == customer_id).first()
    seg_profile: Optional[CustomerSegmentProfile] = None
    if seg:
        seg_profile = CustomerSegmentProfile(
            segment_id=seg.segment_id,
            segment_name=seg.segment_name,
            engagement_score=seg.engagement_score,
            updated_at=seg.updated_at,
        )

    # 2. Fetch Historical Predictions
    db_predictions = (
        db.query(Prediction)
        .filter(Prediction.customer_id == customer_id)
        .order_by(Prediction.scored_at.desc())
        .all()
    )

    pred_records: List[CustomerPredictionRecord] = [
        CustomerPredictionRecord(
            id=p.id,
            churn_probability=p.churn_probability,
            is_churn=p.is_churn,
            predicted_clv=p.predicted_clv,
            risk_tier=p.risk_tier,
            model_version=p.model_version,
            scored_at=p.scored_at,
        )
        for p in db_predictions
    ]

    latest_pred = pred_records[0] if pred_records else None

    # 3. Fetch Historical Anomaly Reports
    db_anomaly = (
        db.query(AnomalyReport)
        .filter(AnomalyReport.customer_id == customer_id)
        .order_by(AnomalyReport.flagged_at.desc())
        .first()
    )
    latest_anomaly: Optional[CustomerAnomalyRecord] = None
    if db_anomaly:
        latest_anomaly = CustomerAnomalyRecord(
            id=db_anomaly.id,
            reconstruction_error=db_anomaly.reconstruction_error,
            is_anomaly_95=db_anomaly.is_anomaly_95,
            is_anomaly_99=db_anomaly.is_anomaly_99,
            flagged_at=db_anomaly.flagged_at,
        )

    # 4. Fetch Historical RFM Metrics
    svc = ModelService.get_instance()
    rfm_metrics = svc.get_rfm_metrics(customer_id)

    return CustomerDetailResponse(
        customer_id=customer.customer_id,
        country=customer.country,
        first_order_date=customer.first_order_date,
        last_order_date=customer.last_order_date,
        is_uk=customer.is_uk,
        segment=seg_profile,
        latest_prediction=latest_pred,
        latest_anomaly=latest_anomaly,
        historical_predictions=pred_records,
        historical_predictions_count=len(pred_records),
        rfm_metrics=rfm_metrics,
    )


@router.get(
    "/customers",
    response_model=List[CustomerDetailResponse],
    status_code=status.HTTP_200_OK,
    summary="List Customers Cohort",
    description="Returns a paginated list of customers with their latest predictions and segments.",
)
def list_customers(
    limit: int = Query(20, ge=1, le=200, description="Page limit"),
    offset: int = Query(0, ge=0, description="Offset starting index"),
    db: Session = Depends(get_db),
) -> List[CustomerDetailResponse]:
    """Returns paginated customer list for dashboard displays."""
    customers = db.query(Customer).order_by(Customer.customer_id.asc()).offset(offset).limit(limit).all()

    svc = ModelService.get_instance()
    results: List[CustomerDetailResponse] = []

    for c in customers:
        seg = db.query(CustomerSegment).filter(CustomerSegment.customer_id == c.customer_id).first()
        seg_profile = (
            CustomerSegmentProfile(
                segment_id=seg.segment_id,
                segment_name=seg.segment_name,
                engagement_score=seg.engagement_score,
                updated_at=seg.updated_at,
            )
            if seg
            else None
        )

        latest_db_pred = (
            db.query(Prediction)
            .filter(Prediction.customer_id == c.customer_id)
            .order_by(Prediction.scored_at.desc())
            .first()
        )
        latest_pred = (
            CustomerPredictionRecord(
                id=latest_db_pred.id,
                churn_probability=latest_db_pred.churn_probability,
                is_churn=latest_db_pred.is_churn,
                predicted_clv=latest_db_pred.predicted_clv,
                risk_tier=latest_db_pred.risk_tier,
                model_version=latest_db_pred.model_version,
                scored_at=latest_db_pred.scored_at,
            )
            if latest_db_pred
            else None
        )

        rfm = svc.get_rfm_metrics(c.customer_id)

        results.append(
            CustomerDetailResponse(
                customer_id=c.customer_id,
                country=c.country,
                first_order_date=c.first_order_date,
                last_order_date=c.last_order_date,
                is_uk=c.is_uk,
                segment=seg_profile,
                latest_prediction=latest_pred,
                historical_predictions_count=1 if latest_pred else 0,
                rfm_metrics=rfm,
            )
        )

    return results
