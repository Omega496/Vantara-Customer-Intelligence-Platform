"""Pydantic schemas for customer retrieval and profile endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RFMMetricsSchema(BaseModel):
    """Historical RFM and behavioral metrics for a customer."""

    recency_days: float = Field(..., description="Days since last transaction")
    frequency: float = Field(..., description="Total lifetime order count")
    total_spend: float = Field(..., description="Total monetary spend (£)")
    avg_order_value: Optional[float] = Field(None, description="Average monetary spend per order (£)")
    avg_basket_size: Optional[float] = Field(None, description="Average units per order")
    tenure_days: Optional[float] = Field(None, description="Customer relationship age in days")
    engagement_score: Optional[float] = Field(None, description="Composite engagement score (0-100)")
    return_rate: Optional[float] = Field(None, description="Historical return line rate")


class CustomerSegmentProfile(BaseModel):
    """Current assigned customer segment and persona."""

    segment_id: int
    segment_name: str
    engagement_score: float
    updated_at: datetime


class CustomerPredictionRecord(BaseModel):
    """Persisted prediction record."""

    id: int
    churn_probability: float
    is_churn: bool
    predicted_clv: float
    risk_tier: str
    model_version: str
    scored_at: datetime


class CustomerAnomalyRecord(BaseModel):
    """Persisted autoencoder reconstruction anomaly report."""

    id: int
    reconstruction_error: float
    is_anomaly_95: bool
    is_anomaly_99: bool
    flagged_at: datetime


class CustomerDetailResponse(BaseModel):
    """Comprehensive customer profile including master data, segments, predictions, and RFM metrics."""

    customer_id: int
    country: str
    first_order_date: Optional[datetime] = None
    last_order_date: Optional[datetime] = None
    is_uk: bool
    segment: Optional[CustomerSegmentProfile] = None
    latest_prediction: Optional[CustomerPredictionRecord] = None
    latest_anomaly: Optional[CustomerAnomalyRecord] = None
    historical_predictions: List[CustomerPredictionRecord] = []
    historical_predictions_count: int = 0
    rfm_metrics: Optional[Dict[str, Any]] = None
