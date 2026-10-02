"""Pydantic schemas for prediction requests and responses with field validations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CustomerFeaturesInput(BaseModel):
    """Customer features payload for single customer prediction."""

    customer_id: Optional[int] = Field(None, description="Unique customer identifier")
    recency_days: float = Field(..., ge=0.0, description="Days since last purchase")
    frequency: float = Field(..., ge=0.0, description="Number of historical orders")
    total_spend: float = Field(..., description="Net total monetary spend (£)")
    gross_spend: Optional[float] = Field(None, description="Gross spend before returns")
    total_items: Optional[float] = Field(None, ge=0.0, description="Total units purchased")
    unique_products_count: Optional[float] = Field(None, ge=0.0, description="Distinct SKUs count")
    avg_order_value: Optional[float] = Field(None, description="Average spend per order")
    avg_basket_size: Optional[float] = Field(None, ge=0.0, description="Average units per basket")
    avg_unique_products_per_order: Optional[float] = Field(None, ge=0.0)
    log_total_spend: Optional[float] = Field(None)
    log_frequency: Optional[float] = Field(None)
    tenure_days: Optional[float] = Field(None, ge=0.0, description="Days since first purchase")
    days_active_span: Optional[float] = Field(None, ge=0.0)
    inter_purchase_mean: Optional[float] = Field(None, ge=0.0)
    inter_purchase_std: Optional[float] = Field(None, ge=0.0)
    is_single_order: Optional[float] = Field(None)
    recent_orders_last_90d: Optional[float] = Field(0.0, ge=0.0)
    recent_orders_ratio: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    velocity_acceleration: Optional[float] = Field(1.0, ge=0.0)
    seasonal_q4_order_share: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    seasonal_q4_spend_share: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    return_line_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    return_spend_ratio: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    has_returned: Optional[float] = Field(0.0)
    discount_sensitivity: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    cat_affinity_22: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    cat_affinity_21: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    cat_affinity_85: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    cat_affinity_84: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    cat_affinity_23: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    cat_affinity_20: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    is_uk: Optional[int] = Field(1, ge=0, le=1)
    engagement_score: Optional[float] = Field(None, ge=0.0, le=100.0)


class SinglePredictionResponse(BaseModel):
    """Prediction results and actionable narrative for an individual customer."""

    customer_id: Optional[int]
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    is_churn: bool
    predicted_clv_90d: float = Field(..., ge=0.0)
    risk_tier: str
    segment_name: str
    is_anomaly: bool
    anomaly_score: float
    plain_language_narrative: str
    top_risk_factors: List[Dict[str, Any]]
    model_version: str
    scored_at: datetime


class BatchPredictionSummary(BaseModel):
    """Batch scoring summary and collection of individual results."""

    total_records: int
    churn_rate: float
    mean_predicted_clv: float
    anomalies_detected: int
    download_csv_payload: Optional[str] = Field(None, description="CSV formatted string of prediction results")
    download_url: Optional[str] = Field(None, description="Direct URL or endpoint to download scored batch CSV")
    predictions: List[SinglePredictionResponse]


class HealthResponse(BaseModel):
    """System health check payload."""

    status: str
    database: str
    loaded_models: List[str]
    timestamp: datetime


class ModelMetadataResponse(BaseModel):
    """Metadata describing active models, versions, and validation metrics."""

    production_churn_model: Dict[str, Any]
    clv_regression_model: Dict[str, Any]
    segmentation_model: Dict[str, Any]
    anomaly_autoencoder: Dict[str, Any]
    feature_count: int
    active_features: List[str]
