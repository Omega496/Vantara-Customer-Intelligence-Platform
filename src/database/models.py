"""SQLAlchemy ORM models for PostgreSQL/SQLite customer intelligence persistence."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Customer(Base):
    """Customer profile record."""

    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True, index=True)
    country = Column(String(100), default="United Kingdom", nullable=False)
    first_order_date = Column(DateTime, nullable=True)
    last_order_date = Column(DateTime, nullable=True)
    is_uk = Column(Boolean, default=True, nullable=False)

    segment = relationship("CustomerSegment", back_populates="customer", uselist=False)
    predictions = relationship("Prediction", back_populates="customer", cascade="all, delete-orphan")
    anomalies = relationship("AnomalyReport", back_populates="customer", cascade="all, delete-orphan")


class CustomerSegment(Base):
    """Assigned marketing persona and health score for a customer."""

    __tablename__ = "customer_segments"

    customer_id = Column(Integer, ForeignKey("customers.customer_id", ondelete="CASCADE"), primary_key=True)
    segment_id = Column(Integer, nullable=False)
    segment_name = Column(String(100), nullable=False)
    engagement_score = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    customer = relationship("Customer", back_populates="segment")


class Prediction(Base):
    """Scored prediction audit record containing churn probability and forward CLV."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    churn_probability = Column(Float, nullable=False)
    is_churn = Column(Boolean, nullable=False)
    predicted_clv = Column(Float, nullable=False)
    risk_tier = Column(String(50), nullable=False)
    model_version = Column(String(50), default="v1.0.0", nullable=False)
    scored_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    customer = relationship("Customer", back_populates="predictions")


class AnomalyReport(Base):
    """Autoencoder reconstruction error and anomaly status."""

    __tablename__ = "anomaly_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    reconstruction_error = Column(Float, nullable=False)
    is_anomaly_95 = Column(Boolean, default=False, nullable=False)
    is_anomaly_99 = Column(Boolean, default=False, nullable=False)
    flagged_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    customer = relationship("Customer", back_populates="anomalies")
