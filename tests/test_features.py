"""Unit and anti-leakage tests for feature engineering pipelines."""

from datetime import datetime

import pandas as pd
import pytest

from src.features.behavioral import extract_behavioral_features
from src.features.builder import build_customer_features
from src.features.engagement import calculate_engagement_score
from src.features.rfm import extract_rfm_features
from src.features.target import generate_targets
from src.features.velocity import extract_velocity_features


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    """Fixture providing transactions spanning before and after a defined cutoff."""
    return pd.DataFrame(
        {
            "invoice": ["INV01", "INV02", "INV03", "INV04", "INV05", "INV06"],
            "stock_code": ["22001", "21002", "85003", "22001", "22001", "21002"],
            "description": ["Homeware Item", "Decor Item", "Lighting Item", "Homeware Item", "Homeware Item", "Decor Item"],
            "quantity": [5, 10, 2, 8, 50, 100],
            "price": [10.0, 5.0, 20.0, 10.0, 10.0, 5.0],
            "customer_id": [101, 101, 102, 102, 101, 103],
            "invoice_date": [
                datetime(2010, 1, 15, 10, 0),   # 101 Order 1 (Pre-cutoff)
                datetime(2010, 3, 20, 11, 0),   # 101 Order 2 (Pre-cutoff)
                datetime(2010, 2, 10, 9, 0),    # 102 Order 1 (Pre-cutoff)
                datetime(2010, 4, 1, 14, 0),    # 102 Order 2 (Pre-cutoff)
                datetime(2010, 8, 15, 12, 0),   # 101 Order 3 (POST-cutoff!)
                datetime(2010, 9, 1, 16, 0),    # 103 Order 1 (POST-cutoff only!)
            ],
            "country": ["United Kingdom"] * 6,
            "total_amount": [50.0, 50.0, 40.0, 80.0, 500.0, 500.0],
            "is_return": [False] * 6,
        }
    )


def test_strict_point_in_time_leakage_prevention(synthetic_transactions: pd.DataFrame):
    """CRITICAL TEST: Verifies that post-cutoff events have ZERO impact on engineered features."""
    cutoff = datetime(2010, 6, 1, 0, 0)

    # 1. Compute features with original post-cutoff transaction (50 units, £500 spend on Aug 15)
    rfm_1 = extract_rfm_features(synthetic_transactions, cutoff_date=cutoff)
    vel_1 = extract_velocity_features(synthetic_transactions, cutoff_date=cutoff)
    beh_1 = extract_behavioral_features(synthetic_transactions, cutoff_date=cutoff)

    cust101_spend_1 = rfm_1.loc[101, "total_spend"]
    cust101_freq_1 = rfm_1.loc[101, "frequency"]
    cust101_recency_1 = rfm_1.loc[101, "recency_days"]

    # Pre-cutoff orders were Jan 15 (£50) and March 20 (£50) -> total spend should be £100
    assert cust101_spend_1 == 100.0
    assert cust101_freq_1 == 2
    # Recency must be measured from March 20 to June 1 (73 days), NOT from August 15!
    expected_recency = (cutoff - datetime(2010, 3, 20, 11, 0)).total_seconds() / 86400.0
    assert abs(cust101_recency_1 - expected_recency) < 0.01

    # Customer 103 only purchased in September (post-cutoff) -> MUST NOT be in pre-cutoff cohort!
    assert 103 not in rfm_1.index

    # 2. Modify post-cutoff data drastically (add 10,000 items and £1,000,000 to post-cutoff order)
    mutated_df = synthetic_transactions.copy()
    mutated_df.loc[mutated_df["invoice"] == "INV05", "total_amount"] = 1_000_000.0
    mutated_df.loc[mutated_df["invoice"] == "INV05", "quantity"] = 10_000

    rfm_2 = extract_rfm_features(mutated_df, cutoff_date=cutoff)
    vel_2 = extract_velocity_features(mutated_df, cutoff_date=cutoff)
    beh_2 = extract_behavioral_features(mutated_df, cutoff_date=cutoff)

    # Assert 100% equality of all feature values before and after the mutation
    pd.testing.assert_series_equal(rfm_1.loc[101], rfm_2.loc[101])
    pd.testing.assert_series_equal(vel_1.loc[101], vel_2.loc[101])
    pd.testing.assert_series_equal(beh_1.loc[101], beh_2.loc[101])


def test_target_generation_logic(synthetic_transactions: pd.DataFrame):
    """Verifies that 90-day churn and forward CLV are calculated accurately from target window."""
    cutoff = datetime(2010, 6, 1, 0, 0)
    # Target window is (2010-06-01, 2010-08-30] (90 days)
    # Customer 101 purchased on 2010-08-15 (in window) -> Churn = 0, CLV = £500
    # Customer 102 had no purchases after 2010-06-01 -> Churn = 1, CLV = £0
    targets = generate_targets(synthetic_transactions, cutoff_date=cutoff, target_window_days=90)

    assert targets.loc[101, "churn"] == 0
    assert targets.loc[101, "clv_next_90d"] == 500.0
    assert targets.loc[101, "target_orders_count"] == 1

    assert targets.loc[102, "churn"] == 1
    assert targets.loc[102, "clv_next_90d"] == 0.0
    assert targets.loc[102, "target_orders_count"] == 0


def test_rfm_basket_metrics():
    """Verifies basket sizes and return adjustments in RFM features."""
    df = pd.DataFrame(
        {
            "customer_id": [201, 201],
            "invoice": ["INV_A", "INV_B"],
            "stock_code": ["SKU1", "SKU2"],
            "quantity": [10, -2],
            "price": [5.0, 5.0],
            "total_amount": [50.0, -10.0],
            "invoice_date": [datetime(2010, 1, 1), datetime(2010, 1, 5)],
            "is_return": [False, True],
        }
    )
    rfm = extract_rfm_features(df, cutoff_date=datetime(2010, 1, 10))
    # Net spend should reflect return deduction (50 - 10 = 40)
    assert rfm.loc[201, "total_spend"] == 40.0
    # Gross spend should only reflect sales order (50)
    assert rfm.loc[201, "gross_spend"] == 50.0
    # Frequency is distinct sales invoices (1)
    assert rfm.loc[201, "frequency"] == 1
    assert rfm.loc[201, "avg_basket_size"] == 10.0


def test_engagement_score_bounds_and_tiers():
    """Verifies that composite engagement score is correctly bounded and tier-classified."""
    df_rfm = pd.DataFrame(
        {
            "recency_days": [5.0, 100.0, 400.0],
            "frequency": [20, 3, 1],
            "total_spend": [5000.0, 200.0, 20.0],
        },
        index=[1, 2, 3],
    )
    scores = calculate_engagement_score(df_rfm)
    assert (scores["engagement_score"] >= 0.0).all()
    assert (scores["engagement_score"] <= 100.0).all()
    # High engagement customer must have higher score than inactive customer
    assert scores.loc[1, "engagement_score"] > scores.loc[2, "engagement_score"]
    assert scores.loc[2, "engagement_score"] > scores.loc[3, "engagement_score"]
    assert scores.loc[1, "engagement_tier"] == "High"
    assert scores.loc[3, "engagement_tier"] == "Low"


def test_build_customer_features_end_to_end(tmp_path, synthetic_transactions: pd.DataFrame):
    """Verifies complete builder pipeline execution, parquet output, and metadata persistence."""
    cutoff = datetime(2010, 6, 1, 0, 0)
    features_df, summary = build_customer_features(
        synthetic_transactions,
        cutoff_date=cutoff,
        target_window_days=90,
        save_outputs=True,
        output_dir=str(tmp_path),
    )

    assert len(features_df) == 2  # Customers 101 and 102
    assert "churn" in features_df.columns
    assert "clv_next_90d" in features_df.columns
    assert "engagement_score" in features_df.columns
    assert (tmp_path / "customer_features.parquet").exists()
    assert (tmp_path / "customer_features.csv").exists()
    assert (tmp_path / "feature_metadata.json").exists()
    assert summary["cohort_size"] == 2
