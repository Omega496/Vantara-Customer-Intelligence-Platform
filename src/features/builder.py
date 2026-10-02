"""Feature engineering orchestration pipeline with strict point-in-time cutoff discipline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.features.behavioral import extract_behavioral_features
from src.features.engagement import calculate_engagement_score
from src.features.rfm import extract_rfm_features
from src.features.target import generate_targets
from src.features.velocity import extract_velocity_features
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_DICTIONARY = {
    "recency_days": "Days since last purchase as of cutoff date. Primary univariate churn predictor.",
    "frequency": "Distinct completed order count in observation window. Separates habitual from one-time buyers.",
    "total_spend": "Net monetary spend in observation window. Key input for value tiering and CLV modeling.",
    "gross_spend": "Gross spend before accounting for return deductions.",
    "total_items": "Total quantity of merchandise items purchased.",
    "unique_products_count": "Number of unique SKUs purchased across all orders.",
    "avg_order_value": "Mean spend per order. Indicates purchase ticket magnitude.",
    "avg_basket_size": "Mean item quantity per order. Distinguishes bulk wholesale buyers from retail consumers.",
    "avg_unique_products_per_order": "Variety of catalog items purchased per order.",
    "log_total_spend": "Log-transformed spend to reduce right-skewness for linear/neural models.",
    "log_frequency": "Log-transformed order frequency to normalize counts.",
    "tenure_days": "Days since customer's very first order. Represents customer relationship length.",
    "days_active_span": "Days between first and last purchase in observation window.",
    "inter_purchase_mean": "Average days between consecutive orders.",
    "inter_purchase_std": "Standard deviation of inter-purchase intervals; erratic timing increases churn unpredictability.",
    "is_single_order": "Binary indicator for one-and-done buyers (frequency == 1).",
    "recent_orders_last_90d": "Orders placed within the 90 days immediately preceding the cutoff date.",
    "recent_orders_ratio": "Fraction of lifetime orders occurring in recent 90-day observation window.",
    "velocity_acceleration": "Ratio of recent purchase pace vs expected tenure pace; <1 indicates deceleration.",
    "seasonal_q4_order_share": "Proportion of orders occurring in Q4 (holiday gift concentration).",
    "seasonal_q4_spend_share": "Proportion of total spend occurring in Q4.",
    "return_line_rate": "Fraction of transaction lines representing returns/cancellations.",
    "return_spend_ratio": "Ratio of return value to gross spend.",
    "has_returned": "Binary flag indicating whether customer has ever returned an item.",
    "discount_sensitivity": "Fraction of items bought at >=10% below median catalog price.",
    "cat_affinity_22": "Spend share in Department 22 (Homeware & Kitchen).",
    "cat_affinity_21": "Spend share in Department 21 (Decor & Party Supplies).",
    "cat_affinity_85": "Spend share in Department 85 (Lighting & Storage Bags).",
    "cat_affinity_84": "Spend share in Department 84 (Ornaments & Craftware).",
    "cat_affinity_23": "Spend share in Department 23 (Novelty & Accessories).",
    "cat_affinity_20": "Spend share in Department 20 (Stationery & Gifts).",
    "is_uk": "Binary flag indicating domestic UK buyer.",
    "engagement_score": "Composite 0-100 score balancing recency, frequency, and monetary percentiles.",
    "engagement_tier": "Customer health tier (Low, Medium, High).",
    "churn": "Target: 1 if customer placed 0 orders in the subsequent 90 days, 0 if retained.",
    "clv_next_90d": "Target: Net forward monetary spend in the subsequent 90-day window.",
}


def build_customer_features(
    df: pd.DataFrame,
    cutoff_date: Optional[datetime] = None,
    target_window_days: int = 90,
    save_outputs: bool = True,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Builds unified customer-level feature dataset with strict point-in-time cutoff.

    Args:
        df: Cleaned transaction DataFrame.
        cutoff_date: Point-in-time cutoff timestamp. If None, derived as max(date) - 90 days.
        target_window_days: Forward prediction window in days (default 90).
        save_outputs: If True, writes processed parquet, csv, and metadata files.
        output_dir: Output directory for processed dataset.

    Returns:
        Tuple of (Full customer features DataFrame, summary dictionary).
    """
    logger.info("==================================================")
    logger.info("Starting Milestone 2: Feature Engineering Pipeline")
    logger.info("==================================================")

    max_dataset_date = df["invoice_date"].max()
    if cutoff_date is None:
        cutoff_date = max_dataset_date - pd.Timedelta(days=target_window_days)

    logger.info(f"Observation window ends at cutoff: {cutoff_date}")
    logger.info(f"Target window spans: {cutoff_date} to {cutoff_date + pd.Timedelta(days=target_window_days)}")

    # 1. Observation Window Transactions (Strict Anti-Leakage Filter)
    obs_df = df[df["invoice_date"] <= cutoff_date].copy()
    if obs_df.empty:
        raise ValueError(f"No transactions found on or before cutoff date: {cutoff_date}")

    obs_customers = set(obs_df["customer_id"].unique())
    logger.info(f"Active observation cohort contains {len(obs_customers):,} unique customers.")

    # 2. Extract Feature Subsets
    rfm_df = extract_rfm_features(obs_df, cutoff_date=cutoff_date)
    velocity_df = extract_velocity_features(obs_df, cutoff_date=cutoff_date)
    behavioral_df = extract_behavioral_features(obs_df, cutoff_date=cutoff_date)
    engagement_df = calculate_engagement_score(rfm_df)

    # 3. Extract Forward Targets
    target_df = generate_targets(
        df=df,
        cutoff_date=cutoff_date,
        target_window_days=target_window_days,
        observation_customers=obs_customers,
    )

    # 4. Join all feature modules
    feature_dfs = [rfm_df, velocity_df, behavioral_df, engagement_df, target_df]
    merged_df = pd.concat(feature_dfs, axis=1)

    # Clean and fill any potential NaNs in engineered features
    numeric_cols = merged_df.select_dtypes(include=["number"]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0.0)

    # Validation: ensure customer IDs match observation customers exactly
    assert len(merged_df) == len(obs_customers), "Mismatch between customer cohort and features!"
    assert merged_df["churn"].isnull().sum() == 0, "Target 'churn' has null values!"

    summary = {
        "cutoff_date": str(cutoff_date),
        "target_window_days": target_window_days,
        "cohort_size": len(merged_df),
        "feature_count": merged_df.shape[1],
        "churn_rate": float(merged_df["churn"].mean()),
        "retained_count": int((merged_df["churn"] == 0).sum()),
        "churned_count": int((merged_df["churn"] == 1).sum()),
        "mean_clv_90d": float(merged_df["clv_next_90d"].mean()),
        "median_clv_90d": float(merged_df["clv_next_90d"].median()),
        "max_clv_90d": float(merged_df["clv_next_90d"].max()),
        "mean_engagement_score": float(merged_df["engagement_score"].mean()),
    }

    # 5. Save Processed Artifacts
    if save_outputs:
        if output_dir is None:
            output_dir = ConfigManager.get("paths.processed_data_dir", "data/processed")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        parquet_file = out_path / "customer_features.parquet"
        csv_file = out_path / "customer_features.csv"
        meta_file = out_path / "feature_metadata.json"

        logger.info(f"Saving customer features to {parquet_file} and {csv_file}...")
        # Reset index so customer_id is a column in the saved file
        export_df = merged_df.reset_index().rename(columns={"index": "customer_id"})
        export_df.to_parquet(parquet_file, index=False)
        export_df.to_csv(csv_file, index=False)

        metadata_payload = {
            "summary": summary,
            "feature_dictionary": FEATURE_DICTIONARY,
            "columns": list(export_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in export_df.dtypes.items()},
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        logger.info(f"Saved feature metadata to {meta_file}")

    logger.info("Milestone 2 Feature Engineering complete!")
    logger.info("==================================================")
    return merged_df, summary


if __name__ == "__main__":
    cleaned_parquet = ConfigManager.get(
        "paths.cleaned_transactions_file", "data/interim/cleaned_transactions.parquet"
    )
    df_transactions = pd.read_parquet(cleaned_parquet)
    build_customer_features(df_transactions)
