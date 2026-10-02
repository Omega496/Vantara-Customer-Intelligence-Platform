"""RFM (Recency, Frequency, Monetary) and basket size feature extractor."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_rfm_features(
    df: pd.DataFrame,
    cutoff_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Extracts point-in-time RFM and basket size features strictly before cutoff_date.

    Args:
        df: Transaction DataFrame with columns ['customer_id', 'invoice', 'stock_code',
            'quantity', 'price', 'total_amount', 'invoice_date', 'is_return'].
        cutoff_date: Timestamp representing the observation cutoff date.
                     Transactions with invoice_date > cutoff_date are excluded.

    Returns:
        DataFrame index by customer_id with engineered RFM columns.
    """
    if cutoff_date is not None:
        df = df[df["invoice_date"] <= cutoff_date].copy()
    else:
        cutoff_date = df["invoice_date"].max()

    logger.info(f"Extracting RFM features for {df['customer_id'].nunique():,} customers as of {cutoff_date}")

    sales_df = df[~df["is_return"]].copy()

    # Base aggregations per customer on sales orders
    grouped_sales = sales_df.groupby("customer_id")

    recency_series = grouped_sales["invoice_date"].max().apply(
        lambda last_dt: max(0.0, (cutoff_date - last_dt).total_seconds() / 86400.0)
    ).rename("recency_days")

    frequency_series = grouped_sales["invoice"].nunique().rename("frequency")
    gross_spend_series = grouped_sales["total_amount"].sum().rename("gross_spend")
    total_items_series = grouped_sales["quantity"].sum().rename("total_items")
    unique_skus_series = grouped_sales["stock_code"].nunique().rename("unique_products_count")

    # Net spend accounting for returns in the observation window
    net_spend_series = df.groupby("customer_id")["total_amount"].sum().rename("total_spend")

    rfm_df = pd.concat(
        [
            recency_series,
            frequency_series,
            gross_spend_series,
            net_spend_series,
            total_items_series,
            unique_skus_series,
        ],
        axis=1,
    )

    # Derived basket and monetary averages
    rfm_df["avg_order_value"] = (rfm_df["total_spend"] / rfm_df["frequency"].clip(lower=1)).round(2)
    rfm_df["avg_basket_size"] = (rfm_df["total_items"] / rfm_df["frequency"].clip(lower=1)).round(2)
    rfm_df["avg_unique_products_per_order"] = (
        rfm_df["unique_products_count"] / rfm_df["frequency"].clip(lower=1)
    ).round(2)

    # Skewness-adjusted log transformations for linear/distance models
    rfm_df["log_total_spend"] = np.log1p(rfm_df["total_spend"].clip(lower=0)).round(4)
    rfm_df["log_frequency"] = np.log1p(rfm_df["frequency"]).round(4)

    return rfm_df
