"""Velocity, purchase interval variance, and engagement trend feature extractor."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_velocity_features(
    df: pd.DataFrame,
    cutoff_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Extracts inter-purchase timing variance, tenure, and purchase velocity trends strictly before cutoff_date.

    Args:
        df: Cleaned transaction DataFrame.
        cutoff_date: Observation window cutoff timestamp.

    Returns:
        DataFrame indexed by customer_id containing velocity and trend metrics.
    """
    if cutoff_date is not None:
        df = df[df["invoice_date"] <= cutoff_date].copy()
    else:
        cutoff_date = df["invoice_date"].max()

    logger.info(f"Extracting velocity features as of cutoff {cutoff_date}")
    sales_df = df[~df["is_return"]].copy()

    # Get distinct order timestamps per customer
    orders = (
        sales_df[["customer_id", "invoice", "invoice_date"]]
        .drop_duplicates(subset=["customer_id", "invoice"])
        .sort_values(by=["customer_id", "invoice_date"])
    )

    records = []
    recent_window_start = cutoff_date - pd.Timedelta(days=90)

    for cust_id, group in orders.groupby("customer_id"):
        order_dates = group["invoice_date"].tolist()
        num_orders = len(order_dates)

        first_date = order_dates[0]
        last_date = order_dates[-1]

        tenure_days = max(1.0, (cutoff_date - first_date).total_seconds() / 86400.0)
        days_active_span = max(0.0, (last_date - first_date).total_seconds() / 86400.0)

        # Inter-purchase interval calculations
        if num_orders > 1:
            intervals = [
                (order_dates[i] - order_dates[i - 1]).total_seconds() / 86400.0
                for i in range(1, num_orders)
            ]
            inter_purchase_mean = float(np.mean(intervals))
            inter_purchase_std = float(np.std(intervals)) if len(intervals) > 1 else 0.0
            is_single_order = 0
        else:
            inter_purchase_mean = tenure_days
            inter_purchase_std = 0.0
            is_single_order = 1

        # Recent velocity: orders in the last 90 days of observation
        recent_orders = sum(1 for d in order_dates if d >= recent_window_start)
        recent_order_ratio = recent_orders / num_orders

        # Velocity ratio: actual recent order rate vs annualized expected rate
        expected_recent_rate = (num_orders / tenure_days) * 90.0
        velocity_acceleration = (
            recent_orders / max(expected_recent_rate, 0.1)
            if expected_recent_rate > 0
            else 0.0
        )

        records.append(
            {
                "customer_id": cust_id,
                "tenure_days": round(tenure_days, 1),
                "days_active_span": round(days_active_span, 1),
                "inter_purchase_mean": round(inter_purchase_mean, 2),
                "inter_purchase_std": round(inter_purchase_std, 2),
                "is_single_order": is_single_order,
                "recent_orders_last_90d": recent_orders,
                "recent_orders_ratio": round(recent_order_ratio, 4),
                "velocity_acceleration": round(float(np.clip(velocity_acceleration, 0.0, 10.0)), 4),
            }
        )

    velocity_df = pd.DataFrame(records).set_index("customer_id")
    return velocity_df
