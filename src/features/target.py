"""Target variable generator for 90-day churn classification and CLV regression."""

from datetime import datetime
from typing import Optional, Set

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_targets(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_window_days: int = 90,
    observation_customers: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Generates ground truth 90-day forward churn label and CLV spend target strictly from post-cutoff transactions.

    Rules:
    - Target window: (cutoff_date, cutoff_date + target_window_days]
    - Churn = 1 if customer had 0 purchase transactions in target window, 0 if customer had >= 1 purchase.
    - CLV (clv_next_90d) = Net sum of spend in target window for the customer.

    Args:
        df: Cleaned transactions DataFrame.
        cutoff_date: The point-in-time timestamp dividing observation and target periods.
        target_window_days: Length of forward prediction window in days (default 90).
        observation_customers: Optional set of customer_ids active in the observation window.
                               If None, inferred as customers with transactions <= cutoff_date.

    Returns:
        DataFrame indexed by customer_id with columns ['churn', 'clv_next_90d', 'target_orders_count'].
    """
    if observation_customers is None:
        obs_df = df[df["invoice_date"] <= cutoff_date]
        observation_customers = set(obs_df["customer_id"].unique())

    target_window_end = cutoff_date + pd.Timedelta(days=target_window_days)
    logger.info(
        f"Generating targets for {len(observation_customers):,} customers across window: "
        f"({cutoff_date} to {target_window_end}]"
    )

    # Filter target window transactions
    target_mask = (df["invoice_date"] > cutoff_date) & (df["invoice_date"] <= target_window_end)
    target_df = df[target_mask].copy()

    # Identify customers with sales orders in the target window
    sales_target_df = target_df[~target_df["is_return"]]
    active_target_customers = set(sales_target_df["customer_id"].unique())

    # Target orders count and net spend per customer in target window
    target_orders = sales_target_df.groupby("customer_id")["invoice"].nunique().to_dict()
    target_spend = target_df.groupby("customer_id")["total_amount"].sum().to_dict()

    records = []
    for cust_id in observation_customers:
        has_purchased = cust_id in active_target_customers
        churn_label = 0 if has_purchased else 1
        spend = round(float(max(0.0, target_spend.get(cust_id, 0.0))), 2)
        orders = target_orders.get(cust_id, 0)

        records.append(
            {
                "customer_id": cust_id,
                "churn": churn_label,
                "clv_next_90d": spend,
                "target_orders_count": orders,
            }
        )

    targets_df = pd.DataFrame(records).set_index("customer_id")
    churn_rate = targets_df["churn"].mean()
    mean_clv = targets_df["clv_next_90d"].mean()

    logger.info(
        f"Target generation complete: Total Customers={len(targets_df):,}, "
        f"Churn Rate={churn_rate:.2%}, Mean CLV (90d)=£{mean_clv:.2f}"
    )
    return targets_df
