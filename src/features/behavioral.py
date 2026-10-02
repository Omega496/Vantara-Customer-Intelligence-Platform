"""Behavioral, seasonality, return rate, category affinity, and discount sensitivity feature extractor."""

from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_CATEGORIES = ["22", "21", "85", "84", "23", "20"]


def extract_behavioral_features(
    df: pd.DataFrame,
    cutoff_date: Optional[datetime] = None,
    top_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extracts seasonality, returns, category affinity vectors, and discount sensitivity features strictly before cutoff_date.

    Args:
        df: Cleaned transaction DataFrame.
        cutoff_date: Observation window cutoff timestamp.
        top_categories: List of 2-character stock code prefixes to track.

    Returns:
        DataFrame indexed by customer_id containing behavioral metrics.
    """
    if cutoff_date is not None:
        df = df[df["invoice_date"] <= cutoff_date].copy()
    else:
        cutoff_date = df["invoice_date"].max()

    if top_categories is None:
        top_categories = TOP_CATEGORIES

    logger.info(f"Extracting behavioral features for {df['customer_id'].nunique():,} customers as of {cutoff_date}")

    # 1. Seasonality (Q4 Holiday Shopping Concentration)
    df["is_q4"] = df["invoice_date"].dt.month.isin([10, 11, 12])
    sales_df = df[~df["is_return"]].copy()

    total_orders = sales_df.groupby("customer_id")["invoice"].nunique()
    q4_orders = sales_df[sales_df["is_q4"]].groupby("customer_id")["invoice"].nunique()
    q4_order_share = (q4_orders / total_orders).fillna(0.0).rename("seasonal_q4_order_share")

    total_spend = sales_df.groupby("customer_id")["total_amount"].sum()
    q4_spend = sales_df[sales_df["is_q4"]].groupby("customer_id")["total_amount"].sum()
    q4_spend_share = (q4_spend / total_spend).fillna(0.0).clip(0.0, 1.0).rename("seasonal_q4_spend_share")

    # 2. Return Behavior
    cust_total_lines = df.groupby("customer_id").size()
    cust_return_lines = df[df["is_return"]].groupby("customer_id").size()
    return_line_rate = (cust_return_lines / cust_total_lines).fillna(0.0).rename("return_line_rate")

    cust_returns_amount = df[df["is_return"]].groupby("customer_id")["total_amount"].sum().abs()
    cust_gross_amount = sales_df.groupby("customer_id")["total_amount"].sum()
    return_spend_ratio = (
        (cust_returns_amount / cust_gross_amount.clip(lower=1.0)).fillna(0.0).clip(0.0, 1.0).rename("return_spend_ratio")
    )
    has_returned = (cust_return_lines > 0).astype(int).fillna(0).rename("has_returned")

    # 3. Discount & Markdown Sensitivity
    # Compute median reference price for each product in observation window
    product_median_price = sales_df.groupby("stock_code")["price"].median()
    sales_df["ref_price"] = sales_df["stock_code"].map(product_median_price)
    # A purchase is discounted if unit price was at least 10% below median catalog price
    sales_df["is_discounted"] = sales_df["price"] < (sales_df["ref_price"] * 0.90)

    cust_sales_lines = sales_df.groupby("customer_id").size()
    cust_discount_lines = sales_df[sales_df["is_discounted"]].groupby("customer_id").size()
    discount_sensitivity = (
        (cust_discount_lines / cust_sales_lines).fillna(0.0).round(4).rename("discount_sensitivity")
    )

    # 4. Product Category Affinity Vector
    sales_df["cat_prefix"] = sales_df["stock_code"].astype(str).str[:2]
    cat_spend = sales_df.groupby(["customer_id", "cat_prefix"])["total_amount"].sum().unstack(fill_value=0.0)

    # Compute affinity proportion for each tracked category
    affinity_dfs = []
    cust_total_cat_spend = cat_spend.sum(axis=1).clip(lower=1.0)
    for cat in top_categories:
        if cat in cat_spend.columns:
            col = (cat_spend[cat] / cust_total_cat_spend).round(4).rename(f"cat_affinity_{cat}")
        else:
            col = pd.Series(0.0, index=cat_spend.index, name=f"cat_affinity_{cat}")
        affinity_dfs.append(col)

    cat_affinity_df = pd.concat(affinity_dfs, axis=1)

    # 5. Primary Country & UK Domestic Flag
    primary_country = df.groupby("customer_id")["country"].agg(
        lambda s: s.mode().iloc[0] if not s.empty else "United Kingdom"
    ).rename("primary_country")
    is_uk = (primary_country == "United Kingdom").astype(int).rename("is_uk")

    behavioral_df = pd.concat(
        [
            q4_order_share,
            q4_spend_share,
            return_line_rate,
            return_spend_ratio,
            has_returned,
            discount_sensitivity,
            cat_affinity_df,
            primary_country,
            is_uk,
        ],
        axis=1,
    )

    return behavioral_df
