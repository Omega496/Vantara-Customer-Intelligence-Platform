"""Exploratory Data Analysis (EDA) module for Online Retail II transaction dataset."""

import json
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


def perform_eda(
    df: pd.DataFrame,
    output_dir: str = "docs/figures",
    save_summary_json: bool = True,
) -> Dict[str, any]:
    """Generates univariate, bivariate, country, and temporal distributions with saved visualizations.

    Args:
        df: Cleaned transaction DataFrame containing 'invoice_date', 'customer_id',
            'quantity', 'price', 'total_amount', 'country', 'is_return'.
        output_dir: Directory to save figure plots.
        save_summary_json: Whether to write summary metrics to a JSON file.

    Returns:
        Dictionary of EDA findings and statistics.
    """
    logger.info("Performing Exploratory Data Analysis (EDA)...")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Style settings
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.autolayout"] = True

    # 1. Temporal Analysis (Monthly Revenue and Invoices)
    df_sales = df[~df["is_return"]].copy()
    df_sales["year_month"] = df_sales["invoice_date"].dt.to_period("M").astype(str)

    monthly_stats = (
        df_sales.groupby("year_month")
        .agg(
            revenue=("total_amount", "sum"),
            order_count=("invoice", "nunique"),
            active_customers=("customer_id", "nunique"),
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    sns.barplot(data=monthly_stats, x="year_month", y="revenue", ax=ax1, color="#3498db", alpha=0.7)
    sns.lineplot(
        data=monthly_stats,
        x="year_month",
        y="active_customers",
        ax=ax2,
        color="#e74c3c",
        marker="o",
        linewidth=2,
    )
    ax1.set_title("Monthly Revenue & Active Customers Trend (2009 - 2011)", fontsize=14, pad=12)
    ax1.set_xlabel("Year-Month", fontsize=11)
    ax1.set_ylabel("Total Revenue (£)", color="#3498db", fontsize=11)
    ax2.set_ylabel("Active Customers", color="#e74c3c", fontsize=11)
    ax1.tick_params(axis="x", rotation=45)
    fig.savefig(out_dir / "eda_monthly_revenue_trend.png", dpi=150)
    plt.close(fig)

    # 2. Country-level Distribution (Top 10 Countries by Revenue)
    country_revenue = (
        df_sales.groupby("country")["total_amount"]
        .sum()
        .reset_index()
        .sort_values(by="total_amount", ascending=False)
    )

    top_countries = country_revenue.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_countries, x="total_amount", y="country", ax=ax, palette="Blues_r")
    ax.set_title("Top 10 Markets by Total Revenue (£)", fontsize=14, pad=12)
    ax.set_xlabel("Revenue (£)", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"£{width:,.0f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )
    fig.savefig(out_dir / "eda_country_revenue.png", dpi=150)
    plt.close(fig)

    # 3. Customer-level RFM Distributions
    ref_date = df["invoice_date"].max()
    cust_rfm = (
        df_sales.groupby("customer_id")
        .agg(
            recency=("invoice_date", lambda x: (ref_date - x.max()).days),
            frequency=("invoice", "nunique"),
            monetary=("total_amount", "sum"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(cust_rfm["recency"], bins=40, kde=True, ax=axes[0], color="#2ecc71")
    axes[0].set_title("Customer Recency Distribution (Days)")
    axes[0].set_xlabel("Days Since Last Purchase")

    sns.histplot(cust_rfm["frequency"], bins=40, kde=True, ax=axes[1], color="#e67e22")
    axes[1].set_title("Customer Frequency Distribution (Orders)")
    axes[1].set_xlabel("Distinct Orders")
    axes[1].set_xlim(0, 50)

    # Log-monetary for skewness handling
    log_spend = np.log1p(cust_rfm["monetary"].clip(lower=0))
    sns.histplot(log_spend, bins=40, kde=True, ax=axes[2], color="#9b59b6")
    axes[2].set_title("Log(1 + Total Spend) Distribution")
    axes[2].set_xlabel("Log Spend")

    fig.savefig(out_dir / "eda_rfm_distributions.png", dpi=150)
    plt.close(fig)

    # 4. Return Rate Analysis
    total_cust_trans = df.groupby("customer_id").size()
    total_cust_returns = df[df["is_return"]].groupby("customer_id").size()
    cust_return_rate = (total_cust_returns / total_cust_trans).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(cust_return_rate, bins=30, kde=False, ax=ax, color="#e74c3c")
    ax.set_title("Distribution of Customer Return Rates", fontsize=14, pad=12)
    ax.set_xlabel("Return Rate (Returns / Total Transactions)")
    ax.set_ylabel("Customer Count")
    fig.savefig(out_dir / "eda_return_rate_distribution.png", dpi=150)
    plt.close(fig)

    findings = {
        "total_active_customers": int(cust_rfm["customer_id"].nunique()),
        "median_recency_days": float(cust_rfm["recency"].median()),
        "mean_recency_days": float(cust_rfm["recency"].mean()),
        "median_frequency_orders": float(cust_rfm["frequency"].median()),
        "mean_frequency_orders": float(cust_rfm["frequency"].mean()),
        "median_spend": float(cust_rfm["monetary"].median()),
        "mean_spend": float(cust_rfm["monetary"].mean()),
        "uk_revenue_share": float(
            country_revenue.loc[country_revenue["country"] == "United Kingdom", "total_amount"].sum()
            / country_revenue["total_amount"].sum()
        ),
        "hypotheses": [
            "H1: Recency is the primary predictor of customer churn; customers inactive for >90 days exhibit near-zero return likelihood.",
            "H2: One-time purchasers (frequency=1) account for ~30-40% of the customer base and require distinct retention treatment.",
            "H3: Return rate correlates positively with churn, acting as an early dissatisfaction signal.",
            "H4: Customer spend follows a heavy right-skewed Pareto distribution (top 20% generate ~80% of revenue), justifying CLV prioritization.",
        ],
    }

    if save_summary_json:
        with open(out_dir / "eda_findings.json", "w", encoding="utf-8") as f:
            json.dump(findings, f, indent=2)

    logger.info(f"EDA successfully completed. Visualizations saved to {out_dir}")
    return findings
