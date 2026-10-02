"""Composite engagement score and customer health tier calculator."""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_engagement_score(
    df: pd.DataFrame,
    recency_weight: Optional[float] = None,
    frequency_weight: Optional[float] = None,
    monetary_weight: Optional[float] = None,
) -> pd.DataFrame:
    """Calculates composite engagement score (0-100) and health tiers from RFM percentiles.

    Args:
        df: DataFrame containing at least ['recency_days', 'frequency', 'total_spend'].
        recency_weight: Weight assigned to recency percentile (default 0.35).
        frequency_weight: Weight assigned to frequency percentile (default 0.35).
        monetary_weight: Weight assigned to monetary percentile (default 0.30).

    Returns:
        DataFrame with 'engagement_score' (float 0-100) and 'engagement_tier' (str).
    """
    if recency_weight is None:
        recency_weight = float(ConfigManager.get("features.engagement_score.recency_weight", 0.35))
    if frequency_weight is None:
        frequency_weight = float(ConfigManager.get("features.engagement_score.frequency_weight", 0.35))
    if monetary_weight is None:
        monetary_weight = float(ConfigManager.get("features.engagement_score.monetary_weight", 0.30))

    # Normalize weights so they sum to 1.0
    total_w = recency_weight + frequency_weight + monetary_weight
    w_r = recency_weight / total_w
    w_f = frequency_weight / total_w
    w_m = monetary_weight / total_w

    # Percentile ranks: lower recency is better (invert it)
    recency_pct = 1.0 - (df["recency_days"].rank(pct=True, ascending=True))
    frequency_pct = df["frequency"].rank(pct=True, ascending=True)
    monetary_pct = df["total_spend"].clip(lower=0).rank(pct=True, ascending=True)

    engagement_score = (
        100.0 * (w_r * recency_pct + w_f * frequency_pct + w_m * monetary_pct)
    ).round(2)

    # Classify into actionable marketing tiers
    engagement_tier = pd.cut(
        engagement_score,
        bins=[-np.inf, 40.0, 70.0, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    res_df = pd.DataFrame(
        {
            "engagement_score": engagement_score,
            "engagement_tier": engagement_tier,
        },
        index=df.index,
    )

    logger.info(
        f"Computed engagement scores: Mean={engagement_score.mean():.2f}, "
        f"Tiers: {engagement_tier.value_counts().to_dict()}"
    )
    return res_df
