"""Data splitting, stratified partitioning, and scaling utilities for ML models."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXCLUDE_COLUMNS = [
    "customer_id",
    "primary_country",
    "engagement_tier",
    "churn",
    "clv_next_90d",
    "target_orders_count",
]


@dataclass
class DataSplits:
    """Holds train, validation, and test datasets along with feature names and scalers."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler


def prepare_data_splits(
    df: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: Optional[int] = None,
    features: Optional[List[str]] = None,
) -> DataSplits:
    """Performs stratified 70/15/15 train/val/test split and fits StandardScaler on training data only.

    Args:
        df: Customer features DataFrame.
        target_col: Name of target column ('churn' or 'clv_next_90d').
        test_size: Fraction of data for test set (default 0.15).
        val_size: Fraction of data for validation set (default 0.15).
        random_seed: Random seed for reproducibility (default from config or 42).
        features: Optional subset of feature column names.

    Returns:
        DataSplits dataclass containing splits, scaled matrices, and fitted scaler.
    """
    if random_seed is None:
        random_seed = int(ConfigManager.get("project.random_seed", 42))

    if features is None:
        features = [col for col in df.columns if col not in EXCLUDE_COLUMNS]

    logger.info(f"Preparing splits for target '{target_col}' using {len(features)} features.")

    X = df[features].copy()
    y = df[target_col].copy()

    # Determine if stratified split should be applied (classification vs regression)
    is_classification = target_col == "churn" or len(np.unique(y)) <= 5
    stratify_target = y if is_classification else None

    # Step 1: Split into Train+Val (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_target,
    )

    # Step 2: Split Train+Val into Train (70% of total) and Val (15% of total)
    # val_fraction = 0.15 / (1.0 - 0.15) = 0.17647
    val_fraction = val_size / (1.0 - test_size)
    stratify_train_val = y_train_val if is_classification else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=random_seed,
        stratify=stratify_train_val,
    )

    logger.info(
        f"Split sizes: Train={len(X_train)} ({len(X_train)/len(df):.1%}), "
        f"Val={len(X_val)} ({len(X_val)/len(df):.1%}), "
        f"Test={len(X_test)} ({len(X_test)/len(df):.1%})"
    )

    if is_classification:
        logger.info(
            f"Class balance: Train={y_train.mean():.2%}, Val={y_val.mean():.2%}, Test={y_test.mean():.2%}"
        )

    # Step 3: Fit scaler strictly on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        feature_names=features,
        scaler=scaler,
    )
