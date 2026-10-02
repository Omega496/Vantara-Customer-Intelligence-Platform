"""Customer Lifetime Value (CLV) regression models predicting 90-day forward spend."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.data_split import DataSplits, prepare_data_splits
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_regressor(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    split_name: str = "Test",
    is_log_target: bool = False,
) -> Dict[str, float]:
    """Evaluates regression model on original monetary scale.

    Args:
        model: Trained regressor.
        X: Feature matrix.
        y_true: True ground truth continuous target values.
        split_name: Split name ('Val' or 'Test').
        is_log_target: If True, model predictions are exponentiated (expm1).

    Returns:
        Dictionary of MAE, RMSE, and R2 metrics.
    """
    y_pred = model.predict(X)
    if is_log_target:
        y_pred = np.expm1(np.clip(y_pred, 0, 15))

    # Clip negative predictions to 0 for monetary spend
    y_pred = np.clip(y_pred, 0.0, None)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {
        f"{split_name}_MAE": round(mae, 2),
        f"{split_name}_RMSE": round(rmse, 2),
        f"{split_name}_R2": round(r2, 4),
    }


def train_and_evaluate_clv_models(
    splits: DataSplits,
    artifacts_dir: Optional[str] = None,
    save_models: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Trains Ridge, Random Forest, and LightGBM regressors predicting 90-day forward spend.

    Args:
        splits: Prepared train, val, and test splits with target clv_next_90d.
        artifacts_dir: Directory to save model artifacts.
        save_models: If True, saves model binaries and metrics report.

    Returns:
        Tuple of (Comparison DataFrame, Dictionary of trained models).
    """
    logger.info("Training and evaluating CLV regression models...")
    random_seed = int(ConfigManager.get("project.random_seed", 42))

    if artifacts_dir is None:
        artifacts_dir = ConfigManager.get("paths.models_dir", "models_artifacts")
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # Dictionary of regression candidates: (model, requires_scaling, log_target)
    regressors: Dict[str, Tuple[Any, bool, bool]] = {
        "Ridge Regressor": (
            Ridge(alpha=100.0, random_state=random_seed),
            True,
            False,
        ),
        "Random Forest Regressor": (
            RandomForestRegressor(
                n_estimators=150,
                max_depth=6,
                min_samples_leaf=5,
                random_state=random_seed,
                n_jobs=-1,
            ),
            False,
            False,
        ),
        "LightGBM Regressor": (
            LGBMRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_seed,
                verbose=-1,
            ),
            False,
            False,
        ),
        "LightGBM (Log-Target)": (
            LGBMRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_seed,
                verbose=-1,
            ),
            False,
            True,
        ),
    }

    comparison_records: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for name, (model, requires_scaling, log_target) in regressors.items():
        logger.info(f"Training CLV model {name}...")

        X_train_curr = splits.X_train_scaled if requires_scaling else splits.X_train
        X_val_curr = splits.X_val_scaled if requires_scaling else splits.X_val
        X_test_curr = splits.X_test_scaled if requires_scaling else splits.X_test

        y_train_curr = np.log1p(splits.y_train) if log_target else splits.y_train

        model.fit(X_train_curr, y_train_curr)

        val_metrics = evaluate_regressor(
            model, X_val_curr, splits.y_val, split_name="Val", is_log_target=log_target
        )
        test_metrics = evaluate_regressor(
            model, X_test_curr, splits.y_test, split_name="Test", is_log_target=log_target
        )

        record = {"Model": name, **test_metrics, **val_metrics}
        comparison_records.append(record)
        trained_models[name] = model

        logger.info(
            f"{name} -> Test R2: {test_metrics['Test_R2']:.4f}, "
            f"Test MAE: £{test_metrics['Test_MAE']:.2f}, "
            f"Test RMSE: £{test_metrics['Test_RMSE']:.2f}"
        )

        if save_models:
            slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            joblib.dump(model, artifacts_path / f"clv_{slug}_model.joblib")

    comp_df = pd.DataFrame(comparison_records).sort_values(
        by="Test_R2", ascending=False
    ).reset_index(drop=True)

    best_model_name = comp_df.iloc[0]["Model"]
    logger.info(f"Top performing CLV model: {best_model_name} (R2 = {comp_df.iloc[0]['Test_R2']:.4f})")

    if save_models:
        best_model_obj = trained_models[best_model_name]
        joblib.dump(best_model_obj, artifacts_path / "clv_best_model.joblib")

        meta_info = {
            "best_model_name": best_model_name,
            "is_log_target": regressors[best_model_name][2],
            "metrics": comp_df.iloc[0].to_dict(),
        }
        with open(artifacts_path / "clv_best_model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)

        comp_df.to_json(artifacts_path / "clv_model_comparison.json", orient="records", indent=2)
        comp_df.to_json(Path("docs") / "clv_model_comparison.json", orient="records", indent=2)

    return comp_df, trained_models


if __name__ == "__main__":
    features_path = ConfigManager.get(
        "paths.customer_features_file", "data/processed/customer_features.parquet"
    )
    df_features = pd.read_parquet(features_path)
    splits = prepare_data_splits(df_features, target_col="clv_next_90d")
    train_and_evaluate_clv_models(splits)
