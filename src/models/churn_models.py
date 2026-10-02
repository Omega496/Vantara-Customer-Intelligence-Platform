"""Classical Machine Learning models for Customer Churn Prediction."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.models.data_split import DataSplits, prepare_data_splits
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_classifier(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    split_name: str = "Test",
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.

    Args:
        model: Trained classifier with predict and predict_proba methods.
        X: Feature matrix.
        y: True binary labels (0 or 1).
        split_name: Name of split (e.g. 'Validation' or 'Test').

    Returns:
        Dictionary of computed evaluation metrics.
    """
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)
    else:
        y_proba = y_pred

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    auc = float(roc_auc_score(y, y_proba))
    cm = confusion_matrix(y, y_pred).tolist()

    metrics = {
        f"{split_name}_Accuracy": round(acc, 4),
        f"{split_name}_Precision": round(prec, 4),
        f"{split_name}_Recall": round(rec, 4),
        f"{split_name}_F1": round(f1, 4),
        f"{split_name}_ROC_AUC": round(auc, 4),
        f"{split_name}_Confusion_Matrix": cm,
    }
    return metrics


def train_and_evaluate_churn_models(
    splits: DataSplits,
    artifacts_dir: Optional[str] = None,
    save_models: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Trains 6 classical classifiers, evaluates on Validation and Test sets, and saves model artifacts.

    Models:
    1. Logistic Regression (L2 penalty, balanced weights, scaled)
    2. Decision Tree (balanced weights, unscaled)
    3. Random Forest (tuned ensemble, unscaled)
    4. XGBoost (gradient boosted trees with early stopping, unscaled)
    5. LightGBM (gradient boosted trees with early stopping, unscaled)
    6. K-Nearest Neighbors (KNN distance-based classifier, scaled)

    Args:
        splits: Prepared train, val, and test splits.
        artifacts_dir: Path to directory for saving model binaries.
        save_models: If True, writes models and comparison report to disk.

    Returns:
        Tuple of (Comparison DataFrame, Dictionary of trained model objects).
    """
    logger.info("Training and evaluating churn classification models...")
    random_seed = int(ConfigManager.get("project.random_seed", 42))

    if artifacts_dir is None:
        artifacts_dir = ConfigManager.get("paths.models_dir", "models_artifacts")
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # Dictionary of models to train
    models: Dict[str, Any] = {
        "Logistic Regression": (
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_seed,
            ),
            True,  # requires_scaling
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=random_seed,
            ),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=7,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
            ),
            False,
        ),
        "KNN": (
            KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                metric="minkowski",
            ),
            True,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="auc",
                random_state=random_seed,
                early_stopping_rounds=30,
            ),
            False,
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=random_seed,
                verbose=-1,
            ),
            False,
        ),
    }

    comparison_records: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for name, (model, requires_scaling) in models.items():
        logger.info(f"Training {name} (scaled={requires_scaling})...")

        X_train_curr = splits.X_train_scaled if requires_scaling else splits.X_train
        X_val_curr = splits.X_val_scaled if requires_scaling else splits.X_val
        X_test_curr = splits.X_test_scaled if requires_scaling else splits.X_test

        # Fit model (with early stopping on validation fold for XGBoost / LightGBM)
        if name == "XGBoost":
            model.fit(
                X_train_curr,
                splits.y_train,
                eval_set=[(X_val_curr, splits.y_val)],
                verbose=False,
            )
        elif name == "LightGBM":
            from lightgbm import early_stopping
            model.fit(
                X_train_curr,
                splits.y_train,
                eval_set=[(X_val_curr, splits.y_val)],
                callbacks=[early_stopping(stopping_rounds=30, verbose=False)],
            )
        else:
            model.fit(X_train_curr, splits.y_train)

        # Evaluate on Validation set
        val_metrics = evaluate_classifier(model, X_val_curr, splits.y_val, split_name="Val")

        # Evaluate on Held-out Test set
        test_metrics = evaluate_classifier(model, X_test_curr, splits.y_test, split_name="Test")

        record = {"Model": name, **test_metrics, **val_metrics}
        comparison_records.append(record)
        trained_models[name] = model

        logger.info(
            f"{name} -> Test ROC-AUC: {test_metrics['Test_ROC_AUC']:.4f}, "
            f"Test Recall: {test_metrics['Test_Recall']:.4f}, "
            f"Test F1: {test_metrics['Test_F1']:.4f}"
        )

        if save_models:
            slug = name.lower().replace(" ", "_")
            joblib.dump(model, artifacts_path / f"churn_{slug}_model.joblib")

    # Save scaler and feature names
    if save_models:
        joblib.dump(splits.scaler, artifacts_path / "scaler.joblib")
        with open(artifacts_path / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(splits.feature_names, f, indent=2)

    # Sort comparison table by ROC-AUC and Recall
    comp_df = pd.DataFrame(comparison_records).sort_values(
        by=["Test_ROC_AUC", "Test_Recall"], ascending=False
    ).reset_index(drop=True)

    # Determine recommended production model
    best_model_name = comp_df.iloc[0]["Model"]
    logger.info(f"Top performing churn model: {best_model_name}")

    if save_models:
        prod_model_file = artifacts_path / "churn_production_model.joblib"
        joblib.dump(trained_models[best_model_name], prod_model_file)


        comp_json = artifacts_path / "churn_model_comparison.json"
        comp_df.to_json(comp_json, orient="records", indent=2)

        docs_comp = Path("docs") / "churn_model_comparison.json"
        docs_comp.parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_json(docs_comp, orient="records", indent=2)

    return comp_df, trained_models


if __name__ == "__main__":
    features_path = ConfigManager.get(
        "paths.customer_features_file", "data/processed/customer_features.parquet"
    )
    df_features = pd.read_parquet(features_path)
    splits = prepare_data_splits(df_features, target_col="churn")
    train_and_evaluate_churn_models(splits)
