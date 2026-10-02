"""Unit tests for classical machine learning and CLV regression models."""

import numpy as np
import pandas as pd
import pytest

from src.models.churn_models import train_and_evaluate_churn_models
from src.models.clv_models import train_and_evaluate_clv_models
from src.models.data_split import prepare_data_splits


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    """Fixture generating a synthetic customer feature dataset with 200 samples."""
    np.random.seed(42)
    n = 200
    recency = np.random.uniform(5, 400, n)
    frequency = np.random.randint(1, 20, n)
    total_spend = frequency * np.random.uniform(20, 200, n)

    # Churn strongly correlated with recency and inversely with frequency
    prob = 1.0 / (1.0 + np.exp(-(recency / 100.0 - frequency / 5.0)))
    churn = (np.random.rand(n) < prob).astype(int)

    # CLV forward spend (higher for retained, 0 for churned)
    clv = np.where(churn == 0, total_spend * 0.4 + np.random.normal(0, 10, n), 0.0)
    clv = np.clip(clv, 0, None)

    df = pd.DataFrame(
        {
            "customer_id": range(1000, 1000 + n),
            "recency_days": recency,
            "frequency": frequency.astype(float),
            "total_spend": total_spend,
            "gross_spend": total_spend * 1.05,
            "avg_basket_size": np.random.uniform(2, 20, n),
            "velocity_acceleration": np.random.uniform(0.1, 2.0, n),
            "return_line_rate": np.random.uniform(0, 0.2, n),
            "engagement_score": np.random.uniform(10, 90, n),
            "is_uk": np.random.choice([0, 1], n),
            "churn": churn,
            "clv_next_90d": clv,
            "primary_country": ["United Kingdom"] * n,
            "engagement_tier": ["Medium"] * n,
        }
    )
    return df


def test_prepare_data_splits(mock_dataset: pd.DataFrame):
    """Verifies that 70/15/15 data partitioning and scaling work as specified."""
    splits = prepare_data_splits(mock_dataset, target_col="churn", test_size=0.15, val_size=0.15, random_seed=42)

    total_len = len(mock_dataset)
    assert abs(len(splits.X_train) - int(total_len * 0.70)) <= 1
    assert abs(len(splits.X_val) - int(total_len * 0.15)) <= 1
    assert len(splits.X_test) == int(total_len * 0.15)


    # Scaled matrix dimensions
    assert splits.X_train_scaled.shape == splits.X_train.shape
    assert splits.X_test_scaled.shape == splits.X_test.shape

    # Training scaled mean should be near 0 and variance near 1
    assert np.allclose(splits.X_train_scaled.mean(axis=0), 0, atol=1e-2)
    assert np.allclose(splits.X_train_scaled.std(axis=0), 1, atol=1e-2)


def test_train_and_evaluate_churn_models(tmp_path, mock_dataset: pd.DataFrame):
    """Verifies that all 6 churn classifiers train and evaluate properly."""
    splits = prepare_data_splits(mock_dataset, target_col="churn", random_seed=42)
    comp_df, trained_models = train_and_evaluate_churn_models(
        splits, artifacts_dir=str(tmp_path), save_models=True
    )

    assert len(comp_df) == 6
    expected_models = {"Logistic Regression", "Decision Tree", "Random Forest", "KNN", "XGBoost", "LightGBM"}
    assert set(comp_df["Model"]) == expected_models

    # Verify metrics columns
    for metric in ["Test_ROC_AUC", "Test_Recall", "Test_Precision", "Test_F1"]:
        assert metric in comp_df.columns
        assert (comp_df[metric] >= 0.0).all()
        assert (comp_df[metric] <= 1.0).all()

    assert (tmp_path / "churn_production_model.joblib").exists()
    assert (tmp_path / "scaler.joblib").exists()


def test_train_and_evaluate_clv_models(tmp_path, mock_dataset: pd.DataFrame):
    """Verifies CLV regressors train and compute MAE, RMSE, and R2."""
    splits = prepare_data_splits(mock_dataset, target_col="clv_next_90d", random_seed=42)
    comp_df, trained_models = train_and_evaluate_clv_models(
        splits, artifacts_dir=str(tmp_path), save_models=True
    )

    assert len(comp_df) >= 3
    assert "Test_R2" in comp_df.columns
    assert "Test_MAE" in comp_df.columns
    assert (tmp_path / "clv_best_model.joblib").exists()
