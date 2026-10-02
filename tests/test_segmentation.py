"""Unit tests for customer segmentation and persona clustering module."""

import numpy as np
import pandas as pd
import pytest

from src.segmentation.clustering import (
    evaluate_cluster_range,
    run_customer_segmentation,
)


@pytest.fixture
def mock_segmentation_data() -> pd.DataFrame:
    """Fixture providing synthetic customer data for clustering tests."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "recency_days": np.random.uniform(5, 300, n),
            "frequency": np.random.randint(1, 15, n).astype(float),
            "total_spend": np.random.uniform(50, 5000, n),
            "avg_basket_size": np.random.uniform(2, 20, n),
            "velocity_acceleration": np.random.uniform(0.1, 2.0, n),
            "return_line_rate": np.random.uniform(0, 0.1, n),
            "engagement_score": np.random.uniform(10, 95, n),
            "churn": np.random.choice([0, 1], n),
            "clv_next_90d": np.random.uniform(0, 1000, n),
        }
    )


def test_evaluate_cluster_range(mock_segmentation_data: pd.DataFrame):
    """Verifies that cluster range evaluation calculates silhouettes and inertias."""
    X = mock_segmentation_data[["recency_days", "frequency", "total_spend"]].values
    metrics = evaluate_cluster_range(X, k_range=(2, 4))

    assert len(metrics["k"]) == 3
    assert len(metrics["silhouette"]) == 3
    assert all(s > -1.0 for s in metrics["silhouette"])
    assert all(i > 0 for i in metrics["inertia"])


def test_run_customer_segmentation(tmp_path, mock_segmentation_data: pd.DataFrame):
    """Verifies that full segmentation pipeline creates clusters, labels personas, and persists outputs."""
    segmented_df, profiles = run_customer_segmentation(
        mock_segmentation_data,
        n_clusters=3,
        artifacts_dir=str(tmp_path),
        save_outputs=True,
    )

    assert "cluster_id" in segmented_df.columns
    assert "segment_name" in segmented_df.columns
    assert segmented_df["cluster_id"].nunique() == 3

    assert "evaluation_metrics" in profiles
    assert "segments" in profiles
    assert len(profiles["segments"]) > 0

    assert (tmp_path / "kmeans_segmentation.joblib").exists()
    assert (tmp_path / "gmm_segmentation.joblib").exists()
    assert (tmp_path / "segment_profiles.json").exists()
