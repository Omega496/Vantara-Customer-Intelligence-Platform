"""Unit tests for PyTorch Deep Learning models: ANN, LSTM Sequence Model, and Autoencoder."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.ann import ChurnANN, train_ann
from src.models.autoencoder import (
    SpendingAutoencoder,
    train_autoencoder,
)
from src.models.data_split import prepare_data_splits
from src.models.lstm import PurchaseLSTM, train_lstm


@pytest.fixture
def sample_dl_data() -> pd.DataFrame:
    """Fixture providing customer features for neural network tests."""
    np.random.seed(42)
    n = 120
    recency = np.random.uniform(5, 300, n)
    frequency = np.random.randint(1, 15, n).astype(float)
    spend = frequency * np.random.uniform(20, 150, n)
    prob = 1.0 / (1.0 + np.exp(-(recency / 100.0 - frequency / 5.0)))
    churn = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "recency_days": recency,
            "frequency": frequency,
            "total_spend": spend,
            "gross_spend": spend * 1.05,
            "avg_order_value": spend / frequency,
            "avg_basket_size": np.random.uniform(2, 20, n),
            "velocity_acceleration": np.random.uniform(0.1, 2.0, n),
            "return_line_rate": np.random.uniform(0, 0.1, n),
            "return_spend_ratio": np.random.uniform(0, 0.1, n),
            "discount_sensitivity": np.random.uniform(0, 0.3, n),
            "engagement_score": np.random.uniform(10, 95, n),
            "is_uk": np.random.choice([0, 1], n),
            "churn": churn,
            "clv_next_90d": np.where(churn == 0, spend * 0.3, 0.0),
        }
    )


def test_churn_ann_forward_and_training(tmp_path, sample_dl_data: pd.DataFrame):
    """Verifies that ChurnANN instantiates, executes forward pass, and completes training."""
    splits = prepare_data_splits(sample_dl_data, target_col="churn", random_seed=42)

    model, metrics, history = train_ann(
        splits,
        epochs=5,
        batch_size=32,
        artifacts_dir=str(tmp_path),
        save_artifacts=True,
    )

    assert isinstance(model, ChurnANN)
    assert "Test_ROC_AUC" in metrics
    assert "Test_Recall" in metrics
    assert len(history["train_loss"]) == 5
    assert (tmp_path / "ann_churn_model.pt").exists()


def test_lstm_sequence_and_training(tmp_path):
    """Verifies sequence construction and LSTM model training."""
    N, L, D = 60, 5, 4
    X_seq = np.random.randn(N, L, D).astype(np.float32)
    y_seq = np.random.choice([0, 1], N).astype(np.float32)

    model = PurchaseLSTM(input_dim=D, hidden_dim=32, num_layers=1)
    dummy_input = torch.tensor(X_seq[:8])
    output = model(dummy_input)
    assert output.shape == (8,)

    trained_model, metrics, history = train_lstm(
        X_seq,
        y_seq,
        epochs=4,
        batch_size=16,
        artifacts_dir=str(tmp_path),
        save_artifacts=True,
    )

    assert "Test_ROC_AUC" in metrics
    assert (tmp_path / "lstm_sequence_model.pt").exists()


def test_spending_autoencoder(tmp_path, sample_dl_data: pd.DataFrame):
    """Verifies autoencoder training, reconstruction error computation, and anomaly thresholding."""
    model, summary, errors = train_autoencoder(
        sample_dl_data,
        latent_dim=4,
        epochs=5,
        batch_size=32,
        anomaly_percentile=95.0,
        artifacts_dir=str(tmp_path),
        save_artifacts=True,
    )

    assert isinstance(model, SpendingAutoencoder)
    assert len(errors) == len(sample_dl_data)
    assert summary["threshold"] > 0.0
    assert summary["anomalous_accounts_count"] > 0
    assert (tmp_path / "spending_autoencoder.pt").exists()
    assert (tmp_path / "anomaly_threshold.json").exists()
