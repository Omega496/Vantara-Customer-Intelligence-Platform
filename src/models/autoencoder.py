"""PyTorch Deep Autoencoder for Unsupervised Customer Spending Anomaly & Fraud Detection."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

AUTOENCODER_FEATURES = [
    "recency_days",
    "frequency",
    "total_spend",
    "gross_spend",
    "avg_order_value",
    "avg_basket_size",
    "velocity_acceleration",
    "return_spend_ratio",
    "return_line_rate",
    "discount_sensitivity",
    "engagement_score",
]


class SpendingAutoencoder(nn.Module):
    """Deep Symmetric Autoencoder for customer spending and purchasing anomaly detection."""

    def __init__(self, input_dim: int = 11, latent_dim: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def compute_reconstruction_errors(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Calculates Mean Squared Error reconstruction loss for each individual sample."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(x_tensor)
        errors = torch.mean((x_tensor - recon) ** 2, dim=1).cpu().numpy()
    return errors


def train_autoencoder(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    latent_dim: int = 6,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    anomaly_percentile: float = 95.0,
    artifacts_dir: Optional[str] = None,
    save_artifacts: bool = True,
) -> Tuple[SpendingAutoencoder, Dict[str, Any], np.ndarray]:
    """Trains unsupervised Autoencoder on spending behaviors and validates anomaly detection threshold.

    Args:
        df: Customer features DataFrame.
        features: Subset of numerical spending features.
        latent_dim: Bottleneck dimension (default 6).
        epochs: Training epochs.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.
        anomaly_percentile: Percentile threshold to flag extreme anomalies (default 95.0).
        artifacts_dir: Directory to save model checkpoint.
        save_artifacts: If True, writes weights, threshold, and distribution plot.

    Returns:
        Tuple of (Trained Autoencoder, Anomaly summary dictionary, Reconstruction errors array).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Spending Autoencoder on device: {device}")

    if features is None:
        features = [f for f in AUTOENCODER_FEATURES if f in df.columns]

    X_raw = df[features].copy().fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(features)
    model = SpendingAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for (x_b,) in loader:
            x_b = x_b.to(device)
            optimizer.zero_grad()
            recon = model(x_b)
            loss = criterion(recon, x_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(x_b)

        epoch_loss = running_loss / len(X_scaled)
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Autoencoder Epoch [{epoch:02d}/{epochs:02d}] - Reconstruction MSE: {epoch_loss:.4f}")

    # Compute reconstruction error per customer across the cohort
    errors = compute_reconstruction_errors(model, X_scaled, device)
    threshold_p95 = float(np.percentile(errors, 95.0))
    threshold_p99 = float(np.percentile(errors, 99.0))
    chosen_threshold = float(np.percentile(errors, anomaly_percentile))

    anomalies_flagged = int((errors > chosen_threshold).sum())
    logger.info(
        f"Reconstruction Loss: Median={np.median(errors):.4f}, "
        f"P95 Threshold={threshold_p95:.4f}, P99 Threshold={threshold_p99:.4f}, "
        f"Flagged Anomalies (>P{anomaly_percentile:.0f})={anomalies_flagged:,} ({anomalies_flagged/len(errors):.1%})"
    )

    summary = {
        "features": features,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "chosen_percentile": anomaly_percentile,
        "threshold": round(chosen_threshold, 4),
        "threshold_p95": round(threshold_p95, 4),
        "threshold_p99": round(threshold_p99, 4),
        "median_error": round(float(np.median(errors)), 4),
        "max_error": round(float(np.max(errors)), 4),
        "anomalous_accounts_count": anomalies_flagged,
    }

    if save_artifacts:
        if artifacts_dir is None:
            artifacts_dir = ConfigManager.get("paths.models_dir", "models_artifacts")
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), artifacts_path / "spending_autoencoder.pt")
        joblib_scaler_file = artifacts_path / "autoencoder_scaler.joblib"
        import joblib
        joblib.dump(scaler, joblib_scaler_file)

        with open(artifacts_path / "anomaly_threshold.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Plot reconstruction error distribution
        fig_dir = Path("docs/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(errors, bins=50, color="#8e44ad", alpha=0.7, edgecolor="black", label="Reconstruction MSE")
        ax.axvline(chosen_threshold, color="#e74c3c", linestyle="--", lw=2, label=f"Threshold (P{anomaly_percentile:.0f}: {chosen_threshold:.2f})")
        ax.set_title("Customer Spending Anomaly Score Distribution", fontsize=13, pad=10)
        ax.set_xlabel("Reconstruction Mean Squared Error", fontsize=11)
        ax.set_ylabel("Customer Count", fontsize=11)
        ax.legend()
        fig.savefig(fig_dir / "autoencoder_reconstruction_error.png", dpi=150)
        plt.close(fig)

    return model, summary, errors


if __name__ == "__main__":
    df_features = pd.read_parquet("data/processed/customer_features.parquet")
    train_autoencoder(df_features)
