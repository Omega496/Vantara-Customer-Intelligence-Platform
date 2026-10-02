"""PyTorch Feed-Forward Artificial Neural Network (ANN) Classifier for Customer Churn."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.models.data_split import DataSplits, prepare_data_splits
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChurnANN(nn.Module):
    """Deep Feed-Forward Neural Network with Batch Normalization and Dropout for Churn Prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def evaluate_ann(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    split_name: str = "Test",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes full classification metrics for PyTorch model."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_proba = model(x_tensor).cpu().numpy()

    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    auc = float(roc_auc_score(y, y_proba))
    cm = confusion_matrix(y, y_pred).tolist()

    return {
        f"{split_name}_Accuracy": round(acc, 4),
        f"{split_name}_Precision": round(prec, 4),
        f"{split_name}_Recall": round(rec, 4),
        f"{split_name}_F1": round(f1, 4),
        f"{split_name}_ROC_AUC": round(auc, 4),
        f"{split_name}_Confusion_Matrix": cm,
    }


def train_ann(
    splits: DataSplits,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 10,
    artifacts_dir: Optional[str] = None,
    save_artifacts: bool = True,
) -> Tuple[ChurnANN, Dict[str, Any], Dict[str, List[float]]]:
    """Trains the ChurnANN classifier with early stopping and loss curve plotting.

    Args:
        splits: Prepared train, val, test splits with standardized features.
        epochs: Max number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam optimizer learning rate.
        weight_decay: L2 penalty parameter.
        patience: Epochs of non-improving val loss before early stopping triggers.
        artifacts_dir: Directory to save model checkpoint.
        save_artifacts: If True, writes weights and loss curve.

    Returns:
        Tuple of (Trained ChurnANN model, Test metrics dictionary, Loss history dictionary).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training PyTorch ChurnANN on device: {device}")

    input_dim = splits.X_train_scaled.shape[1]
    model = ChurnANN(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3).to(device)

    # Convert splits to PyTorch DataLoader
    train_dataset = TensorDataset(
        torch.tensor(splits.X_train_scaled, dtype=torch.float32),
        torch.tensor(splits.y_train.values, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(splits.X_val_scaled, dtype=torch.float32),
        torch.tensor(splits.y_val.values, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights: Optional[Dict[str, Any]] = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(x_b)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                loss = criterion(preds, y_b)
                running_val_loss += loss.item() * len(x_b)

        epoch_val_loss = running_val_loss / len(val_dataset)
        loss_history["train_loss"].append(round(epoch_train_loss, 4))
        loss_history["val_loss"].append(round(epoch_val_loss, 4))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
            )

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Final Test Set Evaluation
    test_metrics = evaluate_ann(model, splits.X_test_scaled, splits.y_test.values, device, split_name="Test")
    val_metrics = evaluate_ann(model, splits.X_val_scaled, splits.y_val.values, device, split_name="Val")

    full_results = {
        "Model": "Deep ANN (PyTorch)",
        **test_metrics,
        **val_metrics,
        "Best_Val_Loss": round(best_val_loss, 4),
        "Total_Epochs_Trained": len(loss_history["train_loss"]),
    }
    logger.info(f"ANN Test Evaluation -> ROC-AUC: {test_metrics['Test_ROC_AUC']:.4f}, Recall: {test_metrics['Test_Recall']:.4f}, F1: {test_metrics['Test_F1']:.4f}")

    # Persistence
    if save_artifacts:
        if artifacts_dir is None:
            artifacts_dir = ConfigManager.get("paths.models_dir", "models_artifacts")
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), artifacts_path / "ann_churn_model.pt")

        # Plot Loss Curves
        fig_dir = Path("docs/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(loss_history["train_loss"], label="Training Loss", color="#2980b9", lw=2)
        ax.plot(loss_history["val_loss"], label="Validation Loss", color="#e74c3c", lw=2)
        ax.set_title("PyTorch ANN Training & Validation Loss Curve", fontsize=13, pad=10)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
        ax.legend()
        fig.savefig(fig_dir / "ann_loss_curve.png", dpi=150)
        plt.close(fig)

    return model, full_results, loss_history


if __name__ == "__main__":
    features_path = ConfigManager.get("paths.customer_features_file", "data/processed/customer_features.parquet")
    df_features = pd.read_parquet(features_path)
    splits = prepare_data_splits(df_features, target_col="churn")
    train_ann(splits)
