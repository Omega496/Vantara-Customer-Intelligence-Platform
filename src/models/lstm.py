"""PyTorch LSTM Sequence Model for Customer Purchase Trajectory and Churn Dynamics."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_customer_sequences(
    df_transactions: pd.DataFrame,
    df_customers: pd.DataFrame,
    seq_length: int = 6,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Builds pre-padded 3D tensor of historical order sequences per customer.

    Features per timestep:
    1. Order Amount (scaled)
    2. Gap Days from prior order
    3. Total Items Quantity (scaled)
    4. Return Flag (0 or 1)

    Args:
        df_transactions: Cleaned transactions DataFrame.
        df_customers: Processed customer features DataFrame with 'churn' label.
        seq_length: Fixed sequence length L (default 6).
        cutoff_date: Observation window cutoff date.

    Returns:
        Tuple of (X tensor of shape [N, L, 4], y vector of shape [N], list of customer IDs).
    """
    logger.info(f"Constructing purchase sequences (L={seq_length}) for {len(df_customers):,} customers...")
    if cutoff_date is not None:
        df_transactions = df_transactions[df_transactions["invoice_date"] <= cutoff_date]

    # Aggregate by customer and invoice to form order-level events
    orders = (
        df_transactions.groupby(["customer_id", "invoice"])
        .agg(
            invoice_date=("invoice_date", "min"),
            total_amount=("total_amount", "sum"),
            total_quantity=("quantity", "sum"),
            is_return=("is_return", "any"),
        )
        .reset_index()
        .sort_values(by=["customer_id", "invoice_date"])
    )

    customer_target_map = df_customers.set_index("customer_id")["churn"].to_dict()
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    valid_customer_ids: List[int] = []

    for cust_id, target in customer_target_map.items():
        cust_orders = orders[orders["customer_id"] == cust_id]
        if cust_orders.empty:
            continue

        order_dates = cust_orders["invoice_date"].values
        amounts = cust_orders["total_amount"].values
        quantities = cust_orders["total_quantity"].values
        returns = cust_orders["is_return"].astype(float).values

        # Calculate inter-order gap days
        gaps = [0.0]
        for i in range(1, len(order_dates)):
            gap = (order_dates[i] - order_dates[i - 1]) / np.timedelta64(1, "D")
            gaps.append(float(np.clip(gap, 0, 365)))

        # Normalize features with robust log transforms
        norm_amounts = np.log1p(np.clip(amounts, 0, None)) / 10.0
        norm_gaps = np.array(gaps) / 100.0
        norm_quantities = np.log1p(np.clip(quantities, 0, None)) / 10.0

        # Stack into [T, 4] matrix
        order_matrix = np.column_stack([norm_amounts, norm_gaps, norm_quantities, returns])

        # Pre-pad or truncate to fixed seq_length
        num_orders = len(order_matrix)
        padded_seq = np.zeros((seq_length, 4), dtype=np.float32)
        if num_orders >= seq_length:
            padded_seq = order_matrix[-seq_length:]
        else:
            padded_seq[-num_orders:] = order_matrix

        sequences.append(padded_seq)
        labels.append(int(target))
        valid_customer_ids.append(cust_id)

    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(labels, dtype=np.float32)
    logger.info(f"Built sequence tensor of shape: {X_seq.shape}")
    return X_seq, y_seq, valid_customer_ids


class PurchaseLSTM(nn.Module):
    """2-Layer Recurrent LSTM network for purchase trajectory sequence modeling."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        out, (hn, _) = self.lstm(x)
        # Use final time step hidden representation
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def train_lstm(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 8,
    artifacts_dir: Optional[str] = None,
    save_artifacts: bool = True,
) -> Tuple[PurchaseLSTM, Dict[str, Any], Dict[str, List[float]]]:
    """Trains the PurchaseLSTM model on sequence tensors with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training PurchaseLSTM on device: {device}")

    # Stratified 70/15/15 split on sequence cohort
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_seq, y_seq, test_size=0.15, random_state=42, stratify=y_seq
    )
    val_frac = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac, random_state=42, stratify=y_train_val
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = PurchaseLSTM(input_dim=4, hidden_dim=64, num_layers=2, dropout_rate=0.2).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_weights: Optional[Dict[str, Any]] = None
    patience_cnt = 0
    loss_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)

        epoch_train_loss = train_loss / len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * len(x_b)

        epoch_val_loss = val_loss / len(X_val)
        loss_history["train_loss"].append(round(epoch_train_loss, 4))
        loss_history["val_loss"].append(round(epoch_val_loss, 4))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"LSTM Epoch [{epoch:02d}/{epochs:02d}] - Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"LSTM Early stopping at epoch {epoch}")
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Evaluate on Test Set
    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_proba = model(x_test_t).cpu().numpy()

    y_test_pred = (y_test_proba >= 0.5).astype(int)
    test_auc = float(roc_auc_score(y_test, y_test_proba))
    test_rec = float(recall_score(y_test, y_test_pred, zero_division=0))
    test_prec = float(precision_score(y_test, y_test_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    test_acc = float(accuracy_score(y_test, y_test_pred))

    metrics = {
        "Model": "Sequence LSTM (PyTorch)",
        "Test_ROC_AUC": round(test_auc, 4),
        "Test_Recall": round(test_rec, 4),
        "Test_Precision": round(test_prec, 4),
        "Test_F1": round(test_f1, 4),
        "Test_Accuracy": round(test_acc, 4),
        "Best_Val_Loss": round(best_val_loss, 4),
    }
    logger.info(f"LSTM Test Evaluation -> ROC-AUC: {test_auc:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")

    if save_artifacts:
        if artifacts_dir is None:
            artifacts_dir = ConfigManager.get("paths.models_dir", "models_artifacts")
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), artifacts_path / "lstm_sequence_model.pt")

        fig_dir = Path("docs/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(loss_history["train_loss"], label="Training Loss", color="#16a085", lw=2)
        ax.plot(loss_history["val_loss"], label="Validation Loss", color="#e67e22", lw=2)
        ax.set_title("PyTorch LSTM Sequence Model Loss Curve", fontsize=13, pad=10)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
        ax.legend()
        fig.savefig(fig_dir / "lstm_loss_curve.png", dpi=150)
        plt.close(fig)

    return model, metrics, loss_history


if __name__ == "__main__":
    df_clean = pd.read_parquet("data/interim/cleaned_transactions.parquet")
    df_cust = pd.read_parquet("data/processed/customer_features.parquet")
    cutoff = pd.Timestamp("2011-09-10 12:50:00")
    X_seq, y_seq, _ = build_customer_sequences(df_clean, df_cust, seq_length=6, cutoff_date=cutoff)
    train_lstm(X_seq, y_seq)
