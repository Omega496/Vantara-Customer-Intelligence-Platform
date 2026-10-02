"""Generates professional Architecture and Entity-Relationship (ER) diagrams for documentation."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def generate_architecture_diagram(output_path: str = "docs/architecture_diagram.png") -> None:
    """Generates a high-resolution 4-layer system architecture diagram."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor("#0F172A")  # Deep slate dark background
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        8,
        9.5,
        "Vantara Customer Intelligence Platform — System Architecture",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#F8FAFC",
    )
    ax.text(
        8,
        9.1,
        "End-to-End Behavioral Prediction, Deep Learning, Explainability & Serving Flow",
        ha="center",
        va="center",
        fontsize=12,
        color="#94A3B8",
    )

    # Layer 1: Data Ingestion & Storage (Left)
    l1 = patches.FancyBboxPatch(
        (0.6, 1.2),
        3.0,
        7.3,
        boxstyle="round,pad=0.3",
        ec="#38BDF8",
        fc="#1E293B",
        lw=2,
    )
    ax.add_patch(l1)
    ax.text(2.1, 8.2, "1. DATA & INGESTION", ha="center", va="center", fontsize=12, fontweight="bold", color="#38BDF8")

    d_boxes = [
        ("UCI Online Retail II\n(1.06M Raw Transactions)", 7.1, "#334155"),
        ("Validation & Cleaning\n(Deduplication & Returns)", 5.6, "#334155"),
        ("Cleaned Parquet Cache\n(794k Transactions)", 4.1, "#334155"),
        ("Point-in-Time Split\n(Zero Leakage Cutoff)", 2.6, "#334155"),
    ]
    for text, y, col in d_boxes:
        b = patches.FancyBboxPatch((0.9, y - 0.5), 2.4, 0.9, boxstyle="round,pad=0.15", ec="#475569", fc=col, lw=1.2)
        ax.add_patch(b)
        ax.text(2.1, y, text, ha="center", va="center", fontsize=9, color="#F1F5F9", fontweight="medium")

    # Layer 2: Feature Store & Processing
    l2 = patches.FancyBboxPatch(
        (4.2, 1.2),
        3.2,
        7.3,
        boxstyle="round,pad=0.3",
        ec="#818CF8",
        fc="#1E293B",
        lw=2,
    )
    ax.add_patch(l2)
    ax.text(
        5.8,
        8.2,
        "2. FEATURE ENGINEERING",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#818CF8",
    )

    f_boxes = [
        ("RFM Core Metrics\n(Recency, Frequency, Spend)", 7.1, "#312E81"),
        ("Velocity & Trends\n(Gap Variance, Order Ratio)", 5.6, "#312E81"),
        ("Behavioral & Affinity\n(Seasonality, Category Vectors)", 4.1, "#312E81"),
        ("Forward Targets\n(90d Churn Label & 90d CLV)", 2.6, "#312E81"),
    ]
    for text, y, col in f_boxes:
        b = patches.FancyBboxPatch((4.5, y - 0.5), 2.6, 0.9, boxstyle="round,pad=0.15", ec="#6366F1", fc=col, lw=1.2)
        ax.add_patch(b)
        ax.text(5.8, y, text, ha="center", va="center", fontsize=9, color="#EEF2FF", fontweight="medium")

    # Layer 3: Modeling & Explainability
    l3 = patches.FancyBboxPatch(
        (8.0, 1.2),
        3.4,
        7.3,
        boxstyle="round,pad=0.3",
        ec="#34D399",
        fc="#1E293B",
        lw=2,
    )
    ax.add_patch(l3)
    ax.text(
        9.7,
        8.2,
        "3. MODELING & XAI SUITE",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#34D399",
    )

    m_boxes = [
        ("Supervised ML\nLightGBM (AUC 0.82) & Ridge CLV", 7.1, "#064E3B"),
        ("Deep Learning (PyTorch)\nANN Classifier & Sequence LSTM", 5.6, "#064E3B"),
        ("Unsupervised AI\nAutoencoder (P95) & K-Means (k=4)", 4.1, "#064E3B"),
        ("Explainability Engine\nSHAP Waterfalls & Marketing Narratives", 2.6, "#064E3B"),
    ]
    for text, y, col in m_boxes:
        b = patches.FancyBboxPatch((8.3, y - 0.5), 2.8, 0.9, boxstyle="round,pad=0.15", ec="#10B981", fc=col, lw=1.2)
        ax.add_patch(b)
        ax.text(9.7, y, text, ha="center", va="center", fontsize=9, color="#ECFDF5", fontweight="medium")

    # Layer 4: Serving & Presentation
    l4 = patches.FancyBboxPatch(
        (12.0, 1.2),
        3.4,
        7.3,
        boxstyle="round,pad=0.3",
        ec="#F472B6",
        fc="#1E293B",
        lw=2,
    )
    ax.add_patch(l4)
    ax.text(13.7, 8.2, "4. SERVING & UI", ha="center", va="center", fontsize=12, fontweight="bold", color="#F472B6")

    s_boxes = [
        ("FastAPI Backend Service\nREST Endpoints (P95 < 50ms)", 7.1, "#831843"),
        ("Database Persistence\nPostgreSQL & SQLite Fallback", 5.6, "#831843"),
        ("Streamlit Web Dashboard\nKPI Cards, Leaderboard, Drilldown", 4.1, "#831843"),
        ("Docker Containerization\nMulti-Container Compose Stack", 2.6, "#831843"),
    ]
    for text, y, col in s_boxes:
        b = patches.FancyBboxPatch((12.3, y - 0.5), 2.8, 0.9, boxstyle="round,pad=0.15", ec="#EC4899", fc=col, lw=1.2)
        ax.add_patch(b)
        ax.text(13.7, y, text, ha="center", va="center", fontsize=9, color="#FDF2F8", fontweight="medium")

    # Inter-layer connection arrows
    arrow_props = dict(arrowstyle="->", color="#94A3B8", lw=2, mutation_scale=15)
    for y in [7.1, 5.6, 4.1, 2.6]:
        ax.annotate("", xy=(4.3, y), xytext=(3.7, y), arrowprops=arrow_props)
        ax.annotate("", xy=(8.1, y), xytext=(7.5, y), arrowprops=arrow_props)
        ax.annotate("", xy=(12.1, y), xytext=(11.5, y), arrowprops=arrow_props)

    # Footer note
    ax.text(
        8,
        0.5,
        "Production-Grade Architecture: Strict Anti-Leakage Split | In-Memory Model Caching | Dual DB Engine | 88% Test Coverage",
        ha="center",
        va="center",
        fontsize=10,
        color="#64748B",
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Architecture diagram generated: {output_path}")


def generate_er_diagram(output_path: str = "docs/er_diagram.png") -> None:
    """Generates a high-resolution Entity-Relationship (ER) diagram for PostgreSQL/SQLite schema."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(
        7,
        8.4,
        "Vantara Customer Intelligence Platform — Relational Schema (ERD)",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#F8FAFC",
    )
    ax.text(
        7,
        8.0,
        "PostgreSQL / SQLite Database Entity Model & Relational Keys",
        ha="center",
        va="center",
        fontsize=11,
        color="#94A3B8",
    )

    # 1. Customers Table (Central Hub)
    c_box = patches.FancyBboxPatch(
        (0.8, 3.2),
        3.8,
        3.6,
        boxstyle="round,pad=0.2",
        ec="#38BDF8",
        fc="#1E293B",
        lw=2.5,
    )
    ax.add_patch(c_box)
    header = patches.Rectangle((0.8, 6.2), 3.8, 0.6, ec="none", fc="#0284C7")
    ax.add_patch(header)
    ax.text(2.7, 6.5, "customers (Master)", ha="center", va="center", fontsize=11, fontweight="bold", color="#FFFFFF")

    c_fields = [
        ("customer_id", "INTEGER (PK, INDEX)", "#38BDF8"),
        ("country", "VARCHAR(100)", "#E2E8F0"),
        ("first_order_date", "DATETIME", "#E2E8F0"),
        ("last_order_date", "DATETIME", "#E2E8F0"),
        ("is_uk", "BOOLEAN", "#E2E8F0"),
    ]
    for idx, (f_name, f_type, col) in enumerate(c_fields):
        y = 5.7 - (idx * 0.5)
        ax.text(1.1, y, f_name, ha="left", va="center", fontsize=9.5, fontweight="bold" if idx == 0 else "normal", color=col)
        ax.text(4.3, y, f_type, ha="right", va="center", fontsize=8.5, color="#94A3B8")

    # 2. Customer Segments Table (Top Right)
    seg_box = patches.FancyBboxPatch(
        (8.4, 5.2),
        4.8,
        2.8,
        boxstyle="round,pad=0.2",
        ec="#A855F7",
        fc="#1E293B",
        lw=2,
    )
    ax.add_patch(seg_box)
    h_seg = patches.Rectangle((8.4, 7.4), 4.8, 0.6, ec="none", fc="#7E22CE")
    ax.add_patch(h_seg)
    ax.text(10.8, 7.7, "customer_segments", ha="center", va="center", fontsize=11, fontweight="bold", color="#FFFFFF")

    seg_fields = [
        ("customer_id", "INTEGER (PK, FK)", "#A855F7"),
        ("segment_id", "INTEGER", "#E2E8F0"),
        ("segment_name", "VARCHAR(100)", "#E2E8F0"),
        ("engagement_score", "FLOAT", "#E2E8F0"),
        ("updated_at", "DATETIME", "#E2E8F0"),
    ]
    for idx, (f_name, f_type, col) in enumerate(seg_fields):
        y = 7.0 - (idx * 0.42)
        ax.text(8.7, y, f_name, ha="left", va="center", fontsize=9, fontweight="bold" if idx == 0 else "normal", color=col)
        ax.text(12.9, y, f_type, ha="right", va="center", fontsize=8.5, color="#94A3B8")

    # 3. Predictions Table (Middle Right)
    pred_box = patches.FancyBboxPatch(
        (8.4, 1.8),
        4.8,
        3.1,
        boxstyle="round,pad=0.2",
        ec="#10B981",
        fc="#1E293B",
        lw=2,
    )
    ax.add_patch(pred_box)
    h_pred = patches.Rectangle((8.4, 4.3), 4.8, 0.6, ec="none", fc="#047857")
    ax.add_patch(h_pred)
    ax.text(10.8, 4.6, "predictions (Audit Log)", ha="center", va="center", fontsize=11, fontweight="bold", color="#FFFFFF")

    p_fields = [
        ("id", "INTEGER (PK, AUTOINC)", "#10B981"),
        ("customer_id", "INTEGER (FK, INDEX)", "#10B981"),
        ("churn_probability", "FLOAT (0.0 - 1.0)", "#E2E8F0"),
        ("is_churn", "BOOLEAN", "#E2E8F0"),
        ("predicted_clv", "FLOAT (Monetary £)", "#E2E8F0"),
        ("risk_tier", "VARCHAR(50)", "#E2E8F0"),
        ("model_version", "VARCHAR(50)", "#E2E8F0"),
        ("scored_at", "DATETIME", "#E2E8F0"),
    ]
    for idx, (f_name, f_type, col) in enumerate(p_fields):
        y = 3.95 - (idx * 0.35)
        ax.text(8.7, y, f_name, ha="left", va="center", fontsize=8.5, fontweight="bold" if idx < 2 else "normal", color=col)
        ax.text(12.9, y, f_type, ha="right", va="center", fontsize=8, color="#94A3B8")

    # 4. Anomaly Reports Table (Bottom)
    ano_box = patches.FancyBboxPatch(
        (0.8, 0.6),
        5.8,
        2.2,
        boxstyle="round,pad=0.2",
        ec="#F59E0B",
        fc="#1E293B",
        lw=2,
    )
    ax.add_patch(ano_box)
    h_ano = patches.Rectangle((0.8, 2.2), 5.8, 0.6, ec="none", fc="#B45309")
    ax.add_patch(h_ano)
    ax.text(3.7, 2.5, "anomaly_reports (Spending Reconstruction Outliers)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#FFFFFF")

    a_fields = [
        ("id", "INTEGER (PK)", "#F59E0B"),
        ("customer_id", "INTEGER (FK, INDEX)", "#F59E0B"),
        ("reconstruction_error", "FLOAT", "#E2E8F0"),
        ("is_anomaly_95", "BOOLEAN", "#E2E8F0"),
        ("is_anomaly_99", "BOOLEAN", "#E2E8F0"),
        ("flagged_at", "DATETIME", "#E2E8F0"),
    ]
    for idx, (f_name, f_type, col) in enumerate(a_fields):
        y = 1.9 - (idx * 0.3)
        ax.text(1.1, y, f_name, ha="left", va="center", fontsize=8.5, fontweight="bold" if idx < 2 else "normal", color=col)
        ax.text(6.3, y, f_type, ha="right", va="center", fontsize=8, color="#94A3B8")

    # Relationship connectors
    # customers -> customer_segments (1:1)
    ax.annotate(
        "1 : 1",
        xy=(8.3, 6.5),
        xytext=(4.8, 5.5),
        arrowprops=dict(arrowstyle="<->", color="#A855F7", lw=2),
        fontsize=9,
        fontweight="bold",
        color="#C084FC",
    )

    # customers -> predictions (1:N)
    ax.annotate(
        "1 : N",
        xy=(8.3, 3.4),
        xytext=(4.8, 4.4),
        arrowprops=dict(arrowstyle="-|>", color="#10B981", lw=2),
        fontsize=9,
        fontweight="bold",
        color="#34D399",
    )

    # customers -> anomaly_reports (1:N)
    ax.annotate(
        "1 : N",
        xy=(3.2, 2.9),
        xytext=(3.2, 3.1),
        arrowprops=dict(arrowstyle="-|>", color="#F59E0B", lw=2),
        fontsize=9,
        fontweight="bold",
        color="#FBBF24",
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"ER diagram generated: {output_path}")


if __name__ == "__main__":
    generate_architecture_diagram()
    generate_er_diagram()
