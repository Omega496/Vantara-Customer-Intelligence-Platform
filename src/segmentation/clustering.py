"""Customer Segmentation using K-Means and Gaussian Mixture Models (GMM)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLUSTERING_FEATURES = [
    "recency_days",
    "frequency",
    "total_spend",
    "avg_basket_size",
    "velocity_acceleration",
    "return_line_rate",
    "engagement_score",
]


def evaluate_cluster_range(
    X_scaled: np.ndarray,
    k_range: Tuple[int, int] = (2, 8),
) -> Dict[str, Any]:
    """Evaluates K-Means clustering across a range of k values to determine the optimal elbow and silhouette scores.

    Args:
        X_scaled: Standardized numerical feature matrix.
        k_range: (min_k, max_k) inclusive.

    Returns:
        Dictionary containing inertias, silhouette scores, and davies-bouldin indices for each k.
    """
    logger.info(f"Evaluating clustering metrics across k in range {k_range}...")
    metrics: Dict[str, List[Any]] = {
        "k": [],
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
    }

    random_seed = int(ConfigManager.get("project.random_seed", 42))

    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))

        metrics["k"].append(k)
        metrics["inertia"].append(round(float(km.inertia_), 2))
        metrics["silhouette"].append(round(sil, 4))
        metrics["davies_bouldin"].append(round(db, 4))

        logger.info(f"k={k} -> Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}, Inertia={km.inertia_:.1f}")

    return metrics


def label_segment_personas(summary_df: pd.DataFrame) -> Dict[int, str]:
    """Generates business-readable persona labels based on cluster centroids."""
    personas = {}
    for cluster_id, row in summary_df.iterrows():
        rec = row["recency_days"]
        freq = row["frequency"]
        spend = row["total_spend"]


        if spend >= 50000 or freq >= 50:
            personas[cluster_id] = "VIP Champions"
        elif freq >= 1 and spend >= 8000 and rec >= 200:
            personas[cluster_id] = "High-Value Inactive"
        elif rec <= 90 and freq >= 5:
            personas[cluster_id] = "Loyal Regulars"
        else:
            personas[cluster_id] = "At-Risk / Lapsed"

    # Ensure uniqueness in case of edge centroids
    seen = set()
    for cid in list(personas.keys()):
        label = personas[cid]
        if label in seen:
            personas[cid] = f"{label} (Cohort {cid})"
        seen.add(personas[cid])

    return personas



def run_customer_segmentation(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_clusters: int = 4,
    artifacts_dir: Optional[str] = None,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Executes customer segmentation pipeline using K-Means and GMM, producing business-readable profiles.

    Args:
        df: Customer features DataFrame containing customer_id and RFM features.
        features: Subset of features for clustering (defaults to CLUSTERING_FEATURES).
        n_clusters: Number of clusters (default 4).
        artifacts_dir: Directory to save model binaries and reports.
        save_outputs: If True, writes artifacts and parquet files.

    Returns:
        Tuple of (Customer DataFrame with segment assignments, Segment Profiles dictionary).
    """
    logger.info("Starting customer segmentation pipeline...")
    random_seed = int(ConfigManager.get("project.random_seed", 42))

    if features is None:
        features = [f for f in CLUSTERING_FEATURES if f in df.columns]

    if artifacts_dir is None:
        artifacts_dir = ConfigManager.get("paths.models_dir", "models_artifacts")
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    X = df[features].copy().fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Evaluate range of k for selection justification
    cluster_eval = evaluate_cluster_range(X_scaled, k_range=(2, 7))

    # 2. Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=15)
    km_labels = kmeans.fit_predict(X_scaled)
    km_sil = float(silhouette_score(X_scaled, km_labels))
    km_db = float(davies_bouldin_score(X_scaled, km_labels))

    # 3. Fit Gaussian Mixture Model (GMM) for comparison
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_seed)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_sil = float(silhouette_score(X_scaled, gmm_labels))
    gmm_bic = float(gmm.bic(X_scaled))

    logger.info(
        f"K-Means (k={n_clusters}): Silhouette={km_sil:.4f}, Davies-Bouldin={km_db:.4f} | "
        f"GMM: Silhouette={gmm_sil:.4f}, BIC={gmm_bic:.1f}"
    )

    # 4. Generate Business Segment Profiles
    segmented_df = df.copy()
    segmented_df["cluster_id"] = km_labels
    segmented_df["gmm_cluster_id"] = gmm_labels

    cluster_summary = segmented_df.groupby("cluster_id").agg(
        customer_count=("customer_id", "count"),
        recency_days=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        total_spend=("total_spend", "mean"),
        avg_basket_size=("avg_basket_size", "mean"),
        engagement_score=("engagement_score", "mean"),
        churn_rate=("churn", "mean") if "churn" in segmented_df.columns else ("cluster_id", "count"),
        forward_clv=("clv_next_90d", "mean") if "clv_next_90d" in segmented_df.columns else ("cluster_id", "count"),
    ).round(2)

    personas = label_segment_personas(cluster_summary)
    segmented_df["segment_name"] = segmented_df["cluster_id"].map(personas)

    # Compile profiles with strategic marketing recommendations
    profile_descriptions = {
        "VIP Champions": "High spend, frequent, and highly active. Recommend VIP loyalty perks, early product access, and dedicated account manager.",
        "Loyal Regulars": "Consistent, steady purchasing cadence with moderate spend. Recommend cross-selling campaigns and replenishment reminders.",
        "High-Value Inactive": "Historically high spenders who made bulk orders but became inactive. Urgent intervention required: win-back discounts and executive outreach.",
        "At-Risk / Lapsed": "Low frequency, low spend buyers with high churn risk. Recommend automated promotional email drips and price-incentive promos.",
    }


    profiles_report: Dict[str, Any] = {
        "evaluation_metrics": {
            "kmeans_silhouette": round(km_sil, 4),
            "kmeans_davies_bouldin": round(km_db, 4),
            "gmm_silhouette": round(gmm_sil, 4),
            "gmm_bic": round(gmm_bic, 2),
            "k_selection_curve": cluster_eval,
        },
        "segments": {},
    }

    for c_id, row in cluster_summary.iterrows():
        p_name = personas.get(c_id, f"Cluster {c_id}")
        profiles_report["segments"][p_name] = {
            "cluster_id": int(c_id),
            "customer_count": int(row["customer_count"]),
            "share_of_cohort": round(float(row["customer_count"] / len(df)), 4),
            "mean_recency_days": float(row["recency_days"]),
            "mean_frequency_orders": float(row["frequency"]),
            "mean_total_spend": float(row["total_spend"]),
            "mean_engagement_score": float(row["engagement_score"]),
            "churn_rate": float(row["churn_rate"]),
            "forward_clv_90d": float(row["forward_clv"]),
            "recommended_strategy": profile_descriptions.get(p_name, "Targeted engagement campaigns."),
        }

    logger.info(f"Segment Profiles generated: {list(profiles_report['segments'].keys())}")

    if save_outputs:
        joblib.dump(kmeans, artifacts_path / "kmeans_segmentation.joblib")
        joblib.dump(gmm, artifacts_path / "gmm_segmentation.joblib")
        joblib.dump(scaler, artifacts_path / "segmentation_scaler.joblib")

        # Save segmented customers to artifacts and processed dir
        segmented_df.to_parquet(artifacts_path / "customer_segments.parquet", index=False)
        proc_dir = Path("data/processed")
        if proc_dir.exists() and artifacts_path.name != "tmp" and "tmp" not in str(artifacts_path):
            segmented_df.to_parquet(proc_dir / "customer_segments.parquet", index=False)


        # Save profiles JSON
        with open(artifacts_path / "segment_profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles_report, f, indent=2)
        with open(Path("docs") / "segment_profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles_report, f, indent=2)

    return segmented_df, profiles_report


if __name__ == "__main__":
    features_path = ConfigManager.get(
        "paths.customer_features_file", "data/processed/customer_features.parquet"
    )
    df_features = pd.read_parquet(features_path)
    run_customer_segmentation(df_features)
