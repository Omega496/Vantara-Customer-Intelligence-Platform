"""SHAP (SHapley Additive exPlanations) engine for global and local model explainability."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShapExplainabilityEngine:
    """Computes global feature attributions and individual customer explanations via SHAP TreeExplainer."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if model_path is None:
            model_path = "models_artifacts/churn_production_model.joblib"

        self.model_path = Path(model_path)
        logger.info(f"Loading production model for SHAP from: {self.model_path}")
        self.model = joblib.load(self.model_path)

        if feature_names is None:
            names_path = Path("models_artifacts/feature_names.json")
            if names_path.exists():
                with open(names_path, "r", encoding="utf-8") as f:
                    self.feature_names = json.load(f)
            else:
                self.feature_names = None
        else:
            self.feature_names = feature_names

        self.explainer = shap.TreeExplainer(self.model)

    def explain_dataset(
        self,
        X: pd.DataFrame | np.ndarray,
        max_samples: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculates SHAP values for a sample of customers."""
        if isinstance(X, pd.DataFrame):
            X_eval = X.iloc[:max_samples]
        else:
            X_eval = X[:max_samples]

        shap_values = self.explainer.shap_values(X_eval)

        # In binary classification, handle list or 2D array of outputs
        if isinstance(shap_values, list):
            # Binary class 1 (churn)
            values = shap_values[1]
        elif len(shap_values.shape) == 3:
            values = shap_values[:, :, 1]
        else:
            values = shap_values

        base_val = (
            self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )
        return values, np.array(X_eval), float(base_val)

    def generate_global_summary_plots(
        self,
        X_test: pd.DataFrame,
        output_dir: str = "docs/figures",
    ) -> Dict[str, float]:
        """Generates and saves global summary and bar importance plots."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        feature_cols = self.feature_names or list(X_test.columns)
        X_df = X_test[feature_cols].copy() if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_cols)

        shap_vals, X_eval, base_val = self.explain_dataset(X_df, max_samples=400)

        # 1. Summary Density Plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_eval, feature_names=feature_cols, show=False)
        plt.title("SHAP Global Feature Attributions (Churn Prediction)", fontsize=13, pad=12)
        summary_fig = out_path / "shap_summary_plot.png"
        plt.tight_layout()
        plt.savefig(summary_fig, dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Mean Absolute SHAP Importance Bar Plot
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({"Feature": feature_cols, "Mean_Abs_SHAP": mean_abs_shap}).sort_values(
            by="Mean_Abs_SHAP", ascending=False
        )

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"].head(12)[::-1], importance_df["Mean_Abs_SHAP"].head(12)[::-1], color="#2980b9")
        plt.title("Top 12 Most Influential Features (Mean |SHAP Value|)", fontsize=13, pad=10)
        plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)", fontsize=11)
        bar_fig = out_path / "shap_bar_importance.png"
        plt.tight_layout()
        plt.savefig(bar_fig, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP global plots to {summary_fig} and {bar_fig}")
        return importance_df.set_index("Feature")["Mean_Abs_SHAP"].to_dict()

    def explain_individual_customer(
        self,
        customer_row: pd.Series | np.ndarray,
        customer_id: Any = None,
        output_plot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generates localized SHAP explanation for a single customer."""
        if isinstance(customer_row, pd.Series):
            features = self.feature_names or list(customer_row.index)
            x_vals = customer_row[features].values.reshape(1, -1)
        else:
            features = self.feature_names
            x_vals = np.array(customer_row).reshape(1, -1)

        shap_vals = self.explainer.shap_values(x_vals)
        if isinstance(shap_vals, list):
            cust_shap = shap_vals[1][0]
        elif len(shap_vals.shape) == 3:
            cust_shap = shap_vals[0, :, 1]
        else:
            cust_shap = shap_vals[0]

        base_val = (
            self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )

        proba = float(self.model.predict_proba(x_vals)[0, 1])

        # Rank top positive contributors (increasing churn risk) and negative contributors (reducing churn risk)
        contribs = list(zip(features, x_vals[0], cust_shap))
        sorted_contribs = sorted(contribs, key=lambda x: abs(x[2]), reverse=True)

        top_factors = []
        for feat, val, s_val in sorted_contribs[:8]:
            top_factors.append(
                {
                    "feature": feat,
                    "value": round(float(val), 2),
                    "shap_attribution": round(float(s_val), 4),
                    "direction": "Increases Risk" if s_val > 0 else "Decreases Risk (Protective)",
                }
            )

        # Plot Waterfall / Force Plot if requested
        if output_plot_path:
            plt.figure(figsize=(9, 4))
            top_df = pd.DataFrame(top_factors).head(6)
            colors = ["#e74c3c" if d == "Increases Risk" else "#2ecc71" for d in top_df["direction"]]
            plt.barh(top_df["feature"][::-1], top_df["shap_attribution"][::-1], color=colors[::-1])
            plt.axvline(0, color="black", lw=1)
            plt.title(f"Customer #{customer_id} SHAP Explanation (Churn Prob: {proba:.1%})", fontsize=12, pad=10)
            plt.xlabel("SHAP Attribution (Log-Odds Impact)", fontsize=10)
            plt.tight_layout()
            plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        result = {
            "customer_id": customer_id,
            "churn_probability": round(proba, 4),
            "base_value": round(float(base_val), 4),
            "top_factors": top_factors,
        }
        return result


def run_shap_explanations(
    X_test: pd.DataFrame,
    df_customers: pd.DataFrame,
    output_dir: str = "docs/figures",
) -> Dict[str, Any]:
    """Runs global SHAP analysis and extracts 3 representative customers: Low-Risk, Borderline, and High-Risk."""
    engine = ShapExplainabilityEngine()
    global_importance = engine.generate_global_summary_plots(X_test, output_dir=output_dir)

    # Predict probabilities across test set to select representative customers
    feature_cols = engine.feature_names
    test_features = X_test[feature_cols].copy()
    probs = engine.model.predict_proba(test_features)[:, 1]

    # Representative profiles:
    # 1. Low-Risk (< 0.20)
    low_idx = int(np.argmin(probs))
    # 2. Borderline (~ 0.50)
    border_idx = int(np.argmin(np.abs(probs - 0.50)))
    # 3. High-Risk (> 0.85)
    high_idx = int(np.argmax(probs))

    cust_ids = df_customers.loc[X_test.index, "customer_id"].values if "customer_id" in df_customers.columns else list(X_test.index)

    low_res = engine.explain_individual_customer(
        X_test.iloc[low_idx],
        customer_id=int(cust_ids[low_idx]),
        output_plot_path=f"{output_dir}/shap_local_low_risk.png",
    )
    border_res = engine.explain_individual_customer(
        X_test.iloc[border_idx],
        customer_id=int(cust_ids[border_idx]),
        output_plot_path=f"{output_dir}/shap_local_borderline.png",
    )
    high_res = engine.explain_individual_customer(
        X_test.iloc[high_idx],
        customer_id=int(cust_ids[high_idx]),
        output_plot_path=f"{output_dir}/shap_local_high_risk.png",
    )

    report = {
        "global_importance_top10": {k: round(v, 4) for k, v in list(global_importance.items())[:10]},
        "representative_customers": {
            "low_risk": low_res,
            "borderline": border_res,
            "high_risk": high_res,
        },
    }

    with open("models_artifacts/shap_metadata.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


    logger.info("SHAP explanations completed successfully!")
    return report


if __name__ == "__main__":
    from src.models.data_split import prepare_data_splits
    df_feat = pd.read_parquet("data/processed/customer_features.parquet")
    splits = prepare_data_splits(df_feat, target_col="churn")
    run_shap_explanations(splits.X_test, df_feat)
