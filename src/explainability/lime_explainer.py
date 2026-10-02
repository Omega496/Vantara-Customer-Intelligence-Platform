"""LIME (Local Interpretable Model-agnostic Explanations) engine for individual customer predictions."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lime.lime_tabular
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LimeExplainabilityEngine:
    """Computes surrogate local linear model explanations for individual predictions via LIME."""

    def __init__(
        self,
        X_train: pd.DataFrame | np.ndarray,
        model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if model_path is None:
            model_path = "models_artifacts/churn_production_model.joblib"

        self.model_path = Path(model_path)
        logger.info(f"Loading production model for LIME from: {self.model_path}")
        self.model = joblib.load(self.model_path)

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns) if feature_names is None else feature_names
            train_vals = X_train[self.feature_names].values
        else:
            self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
            train_vals = np.array(X_train)

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=train_vals,
            feature_names=self.feature_names,
            class_names=["Retained", "Churned"],
            mode="classification",
            random_state=42,
        )

    def explain_instance(
        self,
        instance: pd.Series | np.ndarray,
        customer_id: Any = None,
        num_features: int = 6,
    ) -> Dict[str, Any]:
        """Generates local surrogate linear explanation for a single customer instance."""
        if isinstance(instance, pd.Series):
            x_vals = instance[self.feature_names].values
        else:
            x_vals = np.array(instance)

        exp = self.explainer.explain_instance(
            data_row=x_vals,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
            labels=(1,),  # Explain churn class
        )

        proba = float(self.model.predict_proba(x_vals.reshape(1, -1))[0, 1])
        lime_rules = exp.as_list(label=1)

        factors = []
        for rule, weight in lime_rules:
            factors.append(
                {
                    "rule": rule,
                    "weight": round(float(weight), 4),
                    "impact": "Increases Churn Risk" if weight > 0 else "Protective (Reduces Risk)",
                }
            )

        return {
            "customer_id": customer_id,
            "churn_probability": round(proba, 4),
            "surrogate_score": round(float(exp.score), 4),
            "lime_factors": factors,
        }
