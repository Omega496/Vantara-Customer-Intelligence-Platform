"""Unit tests for model explainability engines: SHAP, LIME, and plain-language narrative generation."""

import numpy as np
import pandas as pd

from src.explainability.narrative import generate_plain_language_narrative


def test_plain_language_narrative_high_risk():
    """Verifies that plain-language generator produces accurate briefing for high risk customer."""
    top_factors = [
        {"feature": "recency_days", "value": 240.0, "shap_attribution": 0.85, "direction": "Increases Risk"},
        {"feature": "velocity_acceleration", "value": 0.1, "shap_attribution": 0.45, "direction": "Increases Risk"},
        {"feature": "total_spend", "value": 1500.0, "shap_attribution": -0.30, "direction": "Decreases Risk (Protective)"},
    ]

    narrative = generate_plain_language_narrative(
        customer_id=14527,
        churn_probability=0.825,
        top_factors=top_factors,
    )

    assert "Customer #14527" in narrative
    assert "High Churn Risk" in narrative
    assert "82.5%" in narrative
    assert "240 days of inactivity" in narrative
    assert "strong lifetime spend" in narrative
    assert "Immediate proactive retention outreach" in narrative


def test_plain_language_narrative_low_risk():
    """Verifies that plain-language generator produces accurate briefing for low risk customer."""
    top_factors = [
        {"feature": "total_spend", "value": 4500.0, "shap_attribution": -0.90, "direction": "Decreases Risk (Protective)"},
        {"feature": "frequency", "value": 25.0, "shap_attribution": -0.75, "direction": "Decreases Risk (Protective)"},
        {"feature": "return_line_rate", "value": 0.05, "shap_attribution": 0.10, "direction": "Increases Risk"},
    ]

    narrative = generate_plain_language_narrative(
        customer_id=13085,
        churn_probability=0.12,
        top_factors=top_factors,
    )

    assert "Customer #13085" in narrative
    assert "Healthy / Low Risk" in narrative
    assert "12.0%" in narrative
    assert "loyalty rewards" in narrative


def test_shap_and_lime_engines(tmp_path):
    """Verifies that SHAP and LIME engines run explanations on trained models."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    from src.explainability.lime_explainer import LimeExplainabilityEngine
    from src.explainability.shap_explainer import ShapExplainabilityEngine

    # Create dummy model and data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y = np.random.choice([0, 1], 50)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    model_file = tmp_path / "dummy_tree.joblib"
    joblib.dump(tree, model_file)

    # Test SHAP Engine
    shap_eng = ShapExplainabilityEngine(model_path=str(model_file), feature_names=list(X.columns))
    global_imp = shap_eng.generate_global_summary_plots(X, output_dir=str(tmp_path / "figures"))
    assert len(global_imp) == 4
    assert (tmp_path / "figures" / "shap_summary_plot.png").exists()

    single_shap = shap_eng.explain_individual_customer(X.iloc[0], customer_id=999)
    assert single_shap["customer_id"] == 999
    assert "churn_probability" in single_shap
    assert len(single_shap["top_factors"]) > 0

    # Test LIME Engine
    lime_eng = LimeExplainabilityEngine(X, model_path=str(model_file), feature_names=list(X.columns))
    lime_res = lime_eng.explain_instance(X.iloc[0], customer_id=999)
    assert lime_res["customer_id"] == 999
    assert len(lime_res["lime_factors"]) > 0

