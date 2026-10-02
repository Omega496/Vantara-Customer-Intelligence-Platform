"""Plain-language explanation generator translating technical feature attributions into actionable marketing insights."""

from typing import Any, Dict, List, Optional

import pandas as pd

FEATURE_FRIENDLY_NAMES = {
    "recency_days": "days since last purchase",
    "frequency": "total lifetime order count",
    "total_spend": "total historical net spend",
    "avg_basket_size": "average order basket size",
    "velocity_acceleration": "recent purchasing pace relative to historical rate",
    "recent_orders_last_90d": "orders placed in the last 90 days",
    "recent_orders_ratio": "percentage of lifetime orders in the last 90 days",
    "inter_purchase_mean": "average days between orders",
    "inter_purchase_std": "consistency of purchase intervals",
    "return_line_rate": "percentage of transactions returned",
    "return_spend_ratio": "ratio of refund value to gross purchases",
    "discount_sensitivity": "share of purchases bought on promotional discount",
    "engagement_score": "overall customer health score",
}


def generate_plain_language_narrative(
    customer_id: Any,
    churn_probability: float,
    top_factors: List[Dict[str, Any]],
    customer_data: Optional[pd.Series] = None,
) -> str:
    """Generates an intuitive, non-technical plain-English briefing for retention teams.

    Args:
        customer_id: The identifier of the customer.
        churn_probability: Model predicted churn probability (0.0 to 1.0).
        top_factors: Ranked list of dictionaries with keys:
                     ['feature', 'value', 'shap_attribution', 'direction']
        customer_data: Optional raw customer record containing metrics like total_spend.

    Returns:
        Formatted multi-sentence plain English narrative explanation.
    """
    prob_pct = churn_probability * 100.0

    # Risk Tier categorization
    if churn_probability >= 0.75:
        tier_str = "High Churn Risk"
        urgency = "Immediate proactive retention outreach is recommended."
    elif churn_probability >= 0.45:
        tier_str = "Borderline / Moderate Risk"
        urgency = "Proactive monitoring and targeted re-engagement campaigns are advised."
    else:
        tier_str = "Healthy / Low Risk"
        urgency = "Customer is in good standing; focus on loyalty rewards and cross-sell."

    # Identify primary drivers increasing churn risk and protective factors
    risk_drivers = [f for f in top_factors if f.get("shap_attribution", 0) > 0]
    protective_factors = [f for f in top_factors if f.get("shap_attribution", 0) < 0]

    # Synthesize risk drivers
    driver_phrases = []
    for d in risk_drivers[:2]:
        fname = d["feature"]
        val = d["value"]
        friendly = FEATURE_FRIENDLY_NAMES.get(fname, fname.replace("_", " "))
        if "recency" in fname:
            driver_phrases.append(f"{val:.0f} days of inactivity")
        elif "velocity" in fname:
            if val < 0.5:
                driver_phrases.append("a significant deceleration in recent ordering pace")
            else:
                driver_phrases.append("irregular purchasing velocity")
        elif "return" in fname:
            driver_phrases.append(f"an elevated return rate of {val*100:.1f}%")
        elif "recent_orders" in fname:
            driver_phrases.append(f"only {val:.0f} orders in the last 90 days")
        else:
            driver_phrases.append(f"{friendly} ({val})")

    # Synthesize protective factors
    protective_phrases = []
    for p in protective_factors[:2]:
        fname = p["feature"]
        val = p["value"]
        friendly = FEATURE_FRIENDLY_NAMES.get(fname, fname.replace("_", " "))
        if "spend" in fname:
            protective_phrases.append(f"strong lifetime spend of £{val:,.2f}")
        elif "frequency" in fname:
            protective_phrases.append(f"a history of {val:.0f} completed orders")
        elif "engagement" in fname:
            protective_phrases.append(f"a resilient customer health score of {val:.0f}/100")
        else:
            protective_phrases.append(f"{friendly} ({val})")

    # Construct the cohesive narrative
    narrative_parts = [
        f"Customer #{customer_id} is evaluated as **{tier_str}** with a predicted churn probability of **{prob_pct:.1f}%**."
    ]

    if driver_phrases:
        drivers_text = " and ".join(driver_phrases)
        narrative_parts.append(
            f"The primary factors driving this risk are **{drivers_text}**."
        )

    if protective_phrases:
        protective_text = " and ".join(protective_phrases)
        narrative_parts.append(
            f"Conversely, the customer's risk is mitigated by **{protective_text}**."
        )

    narrative_parts.append(urgency)
    return " ".join(narrative_parts)
