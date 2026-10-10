"""
app/components/tooltips.py
==========================
Plain-English help text for every metric shown in the UI.

All strings are self-contained (≤3 sentences) so they can be used in
st.help(), st.expander(), or directly in report narratives.
"""

from __future__ import annotations

# ── Fairness metric tooltips ──────────────────────────────────────────────────

TOOLTIPS: dict[str, str] = {
    # ── Fairness metrics ─────────────────────────────────────────────────────
    "statistical_parity_difference": (
        "**Statistical Parity Difference (SPD)** measures the gap in positive-outcome rates "
        "between the unprivileged and privileged groups. "
        "A value of 0 means both groups receive positive outcomes at the same rate; "
        "values below −0.1 or above 0.1 indicate meaningful disparity."
    ),
    "disparate_impact": (
        "**Disparate Impact (DI)** is the ratio of positive-outcome rates: "
        "unprivileged ÷ privileged. "
        "The US EEOC '80% rule' flags a ratio below 0.8 as potentially discriminatory. "
        "Values between 0.8 and 1.25 are generally considered acceptable."
    ),
    "equal_opportunity_difference": (
        "**Equal Opportunity Difference (EOD)** compares true positive rates — "
        "among people who *deserve* a positive outcome, does the model grant it equally? "
        "A value close to 0 means qualified applicants from all groups are treated similarly."
    ),
    "average_odds_difference": (
        "**Average Odds Difference (AOD)** is the mean of the true-positive-rate gap and "
        "false-positive-rate gap between groups. "
        "It captures whether the model's errors are distributed fairly across groups."
    ),
    "theil_index": (
        "**Theil Index** measures inequality in the distribution of benefits across individuals. "
        "Borrowed from economics, a value of 0 means perfect equality; "
        "higher values indicate greater concentration of positive outcomes in one group."
    ),
    # ── Model performance metrics ─────────────────────────────────────────────
    "accuracy": (
        "**Accuracy** is the fraction of all predictions that are correct. "
        "It can be misleading on imbalanced datasets — a model that always predicts "
        "the majority class can have high accuracy while being useless."
    ),
    "precision": (
        "**Precision** is the fraction of positive predictions that are actually correct. "
        "High precision means the model rarely gives a positive outcome to someone "
        "who doesn't deserve it (low false positive rate)."
    ),
    "recall": (
        "**Recall (Sensitivity)** is the fraction of true positives that the model correctly "
        "identifies. High recall means few qualified applicants are incorrectly rejected."
    ),
    "f1": (
        "**F1 Score** is the harmonic mean of precision and recall, balancing both. "
        "It is a better single metric than accuracy when classes are imbalanced."
    ),
    "roc_auc": (
        "**ROC-AUC** measures how well the model distinguishes between classes across all "
        "possible decision thresholds. A value of 0.5 means random guessing; 1.0 is perfect."
    ),
    # ── Mitigation technique tooltips ─────────────────────────────────────────
    "reweighing": (
        "**Reweighing** is a pre-processing technique that assigns different training weights "
        "to examples to compensate for historical under/over-representation. "
        "It does not modify the features or labels — only the importance each row has during training."
    ),
    "disparate_impact_remover": (
        "**Disparate Impact Remover (DIR)** edits non-protected feature values to reduce "
        "their correlation with protected attributes, while preserving data utility. "
        "A repair level of 1.0 gives full repair (maximum fairness, possible utility loss)."
    ),
    "equalized_odds_postprocessing": (
        "**Equalized Odds Postprocessing** adjusts the model's output thresholds per group "
        "after training so that true positive and false positive rates are equalised. "
        "It is applied to predictions, not to the model itself."
    ),
    "reject_option_classification": (
        "**Reject Option Classification (ROC)** flips predictions near the decision boundary "
        "to favour the unprivileged group, exploiting the uncertainty region where the model "
        "is least confident."
    ),
    "exponentiated_gradient": (
        "**Exponentiated Gradient** (Fairlearn) is an in-processing technique that trains "
        "a sequence of models with Lagrangian fairness constraints. "
        "It directly optimises for both accuracy and a chosen fairness metric simultaneously."
    ),
    # ── Explainability tooltips ───────────────────────────────────────────────
    "shap_global": (
        "**SHAP Global Importance** shows, on average, how much each feature contributes "
        "to the model's decisions across the entire dataset. "
        "Features that appear high in protected-attribute proxy lists deserve extra scrutiny."
    ),
    "shap_local": (
        "**SHAP Local Explanation** shows *why the model made this specific prediction* "
        "for one applicant by attributing a contribution score to each feature value. "
        "Red bars push the score toward a positive outcome; blue bars push against it."
    ),
    "proxy_feature": (
        "**Proxy Feature** detection flags features that are strongly correlated with "
        "a protected attribute (via mutual information). "
        "Even if you remove race from the dataset, 'zip code' can still act as a race proxy."
    ),
    "counterfactual": (
        "**Counterfactual Check** flips a single protected attribute (e.g. sex from Male to Female) "
        "for one applicant and checks whether the model's decision changes. "
        "A changed decision is a red flag for individual-level discrimination."
    ),
}


def get_tooltip(key: str) -> str:
    """Return the help text for a given metric key, or a default if unknown."""
    return TOOLTIPS.get(
        key,
        f"No tooltip available for '{key}'. See the Ethics page for general guidance.",
    )
