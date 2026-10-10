"""
core/explainability.py
======================
Model Explainability Engine for JobFair AI using SHAP (SHapley Additive exPlanations).

Provides:
1. Global Feature Importance (mean |SHAP| value per feature).
2. Local Waterfall Explanations for individual applicant decisions.
3. Demographic Group SHAP Disparity analysis to reveal feature-level bias mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from core.models import ModelBundle


@dataclass
class ExplainabilityReport:
    """Container holding global and local SHAP explanations."""
    shap_values: np.ndarray
    expected_value: float
    feature_names: List[str]
    global_importance: pd.DataFrame
    group_disparity: Optional[pd.DataFrame] = None


def compute_shap_explanations(
    model_bundle: ModelBundle,
    max_samples: int = 200,
) -> ExplainabilityReport:
    """
    Computes SHAP values for the test set.
    Handles tree models, linear models, and general classifiers safely.
    """
    model = model_bundle.model
    X_test = model_bundle.X_test.iloc[:max_samples]
    feature_names = model_bundle.feature_names

    try:
        if hasattr(model, "predict_proba") and "Forest" in model_bundle.model_type or "XGBoost" in model_bundle.model_type:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            # Handle multi-output / binary classification shape
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # positive class
            elif shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1]
        elif "Logistic" in model_bundle.model_type:
            explainer = shap.LinearExplainer(model, X_test)
            shap_vals = explainer.shap_values(X_test)
            base_val = explainer.expected_value
        else:
            explainer = shap.Explainer(model.predict, X_test)
            exp = explainer(X_test)
            shap_vals = exp.values
            base_val = exp.base_values.mean()
    except Exception:
        # Fallback to Kernel/Explainer for complex wrappers
        explainer = shap.Explainer(model.predict, X_test)
        exp = explainer(X_test)
        shap_vals = exp.values
        base_val = float(exp.base_values.mean()) if hasattr(exp.base_values, "mean") else 0.5

    # Compute global feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    global_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP Value|": mean_abs_shap,
    }).sort_values("Mean |SHAP Value|", ascending=False).reset_index(drop=True)

    # Compute Demographic SHAP disparity for first protected col
    group_disparity = None
    if model_bundle.data_bundle.protected_cols:
        prot_col = model_bundle.data_bundle.protected_cols[0]
        if prot_col in X_test.columns:
            group_mask = X_test[prot_col] == 1
            shap_priv = shap_vals[group_mask].mean(axis=0) if group_mask.any() else np.zeros(len(feature_names))
            shap_unpriv = shap_vals[~group_mask].mean(axis=0) if (~group_mask).any() else np.zeros(len(feature_names))
            
            group_disparity = pd.DataFrame({
                "Feature": feature_names,
                "Privileged Mean SHAP": shap_priv,
                "Unprivileged Mean SHAP": shap_unpriv,
                "SHAP Impact Disparity": shap_priv - shap_unpriv,
            }).sort_values("SHAP Impact Disparity", ascending=False).reset_index(drop=True)

    return ExplainabilityReport(
        shap_values=shap_vals,
        expected_value=float(base_val),
        feature_names=feature_names,
        global_importance=global_importance,
        group_disparity=group_disparity,
    )
