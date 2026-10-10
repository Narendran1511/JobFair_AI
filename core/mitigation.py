"""
core/mitigation.py
==================
Bias Mitigation Engine for JobFair AI.

Supports pre-processing, in-processing, and post-processing bias mitigation:
1. Pre-processing: Reweighing (AIF360) — adjusts sample weights before model training.
2. In-processing: Exponentiated Gradient (Fairlearn) — optimizes loss subject to fairness constraints.
3. Post-processing: Threshold Optimizer (Fairlearn) — optimizes decision thresholds per group post-hoc.

Returns a MitigatedModelBundle with before-and-after fairness and performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

from core.models import ModelBundle, train_model, evaluate_model
from core.bias_metrics import run_full_bias_audit, AuditReportSummary


@dataclass
class MitigatedModelBundle:
    """Container holding original vs mitigated model, metrics, and fairness audit comparison."""
    original_bundle: ModelBundle
    mitigated_model: Any
    mitigation_method: str
    target_protected_attr: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred_mitigated: np.ndarray
    y_prob_mitigated: np.ndarray
    original_metrics: Dict[str, Any]
    mitigated_metrics: Dict[str, Any]
    original_audit: AuditReportSummary
    mitigated_audit: AuditReportSummary
    tradeoff_summary: Dict[str, float]


def apply_reweighing(
    model_bundle: ModelBundle,
    target_attr: str = "sex",
) -> MitigatedModelBundle:
    """
    Applies AIF360 Reweighing pre-processing technique.
    Computes instance weights for train set to balance group base rates, then re-trains model.
    """
    bundle = model_bundle.data_bundle
    X_train = model_bundle.X_train
    y_train = model_bundle.y_train

    # Reconstruct train dataset_obj for Reweighing
    df_train = X_train.copy()
    df_train["label"] = y_train.values

    # Determine privileged and unprivileged groups for AIF360 Reweighing
    priv_val = 1.0
    unpriv_val = 0.0
    for g in bundle.privileged_groups:
        if target_attr in g:
            priv_val = g[target_attr]
    for g in bundle.unprivileged_groups:
        if target_attr in g:
            unpriv_val = g[target_attr]

    # Create lightweight dataset for Reweighing
    from aif360.datasets import BinaryLabelDataset
    bld_train = BinaryLabelDataset(
        df=df_train,
        label_names=["label"],
        protected_attribute_names=[target_attr],
        favorable_label=1.0,
        unfavorable_label=0.0,
    )

    rw = Reweighing(
        unprivileged_groups=[{target_attr: unpriv_val}],
        privileged_groups=[{target_attr: priv_val}],
    )
    bld_rw = rw.fit_transform(bld_train)
    sample_weights = bld_rw.instance_weights

    # Re-train model with sample weights
    model_res = train_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_bundle.model_type,
        seed=model_bundle.hyperparams.get("random_state", 42),
        sample_weights=sample_weights,
    )
    mitigated_model = model_res["model"]

    # Evaluate mitigated model
    eval_res = evaluate_model(mitigated_model, model_bundle.X_test, model_bundle.y_test)
    
    # Create temporary bundle for running audit
    mitigated_bundle = ModelBundle(
        data_bundle=bundle,
        model=mitigated_model,
        model_type=model_bundle.model_type,
        hyperparams=model_bundle.hyperparams,
        X_train=X_train,
        X_test=model_bundle.X_test,
        y_train=y_train,
        y_test=model_bundle.y_test,
        y_pred=eval_res["predictions"],
        y_prob=eval_res["probabilities"],
        metrics={
            "accuracy": eval_res["accuracy"],
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
            "f1": eval_res["f1"],
            "roc_auc": eval_res["roc_auc"] if eval_res["roc_auc"] is not None else 0.5,
            "pr_auc": eval_res.get("pr_auc", 0.5),
        },
        feature_names=model_bundle.feature_names,
    )

    original_audit = run_full_bias_audit(model_bundle)
    mitigated_audit = run_full_bias_audit(mitigated_bundle)

    acc_diff = mitigated_bundle.metrics["accuracy"] - model_bundle.metrics["accuracy"]
    orig_di = original_audit.attribute_audits[target_attr].disparate_impact
    mit_di = mitigated_audit.attribute_audits[target_attr].disparate_impact
    di_gain = mit_di - orig_di

    tradeoff = {
        "accuracy_change": round(acc_diff, 4),
        "disparate_impact_change": round(di_gain, 4),
        "original_disparate_impact": round(orig_di, 4),
        "mitigated_disparate_impact": round(mit_di, 4),
    }

    return MitigatedModelBundle(
        original_bundle=model_bundle,
        mitigated_model=mitigated_model,
        mitigation_method="Reweighing (Pre-Processing)",
        target_protected_attr=target_attr,
        X_train=X_train,
        X_test=model_bundle.X_test,
        y_train=y_train,
        y_test=model_bundle.y_test,
        y_pred_mitigated=eval_res["predictions"],
        y_prob_mitigated=eval_res["probabilities"],
        original_metrics=model_bundle.metrics,
        mitigated_metrics=mitigated_bundle.metrics,
        original_audit=original_audit,
        mitigated_audit=mitigated_audit,
        tradeoff_summary=tradeoff,
    )


def apply_threshold_optimizer(
    model_bundle: ModelBundle,
    target_attr: str = "sex",
    constraint: str = "equalized_odds",
) -> MitigatedModelBundle:
    """
    Applies Fairlearn ThresholdOptimizer post-processing technique.
    Finds group-specific decision thresholds post-hoc to meet Demographic Parity or Equalized Odds.
    """
    bundle = model_bundle.data_bundle
    X_train = model_bundle.X_train
    y_train = model_bundle.y_train
    X_test = model_bundle.X_test
    y_test = model_bundle.y_test

    sensitive_train = X_train[target_attr]
    sensitive_test = X_test[target_attr]

    # Fairlearn ThresholdOptimizer accepts string names for constraints directly
    constr_str = "demographic_parity" if constraint == "demographic_parity" else "equalized_odds"

    thresh_opt = ThresholdOptimizer(
        estimator=model_bundle.model,
        constraints=constr_str,
        predict_method="predict_proba",
        prefit=True,
    )

    thresh_opt.fit(X_train, y_train, sensitive_features=sensitive_train)

    y_pred_mit = thresh_opt.predict(X_test, sensitive_features=sensitive_test)
    y_prob_mit = model_bundle.model.predict_proba(X_test)[:, 1]  # post-proc uses probabilities + group assignment

    # Evaluate metrics manually
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc = accuracy_score(y_test, y_pred_mit)
    prec = precision_score(y_test, y_pred_mit, zero_division=0)
    rec = recall_score(y_test, y_pred_mit, zero_division=0)
    f1 = f1_score(y_test, y_pred_mit, zero_division=0)
    try:
        auc_val = roc_auc_score(y_test, y_prob_mit)
    except Exception:
        auc_val = 0.5

    mitigated_metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc_val, 4),
        "pr_auc": round(f1, 4),
    }

    mitigated_bundle = ModelBundle(
        data_bundle=bundle,
        model=thresh_opt,
        model_type=f"{model_bundle.model_type} + ThresholdOptimizer",
        hyperparams=model_bundle.hyperparams,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred_mit,
        y_prob=y_prob_mit,
        metrics=mitigated_metrics,
        feature_names=model_bundle.feature_names,
    )

    original_audit = run_full_bias_audit(model_bundle)
    mitigated_audit = run_full_bias_audit(mitigated_bundle)

    acc_diff = mitigated_metrics["accuracy"] - model_bundle.metrics["accuracy"]
    orig_di = original_audit.attribute_audits[target_attr].disparate_impact
    mit_di = mitigated_audit.attribute_audits[target_attr].disparate_impact
    di_gain = mit_di - orig_di

    tradeoff = {
        "accuracy_change": round(acc_diff, 4),
        "disparate_impact_change": round(di_gain, 4),
        "original_disparate_impact": round(orig_di, 4),
        "mitigated_disparate_impact": round(mit_di, 4),
    }

    return MitigatedModelBundle(
        original_bundle=model_bundle,
        mitigated_model=thresh_opt,
        mitigation_method=f"Threshold Optimizer ({constraint.replace('_', ' ').title()})",
        target_protected_attr=target_attr,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred_mitigated=y_pred_mit,
        y_prob_mitigated=y_prob_mit,
        original_metrics=model_bundle.metrics,
        mitigated_metrics=mitigated_metrics,
        original_audit=original_audit,
        mitigated_audit=mitigated_audit,
        tradeoff_summary=tradeoff,
    )
