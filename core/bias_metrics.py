"""
core/bias_metrics.py
====================
Bias Auditing Engine for JobFair AI.

Computes group fairness and individual fairness metrics:
1. Disparate Impact Ratio (DIR): P(Y_hat=1 | Unprivileged) / P(Y_hat=1 | Privileged)
2. Demographic Parity Difference (DPD): P(Y_hat=1 | Unprivileged) - P(Y_hat=1 | Privileged)
3. Equal Opportunity Difference (EOD): TPR_unprivileged - TPR_privileged
4. Average Odds Difference (AOD): 0.5 * [(FPR_unprivileged - FPR_privileged) + (TPR_unprivileged - TPR_privileged)]
5. Theil Index: Individual fairness metric measuring inequality in benefit allocation.

Also provides multi-group / intersectional bias auditing and overall audit status determination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, DatasetMetric


@dataclass
class SingleAttributeAudit:
    """Audit results for one protected attribute."""
    attribute_name: str
    privileged_group: Any
    unprivileged_group: Any
    disparate_impact: float
    demographic_parity_diff: float
    equal_opportunity_diff: float
    average_odds_diff: float
    theil_index: float
    selection_rate_priv: float
    selection_rate_unpriv: float
    tpr_priv: float
    tpr_unpriv: float
    fpr_priv: float
    fpr_unpriv: float
    status: str  # "PASS", "WARNING", "FAIL"
    alerts: List[str] = field(default_factory=list)


@dataclass
class AuditReportSummary:
    """Overall audit report summarizing all protected attributes."""
    overall_status: str  # "PASS", "WARNING", "FAIL"
    attribute_audits: Dict[str, SingleAttributeAudit]
    intersectional_audits: Dict[str, Dict[str, Any]]
    total_violations: int
    recommendation: str


def compute_group_fairness_metrics(
    dataset_true: BinaryLabelDataset,
    dataset_pred: BinaryLabelDataset,
    unprivileged_groups: List[Dict[str, int]],
    privileged_groups: List[Dict[str, int]],
    attr_name: str,
) -> SingleAttributeAudit:
    """
    Computes standard group fairness metrics using AIF360 ClassificationMetric.
    """
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    # 1. Selection Rates (Positive Prediction Rates)
    sr_priv = metric.base_rate(privileged=True)
    sr_unpriv = metric.base_rate(privileged=False)

    # 2. Disparate Impact
    # Avoid division by zero
    if sr_priv == 0:
        di = 1.0 if sr_unpriv == 0 else 0.0
    else:
        di = sr_unpriv / sr_priv

    # 3. Demographic Parity Difference
    dpd = sr_unpriv - sr_priv

    # 4. True Positive Rates (Equal Opportunity)
    tpr_priv = metric.true_positive_rate(privileged=True)
    tpr_unpriv = metric.true_positive_rate(privileged=False)
    eod = tpr_unpriv - tpr_priv

    # 5. False Positive Rates
    fpr_priv = metric.false_positive_rate(privileged=True)
    fpr_unpriv = metric.false_positive_rate(privileged=False)
    
    # 6. Average Odds Difference
    aod = 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))

    # 7. Theil Index (Individual Fairness / Generalized Entropy Index with alpha=1)
    try:
        theil = metric.theil_index()
    except Exception:
        # Fallback estimation if AIF360 internal fails
        b = dataset_pred.labels.ravel() + 1e-6
        mu = np.mean(b)
        theil = float(np.mean((b / mu) * np.log(b / mu)))

    # Determine status & alerts based on standard regulatory thresholds:
    # DIR acceptable range: [0.8, 1.25] (80% rule / 4/5ths rule)
    # DPD, EOD, AOD acceptable range: [-0.1, 0.1]
    alerts = []
    if di < 0.8:
        alerts.append(f"Disparate Impact ratio is {di:.2f}, failing the 80% (4/5ths) rule (< 0.80).")
    elif di > 1.25:
        alerts.append(f"Disparate Impact ratio is {di:.2f}, indicating reverse disparity (> 1.25).")

    if abs(dpd) > 0.1:
        alerts.append(f"Demographic Parity Difference is {dpd:+.3f} (exceeds ±0.10 threshold).")

    if abs(eod) > 0.1:
        alerts.append(f"Equal Opportunity Difference is {eod:+.3f} (exceeds ±0.10 threshold).")

    if abs(aod) > 0.1:
        alerts.append(f"Average Odds Difference is {aod:+.3f} (exceeds ±0.10 threshold).")

    if len(alerts) == 0:
        status = "PASS"
    elif di < 0.7 or abs(eod) > 0.15 or abs(aod) > 0.15:
        status = "FAIL"
    else:
        status = "WARNING"

    return SingleAttributeAudit(
        attribute_name=attr_name,
        privileged_group=privileged_groups,
        unprivileged_group=unprivileged_groups,
        disparate_impact=float(di),
        demographic_parity_diff=float(dpd),
        equal_opportunity_diff=float(eod),
        average_odds_diff=float(aod),
        theil_index=float(theil),
        selection_rate_priv=float(sr_priv),
        selection_rate_unpriv=float(sr_unpriv),
        tpr_priv=float(tpr_priv),
        tpr_unpriv=float(tpr_unpriv),
        fpr_priv=float(fpr_priv),
        fpr_unpriv=float(fpr_unpriv),
        status=status,
        alerts=alerts,
    )


def compute_intersectional_fairness(
    df_raw: pd.DataFrame,
    y_pred: np.ndarray,
    protected_attrs: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Computes selection rate across combinations of 2 protected attributes (e.g. sex x race).
    """
    if len(protected_attrs) < 2:
        return {}

    attr1, attr2 = protected_attrs[0], protected_attrs[1]
    df_temp = df_raw.copy()
    df_temp["y_pred"] = y_pred

    grouped = df_temp.groupby([attr1, attr2])["y_pred"].agg(
        total_count="count",
        favorable_count="sum",
        selection_rate="mean",
    ).reset_index()

    overall_sr = df_temp["y_pred"].mean()
    grouped["disparate_impact_vs_overall"] = grouped["selection_rate"] / (overall_sr if overall_sr > 0 else 1.0)

    results = {}
    for idx, row in grouped.iterrows():
        key = f"{row[attr1]} × {row[attr2]}"
        results[key] = {
            attr1: row[attr1],
            attr2: row[attr2],
            "total_count": int(row["total_count"]),
            "favorable_count": int(row["favorable_count"]),
            "selection_rate": float(row["selection_rate"]),
            "disparate_impact_vs_overall": float(row["disparate_impact_vs_overall"]),
        }
    return results


def run_full_bias_audit(
    model_bundle: Any,
) -> AuditReportSummary:
    """
    Runs a complete bias audit across all protected attributes in the model_bundle.
    """
    bundle = model_bundle.data_bundle
    y_test = model_bundle.y_test
    y_pred = model_bundle.y_pred

    # Build AIF360 true and pred datasets
    # Create dataset_true from bundle's test dataset or reconstruct
    X_test_df = pd.DataFrame(model_bundle.X_test, columns=bundle.feature_names)
    X_test_df["label"] = y_test

    dataset_true = BinaryLabelDataset(
        df=X_test_df,
        label_names=["label"],
        protected_attribute_names=bundle.protected_cols,
        favorable_label=1,
        unfavorable_label=0,
    )

    dataset_pred = dataset_true.copy()
    dataset_pred.labels = y_pred.reshape(-1, 1)

    attribute_audits: Dict[str, SingleAttributeAudit] = {}
    total_violations = 0

    for attr in bundle.protected_cols:
        # Find matching group dict from bundle or construct default binary group dict
        priv_val = 1.0
        unpriv_val = 0.0
        
        # Check if bundle has explicit mapping
        for g in bundle.privileged_groups:
            if attr in g:
                priv_val = g[attr]
                break
        for g in bundle.unprivileged_groups:
            if attr in g:
                unpriv_val = g[attr]
                break

        priv = [{attr: priv_val}]
        unpriv = [{attr: unpriv_val}]

        audit = compute_group_fairness_metrics(
            dataset_true=dataset_true,
            dataset_pred=dataset_pred,
            unprivileged_groups=unpriv,
            privileged_groups=priv,
            attr_name=attr,
        )
        attribute_audits[attr] = audit
        total_violations += len(audit.alerts)

    # Intersectional analysis
    intersectional = compute_intersectional_fairness(
        df_raw=bundle.df_raw.iloc[len(bundle.df_raw) - len(y_test):],  # test split alignment
        y_pred=y_pred,
        protected_attrs=bundle.protected_cols,
    )

    # Determine overall status
    statuses = [a.status for a in attribute_audits.values()]
    if "FAIL" in statuses:
        overall_status = "FAIL"
        recommendation = "High bias detected failing regulatory thresholds. Mitigation is mandatory before deployment."
    elif "WARNING" in statuses:
        overall_status = "WARNING"
        recommendation = "Moderate bias detected violating some fairness guidelines. Mitigation recommended."
    else:
        overall_status = "PASS"
        recommendation = "Model meets all major fairness thresholds across evaluated protected attributes."

    return AuditReportSummary(
        overall_status=overall_status,
        attribute_audits=attribute_audits,
        intersectional_audits=intersectional,
        total_violations=total_violations,
        recommendation=recommendation,
    )
