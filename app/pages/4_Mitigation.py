"""
app/pages/4_Mitigation.py
=========================
Bias Mitigation — Reweighing or Threshold Optimizer debiasing.
Results are cached per (model + attribute + constraint).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.theme import (
    render_status_badge, style_plotly_fig,
    page_header, info_banner,
    PRIMARY_COLOR, ACCENT_COLOR, DANGER_COLOR,
)
from core.mitigation import apply_reweighing, apply_threshold_optimizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bias Mitigation — JobFair AI",
    page_icon="🛠️",
    layout="wide",
)
render_sidebar()
page_header("", "Step 4 — Bias Mitigation",
            "Apply debiasing algorithms (Reweighing or Threshold Optimization) to reduce group disparity while monitoring accuracy trade-offs.")

model_bundle  = st.session_state.get("model_bundle")
audit_summary = st.session_state.get("audit_summary")

if model_bundle is None or audit_summary is None:
    st.warning("⚠️ Complete **[Step 2 — Model Training](pages/2_Model.py)** and "
               "**[Step 3 — Bias Audit](pages/3_Audit.py)** first.")
    st.stop()

info_banner(
    f"Model: <b>{model_bundle.model_type}</b> &nbsp;|&nbsp; "
    f"Baseline audit: {render_status_badge(audit_summary.overall_status)}"
)

st.divider()

# ── Config ────────────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Debiasing Configuration")

c1, c2, c3 = st.columns(3)

with c1:
    mit_technique = st.selectbox(
        "Mitigation strategy",
        options=["Reweighing (Pre-Processing)", "Threshold Optimizer (Post-Processing)"],
        help="Reweighing adjusts training sample weights. Threshold Optimizer tunes group-specific decision cutoffs post-hoc.",
    )

with c2:
    target_attr = st.selectbox(
        "Protected attribute to target",
        options=model_bundle.data_bundle.protected_cols,
    )

with c3:
    if "Threshold" in mit_technique:
        constraint = st.selectbox(
            "Fairness constraint",
            options=["equalized_odds", "demographic_parity"],
            format_func=lambda x: "Equalized Odds (TPR+FPR)" if x == "equalized_odds" else "Demographic Parity (selection rate)",
        )
    else:
        constraint = "equalized_odds"
        st.caption("Reweighing works by adjusting sample weights — no constraint to set.")

if st.button("⚡ Apply Mitigation", type="primary", key="apply_mit_btn"):
    with st.spinner(f"Applying {mit_technique}…"):
        try:
            if "Reweighing" in mit_technique:
                mit_bundle = apply_reweighing(model_bundle, target_attr=target_attr)
            else:
                mit_bundle = apply_threshold_optimizer(
                    model_bundle, target_attr=target_attr, constraint=constraint
                )
            st.session_state["mitigated_bundle"] = mit_bundle
            st.success(f"✅ {mit_technique} applied successfully!")
        except Exception as exc:
            st.error(f"❌ Mitigation failed: {exc}")

# ── Comparison ────────────────────────────────────────────────────────────────
mit_bundle = st.session_state.get("mitigated_bundle")

if mit_bundle is not None:
    st.divider()
    st.markdown(f"### 📊 Before vs. After — *{mit_bundle.mitigation_method}*")

    orig_m  = mit_bundle.original_metrics
    mit_m   = mit_bundle.mitigated_metrics

    # Guard — target_attr may not be in mitigated audit if it changed
    avail_attrs = list(mit_bundle.original_audit.attribute_audits.keys())
    disp_attr   = target_attr if target_attr in avail_attrs else avail_attrs[0]

    orig_audit_attr = mit_bundle.original_audit.attribute_audits[disp_attr]
    mit_audit_attr  = mit_bundle.mitigated_audit.attribute_audits[disp_attr]

    comp_df = pd.DataFrame([
        {
            "Metric": "Accuracy",
            "Before": f"{orig_m['accuracy']:.1%}",
            "After":  f"{mit_m['accuracy']:.1%}",
            "Change": f"{mit_m['accuracy'] - orig_m['accuracy']:+.1%}",
        },
        {
            "Metric": "ROC-AUC",
            "Before": f"{orig_m['roc_auc']:.3f}",
            "After":  f"{mit_m['roc_auc']:.3f}",
            "Change": f"{mit_m['roc_auc'] - orig_m['roc_auc']:+.3f}",
        },
        {
            "Metric": "Disparate Impact",
            "Before": f"{orig_audit_attr.disparate_impact:.2f}",
            "After":  f"{mit_audit_attr.disparate_impact:.2f}",
            "Change": f"{mit_audit_attr.disparate_impact - orig_audit_attr.disparate_impact:+.2f}",
        },
        {
            "Metric": "Demographic Parity Diff",
            "Before": f"{orig_audit_attr.demographic_parity_diff:+.3f}",
            "After":  f"{mit_audit_attr.demographic_parity_diff:+.3f}",
            "Change": f"{mit_audit_attr.demographic_parity_diff - orig_audit_attr.demographic_parity_diff:+.3f}",
        },
        {
            "Metric": "Equal Opportunity Diff",
            "Before": f"{orig_audit_attr.equal_opportunity_diff:+.3f}",
            "After":  f"{mit_audit_attr.equal_opportunity_diff:+.3f}",
            "Change": f"{mit_audit_attr.equal_opportunity_diff - orig_audit_attr.equal_opportunity_diff:+.3f}",
        },
    ])

    st.dataframe(comp_df, use_container_width=True)

    stat_c1, stat_c2 = st.columns(2)
    with stat_c1:
        st.markdown(
            f"**Baseline:** {render_status_badge(mit_bundle.original_audit.overall_status)}",
            unsafe_allow_html=True,
        )
    with stat_c2:
        st.markdown(
            f"**After mitigation:** {render_status_badge(mit_bundle.mitigated_audit.overall_status)}",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Accuracy vs. Disparate Impact")

    tradeoff_df = pd.DataFrame([
        {"Model": "Before", "Accuracy": orig_m["accuracy"], "Disparate Impact": orig_audit_attr.disparate_impact},
        {"Model": "After",  "Accuracy": mit_m["accuracy"],  "Disparate Impact": mit_audit_attr.disparate_impact},
    ])
    fig_tradeoff = px.bar(
        tradeoff_df, x="Model", y=["Accuracy", "Disparate Impact"],
        barmode="group",
        color_discrete_sequence=[ACCENT_COLOR, PRIMARY_COLOR],
    )
    fig_tradeoff.add_hline(
        y=0.80, line_dash="dash", line_color=DANGER_COLOR,
        annotation_text="80% Rule Threshold",
    )
    st.plotly_chart(
        style_plotly_fig(fig_tradeoff, title=f"Accuracy vs. Disparate Impact ({disp_attr})"),
        use_container_width=True,
    )

    st.divider()
    st.page_link("pages/5_Explainability.py", label="Proceed to Step 5 — Explainability →", icon="💡")
