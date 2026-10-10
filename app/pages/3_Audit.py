"""
app/pages/3_Audit.py
====================
Bias Auditing — computes group fairness & individual fairness metrics.
Results are cached so switching attributes doesn't recompute metrics.
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
    render_metric_card, render_status_badge,
    style_plotly_fig, page_header, info_banner,
    PRIMARY_COLOR, ACCENT_COLOR,
)
from core.bias_metrics import run_full_bias_audit

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bias Audit — JobFair AI",
    page_icon="⚖️",
    layout="wide",
)
render_sidebar()
page_header("", "Step 3 — Bias Audit",
            "Measure demographic disparity metrics against EEOC 80% rule, Demographic Parity, Equal Opportunity, and Statistical Parity.")

model_bundle = st.session_state.get("model_bundle")

if model_bundle is None:
    st.warning("⚠️ No trained model yet. Complete **[Step 2 — Model Training](pages/2_Model.py)** first.")
    st.stop()

info_banner(f"Model: <b>{model_bundle.model_type}</b> on <b>{model_bundle.data_bundle.dataset_name}</b>")

st.divider()

# ── Run audit (with caching in session state) ─────────────────────────────────
col_btn, col_status = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("🔍 Run Bias Audit", type="primary", key="run_audit_btn")

audit_summary = st.session_state.get("audit_summary")

if run_clicked or audit_summary is None:
    with st.spinner("Computing fairness metrics across all protected attributes…"):
        try:
            audit_summary = run_full_bias_audit(model_bundle)
            st.session_state["audit_summary"] = audit_summary
        except Exception as exc:
            st.error(f"❌ Audit failed: {exc}")
            st.stop()

with col_status:
    if audit_summary is not None:
        badge = render_status_badge(audit_summary.overall_status)
        st.markdown(
            f"<div style='padding:0.45rem 0;font-size:0.88rem;color:#4B5563;'>"
            f"Overall result: {badge}</div>",
            unsafe_allow_html=True,
        )

# ── Audit results ─────────────────────────────────────────────────────────────
if audit_summary is not None:
    # Status card
    badge_html = render_status_badge(audit_summary.overall_status)
    st.markdown(
        f"""
        <div class="jf-card jf-card-accent">
            <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.4rem;">
                <span style="font-weight:700;font-size:1rem;color:#111827;">Overall Audit Result:</span>
                {badge_html}
            </div>
            <p style="margin:0;font-size:0.88rem;color:#4B5563;">{audit_summary.recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Metrics by Protected Attribute")

    attributes = list(audit_summary.attribute_audits.keys())
    selected_attr = st.selectbox("Protected Attribute", options=attributes, key="audit_attr_select")
    audit = audit_summary.attribute_audits[selected_attr]

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        di_ok = 0.8 <= audit.disparate_impact <= 1.25
        render_metric_card(
            "Disparate Impact",
            f"{audit.disparate_impact:.2f}",
            "✓ Pass (≥0.80)" if di_ok else "✕ Fail (<0.80)",
        )
    with m2:
        dpd_ok = abs(audit.demographic_parity_diff) <= 0.1
        render_metric_card(
            "Dem. Parity Diff",
            f"{audit.demographic_parity_diff:+.3f}",
            "✓ Pass (|x|≤0.10)" if dpd_ok else "✕ Fail",
        )
    with m3:
        eod_ok = abs(audit.equal_opportunity_diff) <= 0.1
        render_metric_card(
            "Equal Opportunity",
            f"{audit.equal_opportunity_diff:+.3f}",
            "✓ Pass (TPR diff)" if eod_ok else "✕ Fail",
        )
    with m4:
        aod_ok = abs(audit.average_odds_diff) <= 0.1
        render_metric_card(
            "Avg. Odds Diff",
            f"{audit.average_odds_diff:+.3f}",
            "✓ Pass (|x|≤0.10)" if aod_ok else "✕ Fail",
        )
    with m5:
        render_metric_card("Theil Index", f"{audit.theil_index:.4f}", "Individual fairness")

    st.markdown("<br>", unsafe_allow_html=True)

    if audit.alerts:
        for alert in audit.alerts:
            st.warning(f"⚠️ {alert}")
    else:
        st.success(f"✅ All metrics for `{selected_attr}` are within benchmark thresholds.")

    # Group rate comparison
    st.markdown("#### Selection Rate & True Positive Rate by Group")
    chart_df = pd.DataFrame([
        {"Group": "Privileged",   "Selection Rate": audit.selection_rate_priv,  "TPR": audit.tpr_priv},
        {"Group": "Unprivileged", "Selection Rate": audit.selection_rate_unpriv, "TPR": audit.tpr_unpriv},
    ])
    fig_comp = px.bar(
        chart_df, x="Group", y=["Selection Rate", "TPR"],
        barmode="group",
        color_discrete_sequence=[PRIMARY_COLOR, ACCENT_COLOR],
    )
    fig_comp.update_layout(yaxis_tickformat=".1%", height=310)
    st.plotly_chart(
        style_plotly_fig(fig_comp, title=f"Group Rates — {selected_attr}"),
        use_container_width=True,
    )

    # Intersectional
    if audit_summary.intersectional_audits:
        st.divider()
        st.markdown("### 🔀 Intersectional Subgroup Breakdown")
        st.markdown("Selection rates across combinations of two protected attributes:")
        inter_df = pd.DataFrame.from_dict(audit_summary.intersectional_audits, orient="index")
        if not inter_df.empty:
            st.dataframe(
                inter_df.style.format({
                    "selection_rate": "{:.1%}",
                    "disparate_impact_vs_overall": "{:.2f}",
                }),
                use_container_width=True,
            )

    st.divider()
    st.page_link("pages/4_Mitigation.py", label="Proceed to Step 4 — Bias Mitigation →", icon="🛠️")
