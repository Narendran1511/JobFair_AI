"""
app/pages/5_Explainability.py
==============================
Explainability & SHAP — global importance, individual candidate breakdowns,
and demographic SHAP disparity. Results cached in session state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.theme import (
    render_metric_card, style_plotly_fig,
    page_header, info_banner,
    PRIMARY_COLOR, ACCENT_COLOR, DANGER_COLOR,
)
from core.explainability import compute_shap_explanations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainability — JobFair AI",
    page_icon="💡",
    layout="wide",
)
render_sidebar()
page_header("", "Step 5 — Explainability & SHAP",
            "Evaluate global feature importance, individual candidate feature attribution, and demographic SHAP disparity.")

model_bundle = st.session_state.get("model_bundle")

if model_bundle is None:
    st.warning("⚠️ No trained model. Go back to **[Step 2 — Model Training](pages/2_Model.py)**.")
    st.stop()

info_banner(f"Model: <b>{model_bundle.model_type}</b> &nbsp;|&nbsp; Dataset: <b>{model_bundle.data_bundle.dataset_name}</b>")

st.divider()

# ── Compute SHAP (session-cached) ─────────────────────────────────────────────
col_btn, col_note = st.columns([1, 3])
with col_btn:
    compute_clicked = st.button("⚡ Compute SHAP", type="primary", key="compute_shap_btn")
with col_note:
    st.markdown(
        "<p style='padding:0.45rem 0;font-size:0.85rem;color:#6B7280;'>"
        "SHAP runs on up to 200 test samples. Tree models are fast; "
        "Logistic Regression uses a linear explainer.</p>",
        unsafe_allow_html=True,
    )

report = st.session_state.get("shap_report")

if compute_clicked or report is None:
    with st.spinner("Computing SHAP explanations — may take 10–30 seconds for tree models…"):
        try:
            report = compute_shap_explanations(model_bundle, max_samples=200)
            st.session_state["shap_report"] = report
            st.success("✅ SHAP values computed successfully!")
        except Exception as exc:
            st.error(f"❌ SHAP failed: {exc}")
            st.stop()

if report is not None:
    # ── Global Feature Importance ─────────────────────────────────────────────
    st.markdown("### 🌐 Global Feature Importance")
    st.markdown(
        "<p style='font-size:0.87rem;color:#6B7280;margin-top:-0.3rem;'>Mean |SHAP| — "
        "features ranked by average absolute impact on predictions.</p>",
        unsafe_allow_html=True,
    )

    top_features = report.global_importance.head(15).sort_values("Mean |SHAP Value|")
    fig_imp = go.Figure(
        go.Bar(
            y=top_features["Feature"],
            x=top_features["Mean |SHAP Value|"],
            orientation="h",
            marker_color=PRIMARY_COLOR,
            marker_opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    st.plotly_chart(
        style_plotly_fig(fig_imp, title="Top 15 Features by Mean |SHAP Value|", height=380),
        use_container_width=True,
    )

    st.divider()

    # ── Individual Candidate ──────────────────────────────────────────────────
    st.markdown("### 👤 Individual Candidate Explanation")

    candidate_idx = st.slider(
        "Candidate index (test set)",
        min_value=0,
        max_value=len(report.shap_values) - 1,
        value=0,
        help="Use this to inspect how features contributed to the decision for any individual test sample.",
    )

    sample_shap     = report.shap_values[candidate_idx]
    sample_features = model_bundle.X_test.iloc[candidate_idx]
    actual_pred     = model_bundle.y_pred[candidate_idx]
    pred_prob       = model_bundle.y_prob[candidate_idx]

    c_meta1, c_meta2 = st.columns(2)
    with c_meta1:
        label = "Favorable (1)" if actual_pred == 1 else "Unfavorable (0)"
        render_metric_card("Predicted Decision", label, f"Candidate #{candidate_idx}")
    with c_meta2:
        render_metric_card("Predicted Probability", f"{pred_prob:.1%}", "Positive class confidence")

    st.markdown("<br>", unsafe_allow_html=True)

    contrib_df = pd.DataFrame({
        "Feature": report.feature_names,
        "Feature Value": sample_features.values,
        "SHAP Contribution": sample_shap,
    }).sort_values(by="SHAP Contribution", key=abs, ascending=True).tail(10)

    colors = [ACCENT_COLOR if v >= 0 else DANGER_COLOR for v in contrib_df["SHAP Contribution"]]

    fig_wf = go.Figure(
        go.Bar(
            y=contrib_df["Feature"],
            x=contrib_df["SHAP Contribution"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3g}" for v in contrib_df["Feature Value"]],
            hovertemplate="<b>%{y}</b> (val=%{text})<br>SHAP: %{x:+.4f}<extra></extra>",
        )
    )
    fig_wf.add_vline(x=0, line_color="#D1D5DB", line_width=1)
    st.plotly_chart(
        style_plotly_fig(fig_wf, title=f"Top Feature Contributions — Candidate #{candidate_idx}", height=350),
        use_container_width=True,
    )

    # ── SHAP Demographic Disparity ────────────────────────────────────────────
    if report.group_disparity is not None:
        st.divider()
        st.markdown("### ⚖️ Demographic SHAP Disparity")
        st.markdown(
            "<p style='font-size:0.87rem;color:#6B7280;margin-top:-0.3rem;'>"
            "Mean SHAP contribution per feature, split by privileged vs. unprivileged group "
            "(first protected attribute). Large disparities indicate features that affect groups differently.</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            report.group_disparity.head(10).style.format({
                "Privileged Mean SHAP":   "{:+.3f}",
                "Unprivileged Mean SHAP": "{:+.3f}",
                "SHAP Impact Disparity":  "{:+.3f}",
            }),
            use_container_width=True,
        )

    st.divider()
    st.page_link("pages/6_Report.py", label="Proceed to Step 6 — Report Export →", icon="📄")
