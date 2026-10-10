"""
app/Home.py
===========
JobFair AI — Executive Overview & Audit Dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.theme import (
    render_card,
    render_metric_card,
    page_header,
    info_banner,
    PRIMARY_COLOR,
    TEXT_MAIN,
    TEXT_SECONDARY,
    TEXT_MUTED,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobFair AI — Bias Auditing & Explainability",
    page_icon="⚖️",
    layout="wide",
)
render_sidebar()

# ── Page Header ───────────────────────────────────────────────────────────────
page_header(
    icon="",
    title="JobFair AI Workbench",
    subtitle="An end-to-end responsible AI auditing toolkit to evaluate demographic fairness, execute debiasing algorithms, and inspect model decisions using SHAP explanations.",
)

# ── System Summary Metrics Row ────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="small")

with col_m1:
    data_b = st.session_state.get("data_bundle")
    ds_name = data_b.dataset_name if data_b else "Not Loaded"
    render_metric_card("1. Dataset", ds_name, "Source benchmark or CSV")

with col_m2:
    model_b = st.session_state.get("model_bundle")
    model_name = model_b.model_type if model_b else "Not Trained"
    sub_acc = f"Accuracy: {model_b.accuracy:.1%}" if model_b else "Select & train model"
    render_metric_card("2. Model", model_name, sub_acc)

with col_m3:
    audit_s = st.session_state.get("audit_summary")
    if audit_s:
        disp_imp = audit_s.metrics.get("disparate_impact", 0.0)
        passes = disp_imp >= 0.80
        audit_val = f"{disp_imp:.2f}"
        audit_sub = "Passes 80% Rule" if passes else "Disparity Detected"
    else:
        audit_val = "Pending"
        audit_sub = "Run fairness metrics"
    render_metric_card("3. Audit Status", audit_val, audit_sub)

with col_m4:
    mit_b = st.session_state.get("mitigated_bundle")
    mit_status = mit_b.algorithm_name if mit_b else "None Applied"
    mit_sub = "Post-mitigation active" if mit_b else "Optional debiasing step"
    render_metric_card("4. Mitigation", mit_status, mit_sub)

st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

# ── System Scope Banner ───────────────────────────────────────────────────────
info_banner(
    "<b>Auditing Scope:</b> This application evaluates binary classification models (such as hiring screening or credit scoring) "
    "for demographic disparities across protected attributes (e.g., Sex, Race, Age) using established regulatory benchmarks."
)

st.markdown("### Workflow Execution Steps")

steps = [
    ("1", "Data Explorer",   "pages/1_Data.py",          "Inspect group representation & baseline label rates", "data_bundle"),
    ("2", "Model Training",  "pages/2_Model.py",         "Train baseline classifier (Logistic, Random Forest, XGBoost)", "model_bundle"),
    ("3", "Bias Audit",      "pages/3_Audit.py",         "Compute group fairness metrics & 80% rule compliance", "audit_summary"),
    ("4", "Bias Mitigation",  "pages/4_Mitigation.py",    "Apply Reweighing or Threshold Optimization debiasing", "mitigated_bundle"),
    ("5", "Explainability",  "pages/5_Explainability.py","Inspect global & individual candidate SHAP explanations", "shap_report"),
    ("6", "Report Export",   "pages/6_Report.py",        "Generate and download structured PDF audit report", "pdf_bytes"),
]

step_cols = st.columns(3, gap="medium")
for i, (num, label, path, desc, key) in enumerate(steps):
    col = step_cols[i % 3]
    with col:
        done = st.session_state.get(key) is not None
        badge_html = '<span class="badge-pass">Complete</span>' if done else '<span class="badge-warning">Pending</span>'
        
        st.markdown(
            f"""
            <div class="jf-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">Step {num}</span>
                    {badge_html}
                </div>
                <div style="font-size: 0.98rem; font-weight: 600; color: #0F172A; margin-bottom: 0.25rem;">
                    {label}
                </div>
                <p style="font-size: 0.82rem; color: #475569; margin-bottom: 0.85rem; line-height: 1.4;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(path, label=f"Open Step {num} →")

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# ── Regulatory Standards Reference Section ─────────────────────────────────────
st.markdown("### Regulatory & Compliance Standards Evaluated")

rc1, rc2, rc3 = st.columns(3, gap="medium")
with rc1:
    render_card(
        "EEOC 80% (4/5ths) Rule",
        "Uniform Guidelines on Employee Selection Procedures. Evaluates whether unprivileged group selection rate is at least 80% of privileged group rate.",
        accent=True,
    )
with rc2:
    render_card(
        "EU AI Act Governance",
        "Technical documentation standards requiring continuous measurement of demographic impact and data quality verification for high-risk AI.",
    )
with rc3:
    render_card(
        "NIST AI RMF 1.0 Framework",
        "Risk management guidelines for measuring demographic trade-offs between predictive accuracy and demographic fairness metrics.",
    )

st.divider()

# ── Primary Call to Action ─────────────────────────────────────────────────────
c_left, c_mid, c_right = st.columns([1, 2, 1])
with c_mid:
    if st.button("Start Audit Workflow (Step 1 — Data Explorer)", type="primary", use_container_width=True):
        st.switch_page("pages/1_Data.py")
