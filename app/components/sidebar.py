"""
app/components/sidebar.py
=========================
Persistent sidebar navigation for JobFair AI.
Designed for precision, clarity, and scannability.
"""

from __future__ import annotations

import streamlit as st
from app.components.theme import inject_custom_css

STEPS = [
    (1, "Data Explorer",    "pages/1_Data.py",          "data_bundle"),
    (2, "Model Training",   "pages/2_Model.py",         "model_bundle"),
    (3, "Bias Audit",       "pages/3_Audit.py",         "audit_summary"),
    (4, "Bias Mitigation",  "pages/4_Mitigation.py",    "mitigated_bundle"),
    (5, "Explainability",   "pages/5_Explainability.py","shap_report"),
    (6, "Report Export",    "pages/6_Report.py",        "pdf_bytes"),
]


def render_sidebar() -> None:
    """Inject global design CSS and render the sidebar."""
    inject_custom_css()

    with st.sidebar:
        # ── Minimalist Product Branding ───────────────────────────────────────
        st.markdown(
            """
            <div style="padding: 0 0 0.75rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="width: 24px; height: 24px; background: #0F172A; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.75rem;">
                        JF
                    </div>
                    <div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; line-height: 1.1; letter-spacing: -0.01em;">JobFair AI</div>
                        <div style="font-size: 0.72rem; color: #64748B; font-weight: 400; margin-top: 1px;">Bias Auditing Workbench</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Primary Links ─────────────────────────────────────────────────────
        st.page_link("Home.py", label="Overview", icon=None)

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

        # ── Workflow Section Header ───────────────────────────────────────────
        completed_count = sum(
            1 for _, _, _, key in STEPS
            if st.session_state.get(key) is not None
        )

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.35rem;">
                <span style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em;">
                    Workflow Progression
                </span>
                <span style="font-size: 0.7rem; font-weight: 600; color: #475569; font-variant-numeric: tabular-nums;">
                    {completed_count}/6
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Minimalist Progress Bar
        pct = int((completed_count / 6) * 100)
        st.markdown(
            f"""
            <div style="height: 4px; background: #E2E8F0; border-radius: 999px; overflow: hidden; margin-bottom: 0.85rem;">
                <div style="height: 100%; width: {pct}%; background: #2563EB; border-radius: 999px; transition: width 0.25s ease;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Step Links ────────────────────────────────────────────────────────
        for num, name, path, key in STEPS:
            is_done = st.session_state.get(key) is not None
            status_prefix = "✓  " if is_done else f"{num}.  "
            label_text = f"{status_prefix}{name}"
            st.page_link(path, label=label_text, icon=None)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='border-top: 1px solid #E2E8F0; margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)

        # ── Secondary Pages ───────────────────────────────────────────────────
        st.page_link("pages/7_Ethics.py", label="Ethics & Limitations", icon=None)

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="margin-top: 2.2rem; padding-top: 0.75rem; border-top: 1px solid #E2E8F0;">
                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 400; line-height: 1.4;">
                    IBM SkillsBuild Capstone Project<br>
                    AI &amp; Machine Learning Domain
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
