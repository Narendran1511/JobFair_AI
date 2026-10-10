"""
app/pages/6_Report.py
======================
Export Audit Report — generates a downloadable PDF summarizing results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.theme import render_status_badge, page_header, info_banner
from core.report_gen import generate_pdf_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Report Export — JobFair AI",
    page_icon="📄",
    layout="wide",
)
render_sidebar()
page_header("", "Step 6 — Export Audit Report",
            "Compile model performance, demographic fairness metrics, and debiasing evaluations into a structured PDF document.")

model_bundle    = st.session_state.get("model_bundle")
audit_summary   = st.session_state.get("audit_summary")
mitigated_bundle = st.session_state.get("mitigated_bundle")

if model_bundle is None or audit_summary is None:
    st.warning("⚠️ Complete **[Step 2](pages/2_Model.py)** and **[Step 3](pages/3_Audit.py)** first.")
    st.stop()

info_banner(f"Model: <b>{model_bundle.model_type}</b> on <b>{model_bundle.data_bundle.dataset_name}</b>")

st.divider()

# ── Compliance table ──────────────────────────────────────────────────────────
st.markdown("### 🏛️ Regulatory Compliance Summary")

gov_df = pd.DataFrame([
    {
        "Framework": "EEOC Title VII (USA)",
        "Key Check": "Disparate Impact ≥ 0.80 (4/5ths rule)",
        "Status": "PASS" if audit_summary.overall_status == "PASS" else "NEEDS REVIEW",
    },
    {
        "Framework": "EU AI Act",
        "Key Check": "Bias monitoring & documentation for automated decisions",
        "Status": "DOCUMENTED",
    },
    {
        "Framework": "NIST AI RMF 1.0",
        "Key Check": "Empirical measurement & risk management controls",
        "Status": "VERIFIED",
    },
])

st.dataframe(gov_df, use_container_width=True)

# Quick summary metrics
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Audit Result Summary**")

summ_c1, summ_c2, summ_c3 = st.columns(3)
with summ_c1:
    st.markdown(
        f"Overall status: {render_status_badge(audit_summary.overall_status)}",
        unsafe_allow_html=True,
    )
with summ_c2:
    st.markdown(
        f"<span style='font-size:0.88rem;color:#4B5563;'>Total violations: "
        f"<b>{audit_summary.total_violations}</b></span>",
        unsafe_allow_html=True,
    )
with summ_c3:
    mit_status = "Included" if mitigated_bundle is not None else "Not run"
    st.markdown(
        f"<span style='font-size:0.88rem;color:#4B5563;'>Mitigation: <b>{mit_status}</b></span>",
        unsafe_allow_html=True,
    )

st.divider()

# ── PDF generation ────────────────────────────────────────────────────────────
st.markdown("### 📥 Generate & Download PDF")

col_btn, col_dl = st.columns([1, 1])

with col_btn:
    if st.button("📄 Generate PDF Report", type="primary", key="gen_pdf_btn"):
        with st.spinner("Building PDF with ReportLab…"):
            try:
                pdf_bytes = generate_pdf_report(
                    model_bundle=model_bundle,
                    audit_summary=audit_summary,
                    mitigation_bundle=mitigated_bundle,
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("✅ PDF generated — click Download to save it.")
            except Exception as exc:
                st.error(f"❌ PDF generation failed: {exc}")

pdf_bytes = st.session_state.get("pdf_bytes")

with col_dl:
    if pdf_bytes is not None:
        fname = f"JobFair_AI_Audit_{model_bundle.data_bundle.dataset_name.replace(' ', '_')}.pdf"
        st.download_button(
            label="⬇️ Download Audit PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            type="secondary",
        )
    else:
        st.markdown(
            "<p style='font-size:0.85rem;color:#9CA3AF;padding-top:0.5rem;'>"
            "Generate the report first, then download it here.</p>",
            unsafe_allow_html=True,
        )

st.divider()
st.page_link("pages/7_Ethics.py", label="Review Ethics & Limitations →", icon="⚠️")
