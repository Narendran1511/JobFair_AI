"""
app/pages/7_Ethics.py
======================
Ethics & Limitations — practical considerations when using this toolkit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.theme import render_card, page_header

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ethics & Limitations — JobFair AI",
    page_icon="⚠️",
    layout="wide",
)
render_sidebar()
page_header(
    "", "Ethics & Methodological Limitations",
    "Essential context, operational constraints, and governance considerations when auditing decision models."
)

st.divider()

render_card(
    "1. Human oversight is mandatory",
    "Fairness metrics are risk signals, not legal verdicts. Any automated hiring shortlisting "
    "or credit evaluation system must retain human review, especially for borderline cases.",
    accent=True,
)

render_card(
    "2. Accuracy vs. fairness trade-offs",
    "Debiasing algorithms like Reweighing or Threshold Optimization can reduce group disparities "
    "but may slightly reduce overall predictive accuracy. Both dimensions matter — avoid "
    "optimising for one at the complete expense of the other.",
)

render_card(
    "3. Historical data & measurement bias",
    "Reweighing adjusts sample weights; Threshold Optimizer adjusts decision cutoffs. Neither "
    "can fix unobserved confounders or flawed historical labels. If the underlying data "
    "encoded discrimination, debiasing techniques partially mitigate — not erase — that.",
)

render_card(
    "4. Benchmark thresholds are starting points",
    "The EEOC 80% (4/5ths) rule and the ±0.10 Demographic Parity thresholds used here are "
    "widely referenced but not universal legal standards. Context matters — "
    "consult legal counsel before relying on audit results for deployment decisions.",
)

render_card(
    "5. Dataset limitations",
    "UCI Adult Income and German Credit are decades-old research datasets. They are useful "
    "for demonstrating auditing methodology but don't reflect current labor or credit markets. "
    "Results on these datasets shouldn't be extrapolated to production use without domain validation.",
)
