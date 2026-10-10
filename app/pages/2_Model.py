"""
app/pages/2_Model.py
====================
Model Training & Performance Evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_curve, confusion_matrix

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.theme import (
    render_metric_card, style_plotly_fig, page_header,
    info_banner, PRIMARY_COLOR,
)
from app.components.cache import make_hyperparams_key
from core.models import train_and_eval_model, check_model_quality

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Training — JobFair AI",
    page_icon="🤖",
    layout="wide",
)
render_sidebar()
page_header("", "Step 2 — Model Training",
            "Train a classification model on your dataset. Evaluation metrics feed directly into the bias audit step.")

bundle = st.session_state.get("data_bundle")

if bundle is None:
    st.warning("⚠️ No dataset loaded yet. Head back to **[Step 1 — Data Explorer](pages/1_Data.py)** first.")
    st.stop()

info_banner(f"Active dataset: <b>{bundle.dataset_name}</b> — "
            f"{len(bundle.X):,} rows, {len(bundle.feature_names)} features")

st.divider()

# ── Model configuration ───────────────────────────────────────────────────────
st.markdown("### ⚙️ Model Configuration")

col_m, col_p = st.columns([1, 2], gap="large")

with col_m:
    model_choice = st.selectbox(
        "Algorithm",
        options=["Logistic Regression", "Random Forest", "XGBoost"],
        help="LR is fastest. RF & XGBoost are tree-ensemble methods — usually higher accuracy.",
    )
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

with col_p:
    st.markdown("**Hyperparameters**")
    hyperparams: dict = {"random_state": seed}

    if model_choice == "Logistic Regression":
        c_val = st.slider("Regularization (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.05)
        solver = st.selectbox("Solver", ["lbfgs", "liblinear"])
        hyperparams.update({"C": c_val, "solver": solver, "max_iter": 1000})

    elif model_choice == "Random Forest":
        n_est = st.slider("Number of trees", 10, 300, 100, step=10)
        max_d = st.slider("Max depth", 1, 20, 10)
        hyperparams.update({"n_estimators": n_est, "max_depth": max_d})

    elif model_choice == "XGBoost":
        n_est = st.slider("Number of trees", 10, 300, 100, step=10)
        max_d = st.slider("Max depth", 1, 15, 6)
        lr    = st.slider("Learning rate", 0.01, 0.5, 0.1, step=0.01)
        hyperparams.update({"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr})

if st.button("🚀 Train Model", type="primary", key="train_btn"):
    with st.spinner(f"Training {model_choice}… this may take a moment"):
        try:
            model_bundle = train_and_eval_model(
                bundle=bundle,
                model_type=model_choice,
                hyperparams=hyperparams,
            )
            st.session_state["model_bundle"] = model_bundle
            # Clear downstream cached results when model changes
            st.session_state.pop("audit_summary", None)
            st.session_state.pop("mitigated_bundle", None)
            st.session_state.pop("shap_report", None)
            st.session_state.pop("pdf_bytes", None)
            st.success(f"✅ {model_choice} trained in {model_bundle.metrics.get('train_time_s', '—')}s")
        except Exception as exc:
            st.error(f"❌ Training failed: {exc}")

# ── Results ───────────────────────────────────────────────────────────────────
model_bundle = st.session_state.get("model_bundle")

if model_bundle is not None:
    st.divider()
    st.markdown(f"### 📈 Results — *{model_bundle.model_type}*")

    metrics = model_bundle.metrics

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_metric_card("Accuracy",   f"{metrics['accuracy']:.1%}",   "Correct predictions")
    with c2:
        render_metric_card("ROC-AUC",    f"{metrics['roc_auc']:.3f}",    "Area under ROC")
    with c3:
        render_metric_card("PR-AUC",     f"{metrics['pr_auc']:.3f}",     "Precision-recall AUC")
    with c4:
        render_metric_card("F1-Score",   f"{metrics['f1']:.3f}",         "Harmonic P/R mean")
    with c5:
        render_metric_card("Precision",  f"{metrics['precision']:.3f}",  "TP / predicted pos")
    with c6:
        render_metric_card("Recall",     f"{metrics['recall']:.3f}",     "TP / actual pos")

    st.markdown("<br>", unsafe_allow_html=True)

    quality_warnings = check_model_quality(metrics)
    if quality_warnings:
        for w in quality_warnings:
            st.warning(f"⚠️ {w}")
    else:
        st.success("✅ All performance metrics pass quality thresholds.")

    chart_c1, chart_c2 = st.columns(2, gap="medium")

    with chart_c1:
        fpr, tpr, _ = roc_curve(model_bundle.y_test, model_bundle.y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"ROC (AUC = {metrics['roc_auc']:.3f})",
            line=dict(color=PRIMARY_COLOR, width=2.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random",
            line=dict(color="#D1D5DB", dash="dash"),
        ))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_plotly_fig(fig_roc, title="ROC Curve"), use_container_width=True)

    with chart_c2:
        cm = confusion_matrix(model_bundle.y_test, model_bundle.y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Unfavorable (0)", "Favorable (1)"],
            y=["Unfavorable (0)", "Favorable (1)"],
            text_auto=True,
            color_continuous_scale=[[0, "#EEF2FF"], [1, PRIMARY_COLOR]],
        )
        st.plotly_chart(
            style_plotly_fig(fig_cm, title="Confusion Matrix (Test Set)"),
            use_container_width=True,
        )

    st.divider()
    st.page_link("pages/3_Audit.py", label="Proceed to Step 3 — Bias Audit →", icon="⚖️")
