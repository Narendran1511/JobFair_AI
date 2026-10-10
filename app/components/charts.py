"""
app/components/charts.py
========================
Plotly chart builders styled for JobFair AI's design system.
Using Slate (#0F172A), Executive Blue (#2563EB), Sky Accent (#0EA5E9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.components.theme import (
    style_plotly_fig,
    PRIMARY_COLOR,
    ACCENT_COLOR,
    DANGER_COLOR,
    WARNING_COLOR,
    TEXT_MAIN,
    TEXT_MUTED,
)

PALETTE = [PRIMARY_COLOR, ACCENT_COLOR, "#64748B", "#475569", "#3B82F6", "#0284C7"]


def group_representation_chart(
    stats_df: pd.DataFrame,
    protected_col: str,
    title: str | None = None,
) -> go.Figure:
    """Horizontal bar chart of group counts."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=stats_df["group"].astype(str),
            x=stats_df["count"],
            orientation="h",
            name="Count",
            marker=dict(color=PRIMARY_COLOR, cornerradius=4),
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Number of Samples",
        yaxis_title=protected_col,
    )
    return style_plotly_fig(fig, title=title or f"Group Representation — {protected_col}")


def base_rate_chart(
    stats_df: pd.DataFrame,
    protected_col: str,
    title: str | None = None,
) -> go.Figure:
    """Bar chart comparing base (positive outcome) rates per group."""
    colors = [
        PRIMARY_COLOR if i == stats_df["base_rate"].idxmax() else "#94A3B8"
        for i in range(len(stats_df))
    ]
    fig = go.Figure(
        go.Bar(
            x=stats_df["group"].astype(str),
            y=stats_df["base_rate"],
            marker=dict(color=colors, cornerradius=4),
            hovertemplate="<b>%{x}</b><br>Positive Rate: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=protected_col,
        yaxis_title="Positive Rate",
        yaxis_tickformat=".0%",
    )
    return style_plotly_fig(fig, title=title or f"Positive Outcome Rate by {protected_col}")


def shap_importance_chart(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "SHAP Global Feature Importance",
) -> go.Figure:
    """Horizontal bar chart of mean |SHAP| values per feature."""
    df = importance_df.nlargest(top_n, "mean_abs_shap").sort_values("mean_abs_shap")
    fig = go.Figure(
        go.Bar(
            y=df["feature"],
            x=df["mean_abs_shap"],
            orientation="h",
            marker=dict(color=PRIMARY_COLOR, cornerradius=4),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Mean |SHAP| Value",
        yaxis_title="Feature",
    )
    return style_plotly_fig(fig, title=title)


def confusion_matrix_heatmap(
    cm: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    if labels is None:
        labels = ["Unfavorable", "Favorable"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale=[[0, "#F8FAFC"], [1, PRIMARY_COLOR]],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
        )
    )
    return style_plotly_fig(fig, title=title)
