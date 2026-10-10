"""
app/components/theme.py
========================
Design System & CSS Theme for JobFair AI.
Inspired by Linear, Vercel, and Stripe dashboards.
Focus: Calm, precise, credible, high-contrast accessibility.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

# ── Design Tokens ─────────────────────────────────────────────────────────────
# Colors: Slate & Precision Blue Palette
PRIMARY_COLOR   = "#2563EB"   # Executive Blue / Primary Accent
PRIMARY_HOVER   = "#1D4ED8"   # Darker Blue on Hover
TEXT_MAIN       = "#0F172A"   # Slate 900 (High contrast body & headers)
TEXT_SECONDARY  = "#475569"   # Slate 600 (Subtitles & descriptive text)
TEXT_MUTED      = "#94A3B8"   # Slate 400 (Captions, timestamps, inactive states)

# Semantics
SUCCESS_COLOR   = "#166534"   # Muted Forest Green
SUCCESS_BG      = "#F0FDF4"
SUCCESS_BORDER  = "#BBF7D0"

WARNING_COLOR   = "#92400E"   # Muted Warm Amber
WARNING_BG      = "#FFFBEB"
WARNING_BORDER  = "#FDE68A"

DANGER_COLOR    = "#991B1B"   # Restrained Deep Red
DANGER_BG       = "#FEF2F2"
DANGER_BORDER   = "#FECACA"

# Surfaces & Borders
BG_PAGE         = "#FAFAFA"   # Off-white / Cool Slate 50
BG_CARD         = "#FFFFFF"   # Crisp White Card
BORDER_CARD     = "#E2E8F0"   # Subtle Slate 200 Border
BORDER_HOVER    = "#CBD5E1"   # Slate 300 Border on Hover

# Backwards compatibility alias for charts module
ACCENT_COLOR    = "#0EA5E9"   # Sky Blue accent
WARNING_COLOR_HEX = "#D97706"

__all__ = [
    "PRIMARY_COLOR",
    "PRIMARY_HOVER",
    "TEXT_MAIN",
    "TEXT_SECONDARY",
    "TEXT_MUTED",
    "SUCCESS_COLOR",
    "WARNING_COLOR",
    "DANGER_COLOR",
    "ACCENT_COLOR",
    "BG_PAGE",
    "BG_CARD",
    "BORDER_CARD",
    "inject_custom_css",
    "style_plotly_fig",
    "render_card",
    "render_metric_card",
    "render_status_badge",
    "page_header",
    "info_banner",
]


def inject_custom_css() -> None:
    """Inject modern, minimalist design system CSS."""
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            /* ── Global Reset & Font ───────────────────────────────────────── */
            html, body, [class*="css"] {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
                color: {TEXT_MAIN};
                -webkit-font-smoothing: antialiased;
            }}
            
            /* Main Content Container */
            .block-container {{
                padding-top: 2rem !important;
                padding-bottom: 3.5rem !important;
                max-width: 1080px !important;
            }}

            /* ── Typography & Hierarchy ───────────────────────────────────── */
            h1, h2, h3, h4, h5 {{
                font-family: 'Inter', sans-serif !important;
                color: {TEXT_MAIN} !important;
                letter-spacing: -0.022em !important;
            }}
            h1 {{ font-size: 1.75rem !important; font-weight: 700 !important; margin-bottom: 0.25rem !important; line-height: 1.25 !important; }}
            h2 {{ font-size: 1.25rem !important; font-weight: 600 !important; margin-top: 1.5rem !important; margin-bottom: 0.6rem !important; }}
            h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 1.2rem !important; margin-bottom: 0.4rem !important; }}
            
            p, li, span, div {{
                font-size: 0.88rem !important;
                line-height: 1.55 !important;
                color: {TEXT_MAIN};
            }}

            /* ── Page Canvas ──────────────────────────────────────────────── */
            .stApp {{
                background-color: {BG_PAGE} !important;
            }}
            .main .block-container {{
                background-color: {BG_PAGE} !important;
            }}

            /* ── Hide Default Navigation & Collapse Extra Spacing ─────────── */
            [data-testid="stSidebarHeader"] {{
                padding-top: 0.5rem !important;
                padding-bottom: 0rem !important;
                height: auto !important;
            }}
            [data-testid="stSidebarNav"] {{
                display: none !important;
                margin: 0 !important;
                padding: 0 !important;
                height: 0 !important;
            }}
            [data-testid="stSidebarUserContent"] {{
                padding-top: 0 !important;
            }}

            /* ── Sidebar Component ────────────────────────────────────────── */
            section[data-testid="stSidebar"] {{
                background-color: #FFFFFF !important;
                border-right: 1px solid {BORDER_CARD} !important;
            }}
            section[data-testid="stSidebar"] .block-container {{
                background-color: #FFFFFF !important;
                padding-top: 0rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}

            /* Sidebar Links */
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
                border-radius: 6px !important;
                padding: 0.45rem 0.75rem !important;
                margin-bottom: 0.15rem !important;
                transition: all 0.12s ease-in-out !important;
                color: {TEXT_SECONDARY} !important;
                font-weight: 500 !important;
                font-size: 0.85rem !important;
                border: 1px solid transparent !important;
            }}
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
                background-color: #F8FAFC !important;
                color: {TEXT_MAIN} !important;
                border-color: {BORDER_CARD} !important;
            }}
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
                background-color: #F1F5F9 !important;
                color: {PRIMARY_COLOR} !important;
                font-weight: 600 !important;
                border-color: #E2E8F0 !important;
            }}

            /* ── Card Containers ──────────────────────────────────────────── */
            .jf-card {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_CARD};
                border-radius: 8px;
                padding: 1.25rem 1.4rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
                transition: border-color 0.15s ease, box-shadow 0.15s ease;
            }}
            .jf-card:hover {{
                border-color: {BORDER_HOVER};
                box-shadow: 0 2px 6px 0 rgba(0, 0, 0, 0.04);
            }}
            .jf-card-accent {{
                border-top: 2px solid {PRIMARY_COLOR};
            }}

            /* ── Metric Display Boxes ─────────────────────────────────────── */
            .jf-metric-box {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_CARD};
                border-radius: 8px;
                padding: 1rem 1.15rem;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
                height: 100%;
                transition: border-color 0.15s ease;
            }}
            .jf-metric-box:hover {{
                border-color: {BORDER_HOVER};
            }}
            .jf-metric-label {{
                font-size: 0.72rem;
                font-weight: 600;
                color: {TEXT_MUTED};
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.35rem;
            }}
            .jf-metric-value {{
                font-size: 1.6rem;
                font-weight: 700;
                color: {TEXT_MAIN};
                line-height: 1.1;
                font-variant-numeric: tabular-nums;
                letter-spacing: -0.02em;
            }}
            .jf-metric-sub {{
                font-size: 0.75rem;
                color: {TEXT_SECONDARY};
                margin-top: 0.35rem;
                line-height: 1.35;
            }}

            /* ── Status Pill Badges ───────────────────────────────────────── */
            .badge-pass {{
                background-color: {SUCCESS_BG};
                color: {SUCCESS_COLOR};
                border: 1px solid {SUCCESS_BORDER};
                font-weight: 600;
                padding: 2px 9px;
                border-radius: 9999px;
                display: inline-flex;
                align-items: center;
                gap: 4px;
                font-size: 0.76rem;
                letter-spacing: 0.01em;
            }}
            .badge-warning {{
                background-color: {WARNING_BG};
                color: {WARNING_COLOR};
                border: 1px solid {WARNING_BORDER};
                font-weight: 600;
                padding: 2px 9px;
                border-radius: 9999px;
                display: inline-flex;
                align-items: center;
                gap: 4px;
                font-size: 0.76rem;
                letter-spacing: 0.01em;
            }}
            .badge-fail {{
                background-color: {DANGER_BG};
                color: {DANGER_COLOR};
                border: 1px solid {DANGER_BORDER};
                font-weight: 600;
                padding: 2px 9px;
                border-radius: 9999px;
                display: inline-flex;
                align-items: center;
                gap: 4px;
                font-size: 0.76rem;
                letter-spacing: 0.01em;
            }}

            /* ── Buttons ──────────────────────────────────────────────────── */
            .stButton > button {{
                border-radius: 6px !important;
                font-weight: 500 !important;
                font-size: 0.86rem !important;
                padding: 0.45rem 1.25rem !important;
                transition: all 0.12s ease-in-out !important;
            }}
            .stButton > button[kind="primary"] {{
                background-color: {PRIMARY_COLOR} !important;
                color: #FFFFFF !important;
                border: 1px solid {PRIMARY_COLOR} !important;
                box-shadow: 0 1px 2px 0 rgba(37, 99, 235, 0.2) !important;
            }}
            .stButton > button[kind="primary"]:hover {{
                background-color: {PRIMARY_HOVER} !important;
                border-color: {PRIMARY_HOVER} !important;
                box-shadow: 0 2px 4px 0 rgba(37, 99, 235, 0.3) !important;
            }}
            .stButton > button[kind="secondary"] {{
                background-color: {BG_CARD} !important;
                color: {TEXT_MAIN} !important;
                border: 1px solid {BORDER_CARD} !important;
                box-shadow: 0 1px 2px 0 rgba(0,0,0,0.02) !important;
            }}
            .stButton > button[kind="secondary"]:hover {{
                background-color: #F8FAFC !important;
                border-color: {BORDER_HOVER} !important;
                color: {TEXT_MAIN} !important;
            }}

            /* ── Streamlit Native Tabs ────────────────────────────────────── */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 4px !important;
                border-bottom: 1px solid {BORDER_CARD} !important;
            }}
            .stTabs [data-baseweb="tab"] {{
                height: 36px !important;
                border-radius: 6px 6px 0 0 !important;
                font-size: 0.85rem !important;
                font-weight: 500 !important;
                color: {TEXT_SECONDARY} !important;
                padding: 0 14px !important;
                border: none !important;
            }}
            .stTabs [aria-selected="true"] {{
                background-color: transparent !important;
                color: {PRIMARY_COLOR} !important;
                font-weight: 600 !important;
                border-bottom: 2px solid {PRIMARY_COLOR} !important;
            }}

            /* ── Info & Notice Banners ────────────────────────────────────── */
            .jf-info-banner {{
                background-color: #F8FAFC;
                border: 1px solid {BORDER_CARD};
                border-left: 3px solid {PRIMARY_COLOR};
                border-radius: 6px;
                padding: 0.8rem 1.1rem;
                margin-bottom: 1rem;
                font-size: 0.86rem;
                color: {TEXT_SECONDARY};
                line-height: 1.5;
            }}

            /* ── Page Subtitles ────────────────────────────────────────────── */
            .jf-page-sub {{
                font-size: 0.9rem;
                color: {TEXT_SECONDARY};
                margin-top: -0.25rem;
                margin-bottom: 1.25rem;
                line-height: 1.5;
            }}

            /* ── DataFrames & Tables ──────────────────────────────────────── */
            .stDataFrame {{
                border-radius: 6px !important;
                border: 1px solid {BORDER_CARD} !important;
            }}

            /* Divider */
            hr {{
                border: none !important;
                border-top: 1px solid {BORDER_CARD} !important;
                margin: 1.5rem 0 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_fig(fig: go.Figure, title: str | None = None, height: int = 330) -> go.Figure:
    """Apply minimalist, high-contrast Plotly chart styling matching the Linear/Vercel palette."""
    fig.update_layout(
        title=dict(
            text=title or "",
            font=dict(size=13, color=TEXT_MAIN, family="Inter, sans-serif"),
            x=0.01,
            pad=dict(b=6),
        ) if title else None,
        font=dict(family="Inter, sans-serif", size=11.5, color=TEXT_MUTED),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=36, r=16, t=40 if title else 14, b=32),
        height=height,
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=BORDER_CARD,
            tickfont=dict(size=11, color=TEXT_SECONDARY),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=BORDER_CARD,
            tickfont=dict(size=11, color=TEXT_SECONDARY),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=BORDER_CARD,
            font_size=11.5,
            font_family="Inter, sans-serif",
            font_color=TEXT_MAIN,
        ),
    )
    return fig


def render_card(title: str, body: str, accent: bool = False) -> None:
    """Render a minimalist card container."""
    accent_class = " jf-card-accent" if accent else ""
    st.markdown(
        f"""
        <div class="jf-card{accent_class}">
            <div style="font-weight:600; font-size:0.92rem; color:{TEXT_MAIN}; margin-bottom:0.3rem;">{title}</div>
            <p style="margin:0; font-size:0.85rem; color:{TEXT_SECONDARY}; line-height:1.5;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str = "") -> None:
    """Render a clean metric display tile."""
    sub_html = f'<div class="jf-metric-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="jf-metric-box">
            <div class="jf-metric-label">{label}</div>
            <div class="jf-metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str) -> str:
    """Return inline HTML for a subtle pill status badge."""
    s = status.upper()
    if s == "PASS":
        return '<span class="badge-pass">PASS</span>'
    elif s == "WARNING":
        return '<span class="badge-warning">WARNING</span>'
    else:
        return '<span class="badge-fail">FAIL</span>'


def page_header(icon: str, title: str, subtitle: str) -> None:
    """Render a clean, professional page header."""
    st.markdown(f"# {title}")
    st.markdown(f'<p class="jf-page-sub">{subtitle}</p>', unsafe_allow_html=True)


def info_banner(text: str) -> None:
    """Render a slate info banner."""
    st.markdown(
        f'<div class="jf-info-banner">{text}</div>',
        unsafe_allow_html=True,
    )
