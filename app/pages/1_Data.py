"""
app/pages/1_Data.py
===================
Data Explorer — load a dataset, inspect demographic group statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.charts import group_representation_chart, base_rate_chart
from app.components.theme import render_metric_card, page_header, info_banner
from app.components.cache import cached_load_adult, cached_load_german

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Explorer — JobFair AI",
    page_icon="📊",
    layout="wide",
)
render_sidebar()
page_header("", "Step 1 — Data Explorer",
            "Load a benchmark dataset or upload a custom CSV. Inspect demographic representation "
            "and baseline label rates before model training.")

# ── Dataset selector ──────────────────────────────────────────────────────────
dataset_choice = st.radio(
    "Dataset source",
    options=["UCI Adult Income", "German Credit", "Upload Custom CSV"],
    horizontal=True,
    key="dataset_choice",
)

bundle = None
load_error: str | None = None

# ── Bundled datasets ──────────────────────────────────────────────────────────
if dataset_choice in ("UCI Adult Income", "German Credit"):
    from core.data_loader import DATASET_REGISTRY
    info = DATASET_REGISTRY[dataset_choice]

    st.markdown(
        f"""
        <div class="jf-card jf-card-accent" style="margin-top:0.75rem;">
            <div style="font-weight:600;color:#111827;margin-bottom:0.25rem;">{dataset_choice}</div>
            <div style="font-size:0.87rem;color:#4B5563;">{info['description']}</div>
            <div style="font-size:0.8rem;color:#9CA3AF;margin-top:0.4rem;">⚠️ {info['known_issues']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_s, col_seed = st.columns([2, 1])
    with col_s:
        sample_size = st.select_slider(
            "Sample size (rows)",
            options=[500, 1000, 5000, 10000, "All"],
            value=5000,
            help="Smaller samples load faster; larger samples give more stable fairness estimates.",
        )
    with col_seed:
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

    n = None if sample_size == "All" else int(sample_size)

    if st.button("🔄 Load Dataset", type="primary", key="load_bundled"):
        with st.spinner(f"Loading {dataset_choice}…"):
            try:
                if dataset_choice == "UCI Adult Income":
                    bundle = cached_load_adult(sample_size=n, seed=seed)
                else:
                    bundle = cached_load_german(sample_size=n, seed=seed)
                st.session_state["data_bundle"] = bundle
                st.session_state.pop("model_bundle", None)   # stale model if dataset changed
                st.session_state.pop("audit_summary", None)
                st.success(f"✅ Loaded {len(bundle.X):,} rows × {len(bundle.feature_names)} features.")
            except Exception as exc:
                load_error = str(exc)

# ── Custom CSV ─────────────────────────────────────────────────────────────────
else:
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Must have column headers and a binary target column.",
    )

    if uploaded is not None:
        try:
            preview_df = pd.read_csv(uploaded)
            with st.expander("Preview uploaded file (first 5 rows)", expanded=True):
                st.dataframe(preview_df.head(), use_container_width=True)

            all_cols = preview_df.columns.tolist()

            c_t, c_f = st.columns(2)
            with c_t:
                target_col = st.selectbox("Target column (label)", options=all_cols, key="custom_target")
            with c_f:
                favorable_label = st.selectbox(
                    "Favorable label value (positive outcome)",
                    options=sorted(preview_df[target_col].dropna().unique().tolist()),
                    key="custom_fav_label",
                )

            protected_attrs = st.multiselect(
                "Protected attribute columns (e.g. sex, race, age)",
                options=[c for c in all_cols if c != target_col],
                key="custom_protected",
            )

            priv_vals: dict[str, str] = {}
            unpriv_vals: dict[str, str] = {}
            if protected_attrs:
                st.markdown("**Define privileged / unprivileged groups:**")
                for attr in protected_attrs:
                    avail = sorted(preview_df[attr].dropna().unique().tolist())
                    c1, c2 = st.columns(2)
                    with c1:
                        priv_vals[attr] = st.selectbox(f"Privileged — {attr}", avail, key=f"priv_{attr}")
                    with c2:
                        unpriv_vals[attr] = st.selectbox(f"Unprivileged — {attr}", avail, key=f"unpriv_{attr}")

            seed_custom = st.number_input("Random seed", value=42, key="custom_seed")

            if st.button("🔄 Load Custom Dataset", type="primary", key="load_custom_btn"):
                if not protected_attrs:
                    st.error("Please select at least one protected attribute.")
                else:
                    with st.spinner("Processing custom dataset…"):
                        try:
                            import tempfile
                            from core.data_loader import load_custom
                            uploaded.seek(0)
                            custom_dir = ROOT / "data" / "custom"
                            custom_dir.mkdir(parents=True, exist_ok=True)
                            with tempfile.NamedTemporaryFile(
                                suffix=".csv", delete=False, dir=str(custom_dir)
                            ) as tmp:
                                tmp.write(uploaded.read())
                                tmp_path = tmp.name
                            bundle = load_custom(
                                path=tmp_path,
                                target_col=target_col,
                                protected_attrs=protected_attrs,
                                favorable_label=favorable_label,
                                privileged_vals=priv_vals,
                                unprivileged_vals=unpriv_vals,
                                seed=seed_custom,
                            )
                            st.session_state["data_bundle"] = bundle
                            st.session_state.pop("model_bundle", None)
                            st.session_state.pop("audit_summary", None)
                            st.success(f"✅ Loaded {len(bundle.X):,} rows × {len(bundle.feature_names)} features.")
                        except Exception as exc:
                            load_error = str(exc)
        except Exception as exc:
            load_error = f"Could not read CSV: {exc}"

if load_error:
    st.error(f"❌ Load failed: {load_error}")

# ── Dataset Overview ──────────────────────────────────────────────────────────
bundle = st.session_state.get("data_bundle")

if bundle is not None:
    st.divider()
    st.markdown(f"### Dataset Overview — *{bundle.dataset_name}*")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        render_metric_card("Total Samples", f"{len(bundle.X):,}", "Rows in selected sample")
    with mc2:
        render_metric_card("Features", str(len(bundle.feature_names)), "Predictor columns")
    with mc3:
        render_metric_card("Positive Rate", f"{bundle.y.mean():.1%}", f"Label: {bundle.label_col}")
    with mc4:
        render_metric_card("Protected Attrs", str(len(bundle.protected_cols)), ", ".join(bundle.protected_cols))

    st.markdown("<br>", unsafe_allow_html=True)

    c_raw, c_miss = st.columns(2)
    with c_raw:
        with st.expander("🔍 Raw data preview (first 100 rows)"):
            st.dataframe(bundle.df_raw.head(100), use_container_width=True)
    with c_miss:
        with st.expander("🕳️ Missing value summary"):
            missing = bundle.df_raw.isnull().sum()
            missing = missing[missing > 0]
            if missing.empty:
                st.success("No missing values after preprocessing.")
            else:
                st.dataframe(
                    pd.DataFrame({
                        "Column": missing.index,
                        "Missing Count": missing.values,
                        "Missing %": (missing.values / len(bundle.df_raw) * 100).round(2),
                    }),
                    use_container_width=True,
                )

    st.divider()

    # ── Group Breakdown ────────────────────────────────────────────────────────
    st.markdown("### 👥 Demographic Group Breakdown")

    if bundle.group_stats:
        available_attrs = list(bundle.group_stats.keys())
        selected_attr = st.selectbox(
            "Attribute to inspect",
            options=available_attrs,
            key="group_attr_select",
        )

        stats_df = bundle.group_stats[selected_attr]

        if not stats_df.empty:
            st.dataframe(
                stats_df.style.format({
                    "fraction": "{:.1%}",
                    "base_rate": "{:.1%}",
                    "base_rate_diff_from_overall": "{:+.1%}",
                }),
                use_container_width=True,
            )

            chart_col1, chart_col2 = st.columns(2, gap="medium")
            with chart_col1:
                st.plotly_chart(
                    group_representation_chart(stats_df, selected_attr),
                    use_container_width=True,
                )
            with chart_col2:
                if stats_df["base_rate"].notna().any():
                    st.plotly_chart(
                        base_rate_chart(stats_df, selected_attr),
                        use_container_width=True,
                    )
                else:
                    st.info("Base rate chart not available for this attribute.")

            if stats_df["base_rate_diff_from_overall"].notna().any():
                max_diff = stats_df["base_rate_diff_from_overall"].abs().max()
                if max_diff > 0.1:
                    st.warning(
                        f"⚠️ Base-rate gap for `{selected_attr}` is **{max_diff:.1%}** "
                        "(exceeds the 10% benchmark) — signals baseline group imbalance."
                    )
                else:
                    st.success(f"✅ Base-rate differences for `{selected_attr}` are within ±10%.")
    else:
        st.info("No group statistics available for this dataset.")

    st.divider()
    st.page_link("pages/2_Model.py", label="Proceed to Step 2 — Model Training →", icon="🤖")
