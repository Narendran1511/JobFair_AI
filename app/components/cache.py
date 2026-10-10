"""
app/components/cache.py
=======================
Centralized Streamlit caching wrappers for all heavy operations.

Using @st.cache_data for data/results (serialisable) and
@st.cache_resource for non-serialisable objects (sklearn models, SHAP explainers).

All pages should import from here instead of calling core functions directly,
so caching is applied once in one place.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def cached_load_adult(sample_size: int | None, seed: int):
    """Load Adult dataset — cached to disk for 1h, avoids re-download on every rerun."""
    from core.data_loader import load_adult
    return load_adult(sample_size=sample_size, seed=seed)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_load_german(sample_size: int | None, seed: int):
    """Load German Credit dataset — cached to disk for 1h."""
    from core.data_loader import load_german
    return load_german(sample_size=sample_size, seed=seed)


# ── Model Training ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cached_train_model(
    dataset_name: str,
    sample_size: int | None,
    seed: int,
    model_type: str,
    hyperparams_key: str,  # JSON string of hyperparams for cache key
):
    """
    Train a model — cached per (dataset, sample_size, seed, model_type, hyperparams).
    Using cache_resource because sklearn models aren't trivially serialisable by Arrow.
    """
    import json
    from core.data_loader import load_adult, load_german
    from core.models import train_and_eval_model

    hyperparams = json.loads(hyperparams_key)

    if dataset_name == "UCI Adult Income":
        bundle = load_adult(sample_size=sample_size, seed=seed)
    elif dataset_name == "German Credit":
        bundle = load_german(sample_size=sample_size, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_and_eval_model(bundle=bundle, model_type=model_type, hyperparams=hyperparams)


# ── Bias Audit ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_bias_audit(
    dataset_name: str,
    model_type: str,
    sample_size: int | None,
    seed: int,
    hyperparams_key: str,
):
    """
    Run full bias audit — result is cached per model configuration.
    The cache key covers all parameters that affect model predictions.
    """
    from core.bias_metrics import run_full_bias_audit

    # Re-use the cached model_bundle from cache_resource
    model_bundle = cached_train_model(
        dataset_name=dataset_name,
        sample_size=sample_size,
        seed=seed,
        model_type=model_type,
        hyperparams_key=hyperparams_key,
    )
    return run_full_bias_audit(model_bundle)


# ── SHAP Explanations ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_shap(
    dataset_name: str,
    model_type: str,
    sample_size: int | None,
    seed: int,
    hyperparams_key: str,
    max_samples: int = 200,
):
    """
    Compute SHAP values — the slowest operation. Cached heavily.
    Re-runs only when the underlying model configuration changes.
    """
    from core.explainability import compute_shap_explanations

    model_bundle = cached_train_model(
        dataset_name=dataset_name,
        sample_size=sample_size,
        seed=seed,
        model_type=model_type,
        hyperparams_key=hyperparams_key,
    )
    return compute_shap_explanations(model_bundle, max_samples=max_samples)


# ── Mitigation ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_reweighing(
    dataset_name: str,
    model_type: str,
    sample_size: int | None,
    seed: int,
    hyperparams_key: str,
    target_attr: str,
):
    """Apply Reweighing debiasing — cached per model + target attribute."""
    from core.mitigation import apply_reweighing

    model_bundle = cached_train_model(
        dataset_name=dataset_name,
        sample_size=sample_size,
        seed=seed,
        model_type=model_type,
        hyperparams_key=hyperparams_key,
    )
    return apply_reweighing(model_bundle, target_attr=target_attr)


@st.cache_data(show_spinner=False)
def cached_threshold_optimizer(
    dataset_name: str,
    model_type: str,
    sample_size: int | None,
    seed: int,
    hyperparams_key: str,
    target_attr: str,
    constraint: str,
):
    """Apply Threshold Optimizer debiasing — cached per model + target + constraint."""
    from core.mitigation import apply_threshold_optimizer

    model_bundle = cached_train_model(
        dataset_name=dataset_name,
        sample_size=sample_size,
        seed=seed,
        model_type=model_type,
        hyperparams_key=hyperparams_key,
    )
    return apply_threshold_optimizer(model_bundle, target_attr=target_attr, constraint=constraint)


# ── Cache key helpers ─────────────────────────────────────────────────────────

def make_hyperparams_key(hyperparams: dict) -> str:
    """Convert a hyperparams dict to a stable JSON string for use as a cache key."""
    import json
    return json.dumps(hyperparams, sort_keys=True)


def get_model_cache_args(session: dict) -> dict | None:
    """
    Extract cache key arguments from session state.
    Returns None if session state doesn't have the necessary data yet.
    """
    bundle = session.get("data_bundle")
    model_bundle = session.get("model_bundle")

    if bundle is None or model_bundle is None:
        return None

    return {
        "dataset_name": bundle.dataset_name,
        "model_type": model_bundle.model_type,
        "sample_size": len(bundle.X) if bundle else None,
        "seed": model_bundle.hyperparams.get("random_state", 42),
        "hyperparams_key": make_hyperparams_key(model_bundle.hyperparams),
    }
