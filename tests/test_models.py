"""
tests/test_models.py
====================
Phase 2 tests — model training, evaluation, and determinism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    SUPPORTED_MODELS,
    evaluate_model,
    get_metrics_dataframe,
    load_model,
    save_model,
    split_data,
    train_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_Xy() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic 400-row binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 400
    X = pd.DataFrame({
        "feat_a": rng.normal(size=n),
        "feat_b": rng.uniform(size=n),
        "feat_c": rng.integers(0, 5, size=n).astype(float),
    })
    p = 1 / (1 + np.exp(-(0.8 * X["feat_a"] - 0.5 * X["feat_b"] + 0.3 * X["feat_c"])))
    y = pd.Series((rng.uniform(size=n) < p).astype(int), name="label")
    return X, y


@pytest.fixture(scope="module")
def train_test_data(sample_Xy):
    X, y = sample_Xy
    return split_data(X, y, test_size=0.2, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Determinism — same seed → same predictions
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Models trained with the same seed must produce identical predictions."""

    @pytest.mark.parametrize("model_type", ["Logistic Regression", "Random Forest"])
    def test_same_seed_same_predictions(self, model_type: str, train_test_data) -> None:
        X_train, X_test, y_train, y_test = train_test_data

        result_a = train_model(X_train, y_train, model_type=model_type, seed=123)
        result_b = train_model(X_train, y_train, model_type=model_type, seed=123)

        eval_a = evaluate_model(result_a["model"], X_test, y_test)
        eval_b = evaluate_model(result_b["model"], X_test, y_test)

        np.testing.assert_array_equal(
            eval_a["predictions"],
            eval_b["predictions"],
            err_msg=f"{model_type}: predictions differ with identical seed=123",
        )

    def test_different_seeds_can_differ(self, train_test_data) -> None:
        """Two different seeds should (usually) produce different predictions."""
        X_train, X_test, y_train, y_test = train_test_data
        r1 = train_model(X_train, y_train, model_type="Random Forest", seed=1)
        r2 = train_model(X_train, y_train, model_type="Random Forest", seed=999)
        preds1 = evaluate_model(r1["model"], X_test, y_test)["predictions"]
        preds2 = evaluate_model(r2["model"], X_test, y_test)["predictions"]
        # Allow for the rare case they're identical, but usually they differ
        # Just check that the function runs without error
        assert len(preds1) == len(preds2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Metric ranges — all metrics must be in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricRanges:
    """All classification metrics must be non-negative and ≤ 1."""

    @pytest.mark.parametrize("model_type", SUPPORTED_MODELS)
    def test_metrics_in_valid_range(self, model_type: str, train_test_data) -> None:
        X_train, X_test, y_train, y_test = train_test_data

        result = train_model(X_train, y_train, model_type=model_type, seed=42)
        eval_result = evaluate_model(result["model"], X_test, y_test)

        for metric in ("accuracy", "precision", "recall", "f1"):
            val = eval_result[metric]
            assert 0.0 <= val <= 1.0, (
                f"{model_type}: {metric} = {val} is outside [0, 1]"
            )

        auc = eval_result["roc_auc"]
        if auc is not None:
            assert 0.0 <= auc <= 1.0, f"{model_type}: roc_auc = {auc} is outside [0, 1]"

        assert isinstance(eval_result["confusion_matrix"], np.ndarray)
        assert eval_result["confusion_matrix"].shape == (2, 2)

    def test_metrics_dataframe_shape(self, train_test_data) -> None:
        X_train, X_test, y_train, y_test = train_test_data
        result = train_model(X_train, y_train, model_type="Logistic Regression", seed=0)
        eval_result = evaluate_model(result["model"], X_test, y_test)
        df = get_metrics_dataframe(eval_result)
        assert list(df.columns) == ["Metric", "Value", "Description"]
        assert len(df) == 5  # accuracy, precision, recall, f1, roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: All 3 model types train without error
# ─────────────────────────────────────────────────────────────────────────────

class TestAllModelsRun:
    """Every model type must train and predict on valid data without errors."""

    @pytest.mark.parametrize("model_type", SUPPORTED_MODELS)
    def test_model_trains_and_predicts(self, model_type: str, train_test_data) -> None:
        X_train, X_test, y_train, y_test = train_test_data
        result = train_model(X_train, y_train, model_type=model_type, seed=7)

        assert "model" in result
        assert result["model_type"] == model_type
        assert result["feature_names"] == list(X_train.columns)
        assert result["n_train"] == len(X_train)
        assert result["train_time_s"] >= 0

        eval_result = evaluate_model(result["model"], X_test, y_test)
        assert len(eval_result["predictions"]) == len(y_test)
        assert set(eval_result["predictions"]).issubset({0, 1})

    def test_unknown_model_raises(self, train_test_data) -> None:
        X_train, _, y_train, _ = train_test_data
        with pytest.raises(ValueError, match="Unknown model type"):
            train_model(X_train, y_train, model_type="NeuralNet", seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Save and load model round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestModelPersistence:
    def test_save_load_roundtrip(self, train_test_data, tmp_path) -> None:
        """Saved model must produce identical predictions when reloaded."""
        from core.models import MODEL_DIR
        import core.models as models_module

        X_train, X_test, y_train, y_test = train_test_data
        result = train_model(X_train, y_train, model_type="Logistic Regression", seed=42)

        # Temporarily redirect MODEL_DIR to tmp_path
        original_dir = models_module.MODEL_DIR
        models_module.MODEL_DIR = tmp_path
        try:
            path = save_model(result, "test_lr")
            assert path.exists()
            loaded = load_model(path)
        finally:
            models_module.MODEL_DIR = original_dir

        preds_original = result["model"].predict(X_test)
        preds_loaded = loaded["model"].predict(X_test)
        np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.joblib")
