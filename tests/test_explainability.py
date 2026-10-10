"""
tests/test_explainability.py
=============================
Phase 4 tests — Model Explainability using SHAP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_adult
from core.models import train_and_eval_model
from core.explainability import compute_shap_explanations


class TestExplainabilityEngine:
    @pytest.fixture(scope="class")
    def trained_adult_model(self):
        bundle = load_adult(sample_size=500, seed=42)
        model_bundle = train_and_eval_model(bundle, model_type="Logistic Regression", hyperparams={"random_state": 42})
        return model_bundle

    def test_compute_shap_explanations(self, trained_adult_model):
        exp = compute_shap_explanations(trained_adult_model, max_samples=50)
        assert len(exp.shap_values) == 50
        assert not exp.global_importance.empty
        assert "Feature" in exp.global_importance.columns
        assert "Mean |SHAP Value|" in exp.global_importance.columns
