"""
tests/test_mitigation.py
========================
Phase 4 tests — Bias Mitigation techniques (Reweighing & ThresholdOptimizer).
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
from core.mitigation import apply_reweighing, apply_threshold_optimizer


class TestMitigationEngine:
    @pytest.fixture(scope="class")
    def trained_adult_model(self):
        bundle = load_adult(sample_size=1000, seed=42)
        model_bundle = train_and_eval_model(bundle, model_type="Logistic Regression", hyperparams={"random_state": 42})
        return model_bundle

    def test_apply_reweighing(self, trained_adult_model):
        res = apply_reweighing(trained_adult_model, target_attr="sex")
        assert res.mitigation_method == "Reweighing (Pre-Processing)"
        assert "accuracy" in res.mitigated_metrics
        assert res.mitigated_audit.overall_status in ("PASS", "WARNING", "FAIL")
        assert "disparate_impact_change" in res.tradeoff_summary

    def test_apply_threshold_optimizer(self, trained_adult_model):
        res = apply_threshold_optimizer(trained_adult_model, target_attr="sex", constraint="equalized_odds")
        assert "Threshold Optimizer" in res.mitigation_method
        assert "accuracy" in res.mitigated_metrics
        assert res.mitigated_audit.overall_status in ("PASS", "WARNING", "FAIL")
