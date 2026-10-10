"""
tests/test_bias_metrics.py
===========================
Phase 3 tests — Bias auditing metrics and group fairness verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_adult, load_german
from core.models import train_and_eval_model
from core.bias_metrics import run_full_bias_audit, compute_intersectional_fairness


class TestBiasAuditEngine:
    @pytest.fixture(scope="class")
    def trained_adult_model(self):
        bundle = load_adult(sample_size=1000, seed=42)
        model_bundle = train_and_eval_model(bundle, model_type="Logistic Regression", hyperparams={"random_state": 42})
        return model_bundle

    def test_run_full_bias_audit_adult(self, trained_adult_model):
        audit_summary = run_full_bias_audit(trained_adult_model)
        assert audit_summary.overall_status in ("PASS", "WARNING", "FAIL")
        assert len(audit_summary.attribute_audits) > 0

        # Check adult protected attributes exist
        assert "sex" in audit_summary.attribute_audits or "race" in audit_summary.attribute_audits

        for attr_name, audit in audit_summary.attribute_audits.items():
            assert 0.0 <= audit.disparate_impact <= 5.0
            assert -1.0 <= audit.demographic_parity_diff <= 1.0
            assert -1.0 <= audit.equal_opportunity_diff <= 1.0
            assert -1.0 <= audit.average_odds_diff <= 1.0
            assert audit.theil_index >= 0.0
            assert audit.status in ("PASS", "WARNING", "FAIL")

    def test_intersectional_fairness(self, trained_adult_model):
        bundle = trained_adult_model.data_bundle
        y_pred = trained_adult_model.y_pred

        res = compute_intersectional_fairness(
            df_raw=bundle.df_raw.iloc[:len(y_pred)],
            y_pred=y_pred,
            protected_attrs=["sex", "race"],
        )
        assert isinstance(res, dict)
        if len(res) > 0:
            first_key = list(res.keys())[0]
            assert "selection_rate" in res[first_key]
            assert "disparate_impact_vs_overall" in res[first_key]
