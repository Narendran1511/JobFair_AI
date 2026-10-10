"""
tests/test_report.py
====================
Phase 5 tests — Automated PDF Report Generation.
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
from core.bias_metrics import run_full_bias_audit
from core.report_gen import generate_pdf_report


class TestReportGenerator:
    @pytest.fixture(scope="class")
    def sample_audit(self):
        bundle = load_adult(sample_size=500, seed=42)
        model_bundle = train_and_eval_model(bundle, model_type="Logistic Regression", hyperparams={"random_state": 42})
        audit_summary = run_full_bias_audit(model_bundle)
        return model_bundle, audit_summary

    def test_generate_pdf_report(self, sample_audit, tmp_path):
        model_bundle, audit_summary = sample_audit
        pdf_path = tmp_path / "test_report.pdf"
        
        pdf_bytes = generate_pdf_report(
            model_bundle=model_bundle,
            audit_summary=audit_summary,
            output_path=pdf_path,
        )
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF")
        assert pdf_path.exists()
