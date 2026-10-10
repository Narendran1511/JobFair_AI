"""
tests/test_integration.py
==========================
Phase 6 tests — End-to-End Pipeline Integration Test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_adult, load_german
from core.models import train_and_eval_model
from core.bias_metrics import run_full_bias_audit
from core.mitigation import apply_reweighing, apply_threshold_optimizer
from core.explainability import compute_shap_explanations
from core.report_gen import generate_pdf_report
from core.db import init_db, save_audit_run, get_all_runs, delete_audit_run


class TestEndToEndPipeline:
    def test_full_pipeline_adult(self, tmp_path):
        # 1. Load Data
        bundle = load_adult(sample_size=300, seed=42)
        assert len(bundle.X) > 0

        # 2. Train Model
        model_bundle = train_and_eval_model(bundle, model_type="Logistic Regression", hyperparams={"random_state": 42})
        assert "accuracy" in model_bundle.metrics

        # 3. Run Bias Audit
        audit_summary = run_full_bias_audit(model_bundle)
        assert audit_summary.overall_status in ("PASS", "WARNING", "FAIL")

        # 4. Apply Bias Mitigation
        mit_bundle = apply_reweighing(model_bundle, target_attr="sex")
        assert mit_bundle.mitigated_audit.overall_status in ("PASS", "WARNING", "FAIL")

        # 5. Compute SHAP Explainability
        shap_report = compute_shap_explanations(model_bundle, max_samples=30)
        assert len(shap_report.shap_values) == 30

        # 6. Generate PDF Report
        pdf_path = tmp_path / "integration_report.pdf"
        pdf_bytes = generate_pdf_report(
            model_bundle=model_bundle,
            audit_summary=audit_summary,
            mitigation_bundle=mit_bundle,
            output_path=pdf_path,
        )
        assert len(pdf_bytes) > 2000


        # 7. Database Persistence
        db_path = tmp_path / "test_audit_history.db"
        engine = init_db(db_path)
        run_id = save_audit_run(
            engine=engine,
            dataset_name=bundle.dataset_name,
            model_type=model_bundle.model_type,
            audit_summary=audit_summary,
            report_path=pdf_path,
        )
        assert run_id > 0

        runs = get_all_runs(engine)
        assert len(runs) == 1
        assert runs[0]["dataset_name"] == bundle.dataset_name

        delete_audit_run(engine, run_id)
        assert len(get_all_runs(engine)) == 0
