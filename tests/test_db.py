"""
tests/test_db.py
================
Phase 2 tests — SQLite persistence via core.db.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """
    Redirect the DB to a temp directory for each test so tests are isolated.
    Returns the patched db module.
    """
    import core.db as db_module

    tmp_db = tmp_path / "test_audit.db"
    monkeypatch.setattr(db_module, "_DB_PATH", tmp_db)
    monkeypatch.setattr(db_module, "_ENGINE_URL", f"sqlite:///{tmp_db}")
    yield db_module


@pytest.fixture()
def sample_eval_result():
    return {
        "accuracy": 0.82,
        "precision": 0.79,
        "recall": 0.75,
        "f1": 0.77,
        "roc_auc": 0.88,
        "n_train": 800,
        "n_test": 200,
        "threshold": 0.5,
        "quality_warnings": [],
    }


@pytest.fixture()
def sample_fairness_results():
    return {
        "statistical_parity_difference": {"value": -0.18, "status": "Fail"},
        "disparate_impact": {"value": 0.72, "status": "Fail"},
        "equal_opportunity_difference": {"value": -0.12, "status": "Warning"},
        "average_odds_difference": {"value": -0.10, "status": "Warning"},
        "theil_index": {"value": 0.08, "status": "Pass"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Save and retrieve round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveRetrieve:
    def test_save_and_get_by_id(
        self, isolated_db, sample_eval_result, sample_fairness_results
    ) -> None:
        """Saved run must be retrievable by ID with correct values."""
        run_id = isolated_db.save_run(
            dataset_name="UCI Adult Income",
            model_type="Random Forest",
            eval_result=sample_eval_result,
            fairness_results=sample_fairness_results,
            protected_attrs=["sex", "race"],
            mitigation="Reweighing",
            spd_after=-0.05,
            accuracy_after=0.80,
            report_path="/reports/run_001.html",
            notes="Phase 2 test run",
        )

        assert isinstance(run_id, int), f"Expected int ID, got {type(run_id)}"
        assert run_id >= 1

        row = isolated_db.get_run_by_id(run_id)
        assert row is not None, "Row not found after save"

        assert row["dataset_name"] == "UCI Adult Income"
        assert row["model_type"] == "Random Forest"
        assert abs(row["accuracy"] - 0.82) < 1e-6
        assert abs(row["roc_auc"] - 0.88) < 1e-6
        assert abs(row["spd"] - (-0.18)) < 1e-6
        assert abs(row["disparate_impact"] - 0.72) < 1e-6
        assert row["mitigation"] == "Reweighing"
        assert abs(row["spd_after"] - (-0.05)) < 1e-6
        assert row["notes"] == "Phase 2 test run"
        assert isinstance(row["protected_attrs"], list)
        assert "sex" in row["protected_attrs"]

    def test_get_nonexistent_returns_none(self, isolated_db) -> None:
        result = isolated_db.get_run_by_id(9999)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: get_all_runs returns a list
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllRuns:
    def test_get_all_returns_list(
        self, isolated_db, sample_eval_result
    ) -> None:
        """get_all_runs() must return a list, even if empty."""
        # Initially empty
        runs = isolated_db.get_all_runs()
        assert isinstance(runs, list)

        # Save two runs
        isolated_db.save_run(
            dataset_name="German Credit",
            model_type="Logistic Regression",
            eval_result=sample_eval_result,
        )
        isolated_db.save_run(
            dataset_name="UCI Adult Income",
            model_type="XGBoost",
            eval_result=sample_eval_result,
        )

        runs = isolated_db.get_all_runs()
        assert len(runs) == 2
        # Most recent first
        assert runs[0]["dataset_name"] == "UCI Adult Income"
        assert runs[1]["dataset_name"] == "German Credit"

    def test_full_metrics_json_parseable(
        self, isolated_db, sample_eval_result, sample_fairness_results
    ) -> None:
        """The stored full_metrics_json column must be valid JSON."""
        isolated_db.save_run(
            dataset_name="Test",
            model_type="Random Forest",
            eval_result=sample_eval_result,
            fairness_results=sample_fairness_results,
        )
        runs = isolated_db.get_all_runs()
        assert len(runs) >= 1
        full = runs[0].get("full_metrics_json")
        assert isinstance(full, dict), "full_metrics_json should be a parsed dict"
        assert "eval" in full
        assert "fairness" in full

    def test_delete_run(self, isolated_db, sample_eval_result) -> None:
        """Deleted run must no longer appear in get_all_runs()."""
        run_id = isolated_db.save_run(
            dataset_name="To Delete",
            model_type="Logistic Regression",
            eval_result=sample_eval_result,
        )
        assert isolated_db.get_run_by_id(run_id) is not None
        deleted = isolated_db.delete_run(run_id)
        assert deleted is True
        assert isolated_db.get_run_by_id(run_id) is None
        assert isolated_db.delete_run(9999) is False
