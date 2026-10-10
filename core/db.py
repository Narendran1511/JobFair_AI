"""
core/db.py
==========
SQLAlchemy ORM models and CRUD helpers for JobFair AI audit history.

Schema
------
AuditRun  — one row per complete audit session
  └─ stores dataset name, model type, key metrics, timestamp, report path

All tables use SQLite (file at project root: audit_runs.db).
Connection is created per-call to avoid threading issues in Streamlit.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "audit_runs.db"
_ENGINE_URL = f"sqlite:///{_DB_PATH}"


# ─────────────────────────────────────────────────────────────────────────────
# ORM models
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class AuditRun(Base):
    """
    One row per completed audit run.

    Columns
    -------
    id              — auto-increment primary key
    timestamp       — UTC datetime of run creation
    dataset_name    — e.g. "UCI Adult Income"
    model_type      — e.g. "Random Forest"
    n_train         — training set size
    n_test          — test set size
    accuracy        — model accuracy on test set
    roc_auc         — ROC-AUC on test set
    f1              — F1 score
    protected_attrs — JSON list of protected attribute names
    spd             — Statistical Parity Difference (primary protected attr)
    disparate_impact — Disparate Impact ratio
    eod             — Equal Opportunity Difference
    aod             — Average Odds Difference
    theil_index     — Theil Index
    mitigation      — mitigation method applied (None if no mitigation)
    spd_after       — SPD after mitigation (None if not mitigated)
    accuracy_after  — accuracy after mitigation
    report_path     — path to exported HTML/PDF report (None if not exported)
    notes           — free-text notes field
    full_metrics_json — complete metrics dict as JSON for full re-display
    """

    __tablename__ = "audit_runs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    dataset_name: str = Column(String(128), nullable=False)
    model_type: str = Column(String(64), nullable=False)
    n_train: int = Column(Integer, nullable=True)
    n_test: int = Column(Integer, nullable=True)
    accuracy: float = Column(Float, nullable=True)
    roc_auc: float = Column(Float, nullable=True)
    f1: float = Column(Float, nullable=True)
    protected_attrs: str = Column(Text, nullable=True)   # JSON
    spd: float = Column(Float, nullable=True)
    disparate_impact: float = Column(Float, nullable=True)
    eod: float = Column(Float, nullable=True)
    aod: float = Column(Float, nullable=True)
    theil_index: float = Column(Float, nullable=True)
    mitigation: str = Column(String(64), nullable=True)
    spd_after: float = Column(Float, nullable=True)
    accuracy_after: float = Column(Float, nullable=True)
    report_path: str = Column(String(512), nullable=True)
    notes: str = Column(Text, nullable=True)
    full_metrics_json: str = Column(Text, nullable=True)  # JSON

    def to_dict(self) -> dict[str, Any]:
        """Convert ORM row to a plain dict (safe for JSON serialisation)."""
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("timestamp") and isinstance(d["timestamp"], datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        if d.get("protected_attrs"):
            try:
                d["protected_attrs"] = json.loads(d["protected_attrs"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("full_metrics_json"):
            try:
                d["full_metrics_json"] = json.loads(d["full_metrics_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Engine / session helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_db(db_path: str | Path | None = None):
    """Create (or reuse) SQLite engine."""
    url = f"sqlite:///{db_path}" if db_path else _ENGINE_URL
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def _get_engine():
    """Create (or reuse) default SQLite engine and ensure tables exist."""
    return init_db(_DB_PATH)


def save_audit_run(
    engine: Any,
    dataset_name: str,
    model_type: str,
    audit_summary: Any,
    report_path: str | Path | None = None,
) -> int:
    """Save audit run summary directly from an AuditReportSummary object."""
    first_attr = list(audit_summary.attribute_audits.keys())[0] if audit_summary.attribute_audits else "unknown"
    audit = audit_summary.attribute_audits.get(first_attr)

    run = AuditRun(
        timestamp=datetime.now(timezone.utc),
        dataset_name=dataset_name,
        model_type=model_type,
        disparate_impact=audit.disparate_impact if audit else 1.0,
        eod=audit.equal_opportunity_diff if audit else 0.0,
        aod=audit.average_odds_diff if audit else 0.0,
        theil_index=audit.theil_index if audit else 0.0,
        report_path=str(report_path) if report_path else None,
        notes=audit_summary.recommendation,
    )
    with Session(engine) as session:
        session.add(run)
        session.commit()
        return run.id


def delete_audit_run(engine: Any, run_id: int) -> bool:
    """Delete an audit run by ID on a specific engine instance."""
    with Session(engine) as session:
        row = session.get(AuditRun, run_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True



# ─────────────────────────────────────────────────────────────────────────────
# CRUD helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_run(
    dataset_name: str,
    model_type: str,
    eval_result: dict[str, Any],
    fairness_results: dict[str, Any] | None = None,
    protected_attrs: list[str] | None = None,
    mitigation: str | None = None,
    spd_after: float | None = None,
    accuracy_after: float | None = None,
    report_path: str | None = None,
    notes: str | None = None,
) -> int:
    """
    Persist an audit run to SQLite and return the new row ID.

    Parameters
    ----------
    dataset_name : str
    model_type : str
    eval_result : dict — output of core.models.evaluate_model()
    fairness_results : dict or None — output of core.fairness.compute_all_metrics()
    protected_attrs : list[str] or None
    mitigation : str or None — name of applied mitigation method
    spd_after : float or None
    accuracy_after : float or None
    report_path : str or None
    notes : str or None

    Returns
    -------
    int — new row ID
    """
    engine = _get_engine()

    # Extract fairness values safely
    fr = fairness_results or {}

    def _get_metric(key: str, subkey: str = "value") -> float | None:
        entry = fr.get(key)
        if isinstance(entry, dict):
            val = entry.get(subkey)
            return float(val) if val is not None else None
        return None

    run = AuditRun(
        timestamp=datetime.now(timezone.utc),
        dataset_name=dataset_name,
        model_type=model_type,
        n_train=eval_result.get("n_train"),
        n_test=eval_result.get("n_test"),
        accuracy=eval_result.get("accuracy"),
        roc_auc=eval_result.get("roc_auc"),
        f1=eval_result.get("f1"),
        protected_attrs=json.dumps(protected_attrs or []),
        spd=_get_metric("statistical_parity_difference"),
        disparate_impact=_get_metric("disparate_impact"),
        eod=_get_metric("equal_opportunity_difference"),
        aod=_get_metric("average_odds_difference"),
        theil_index=_get_metric("theil_index"),
        mitigation=mitigation,
        spd_after=spd_after,
        accuracy_after=accuracy_after,
        report_path=report_path,
        notes=notes,
        full_metrics_json=json.dumps(
            {
                "eval": {k: v for k, v in eval_result.items()
                         if k not in ("confusion_matrix", "predictions", "probabilities")},
                "fairness": fr,
            },
            default=str,
        ),
    )

    with Session(engine) as session:
        session.add(run)
        session.commit()
        new_id = run.id

    logger.info("Saved audit run ID=%d to %s", new_id, _DB_PATH)
    return new_id


def get_all_runs(engine: Any = None, limit: int = 200) -> list[dict[str, Any]]:
    """
    Retrieve all audit runs, most recent first.

    Parameters
    ----------
    engine : optional SQLAlchemy engine
    limit : int — maximum rows to return

    Returns
    -------
    list of dicts
    """
    if engine is None:
        engine = _get_engine()
    with Session(engine) as session:
        stmt = select(AuditRun).order_by(AuditRun.timestamp.desc(), AuditRun.id.desc()).limit(limit)

        rows = session.scalars(stmt).all()
        return [r.to_dict() for r in rows]



def get_run_by_id(run_id: int) -> dict[str, Any] | None:
    """
    Retrieve a single audit run by primary key.

    Parameters
    ----------
    run_id : int

    Returns
    -------
    dict or None if not found
    """
    engine = _get_engine()
    with Session(engine) as session:
        row = session.get(AuditRun, run_id)
        return row.to_dict() if row else None


def delete_run(run_id: int) -> bool:
    """
    Delete an audit run by ID.

    Returns
    -------
    True if deleted, False if not found.
    """
    engine = _get_engine()
    with Session(engine) as session:
        row = session.get(AuditRun, run_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        logger.info("Deleted audit run ID=%d", run_id)
        return True
