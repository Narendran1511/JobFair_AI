"""
core/models.py
==============
Model training, evaluation, and serialisation for JobFair AI.

Supported models
----------------
* Logistic Regression  (sklearn)
* Random Forest        (sklearn)
* XGBoost              (xgboost, with sklearn fallback to ExtraTreesClassifier)

All training is deterministic (fixed seeds).
Model objects are saved to disk via joblib so they can be reloaded by the
explainability and mitigation modules without re-training.

Returns
-------
All public functions return plain Python dicts / standard objects so the
Streamlit UI layer stays framework-agnostic.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier  # type: ignore[import]
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("xgboost not available — will use ExtraTreesClassifier as fallback.")
    from sklearn.ensemble import ExtraTreesClassifier  # noqa: F401 (fallback)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

SUPPORTED_MODELS = ["Logistic Regression", "Random Forest", "XGBoost"]

from dataclasses import dataclass, field
from sklearn.metrics import precision_recall_curve, auc

# Thresholds for model quality warnings
MIN_ACCEPTABLE_AUC = 0.55
MIN_ACCEPTABLE_ACC = 0.50


@dataclass
class ModelBundle:
    """Container holding a trained model, test dataset split, predictions, and evaluation metrics."""
    data_bundle: Any
    model: Any
    model_type: str
    hyperparams: dict[str, Any]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_prob: np.ndarray
    metrics: dict[str, Any]
    feature_names: list[str]


def train_and_eval_model(
    bundle: Any,
    model_type: str = "Random Forest",
    hyperparams: dict[str, Any] | None = None,
) -> ModelBundle:
    """Train a model on bundle.X and evaluate on a 20% test split."""
    if hyperparams is None:
        hyperparams = {}

    seed = hyperparams.get("random_state", 42)
    X_train, X_test, y_train, y_test = split_data(bundle.X, bundle.y, test_size=0.2, seed=seed)

    model_res = train_model(X_train, y_train, model_type=model_type, seed=seed)
    model = model_res["model"]

    # Apply any specific hyperparams if set
    if "C" in hyperparams and hasattr(model, "C"):
        model.C = hyperparams["C"]
        model.fit(X_train, y_train)
    elif "n_estimators" in hyperparams and hasattr(model, "n_estimators"):
        model.n_estimators = hyperparams["n_estimators"]
        model.fit(X_train, y_train)

    eval_res = evaluate_model(model, X_test, y_test)
    y_prob = eval_res["probabilities"]
    y_pred = eval_res["predictions"]

    precision_p, recall_p, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = float(auc(recall_p, precision_p))

    metrics = {
        "accuracy": eval_res["accuracy"],
        "precision": eval_res["precision"],
        "recall": eval_res["recall"],
        "f1": eval_res["f1"],
        "roc_auc": eval_res["roc_auc"] if eval_res["roc_auc"] is not None else 0.5,
        "pr_auc": round(pr_auc, 4),
    }

    return ModelBundle(
        data_bundle=bundle,
        model=model,
        model_type=model_type,
        hyperparams=hyperparams,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=metrics,
        feature_names=bundle.feature_names,
    )


def check_model_quality(metrics: dict[str, Any]) -> list[str]:
    """Return quality warning messages if metrics fall below threshold."""
    warnings = []
    if metrics.get("roc_auc", 1.0) < MIN_ACCEPTABLE_AUC:
        warnings.append(f"ROC-AUC ({metrics.get('roc_auc'):.3f}) is below acceptable threshold ({MIN_ACCEPTABLE_AUC}).")
    if metrics.get("accuracy", 1.0) < MIN_ACCEPTABLE_ACC:
        warnings.append(f"Accuracy ({metrics.get('accuracy'):.1%}) is below acceptable threshold ({MIN_ACCEPTABLE_ACC:.0%}).")
    return warnings



# ─────────────────────────────────────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Parameters
    ----------
    X : pd.DataFrame — feature matrix
    y : pd.Series — binary labels
    test_size : float — fraction for test (default 0.2)
    seed : int — random seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(model_type: str, seed: int) -> Any:
    """Instantiate an unfitted model of the requested type."""
    if model_type == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
            class_weight="balanced",
        )
    elif model_type == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_type == "XGBoost":
        if XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=seed,
                verbosity=0,
            )
        else:
            logger.warning("XGBoost not available — using ExtraTreesClassifier instead.")
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            )
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: {SUPPORTED_MODELS}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "Random Forest",
    seed: int = 42,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Train a classification model.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series — binary labels (0/1)
    model_type : str — one of SUPPORTED_MODELS
    seed : int
    sample_weights : np.ndarray or None
        Per-sample weights (used by Reweighing mitigation).

    Returns
    -------
    dict with keys:
        model        — fitted sklearn-compatible model
        model_type   — str
        seed         — int
        feature_names — list[str]
        train_time_s  — float (seconds)
        n_train       — int
    """
    model = _build_model(model_type, seed)

    t0 = time.perf_counter()
    fit_kwargs: dict[str, Any] = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights
    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.perf_counter() - t0

    logger.info("Trained %s in %.2fs on %d samples.", model_type, elapsed, len(X_train))

    return {
        "model": model,
        "model_type": model_type,
        "seed": seed,
        "feature_names": list(X_train.columns),
        "train_time_s": round(elapsed, 3),
        "n_train": len(X_train),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute classification metrics for a fitted model.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    X_test : pd.DataFrame
    y_test : pd.Series
    threshold : float — decision threshold (default 0.5)

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc   — float
        confusion_matrix                            — np.ndarray (2×2)
        predictions                                 — np.ndarray (0/1)
        probabilities                               — np.ndarray (positive class prob)
        n_test                                      — int
        threshold                                   — float
        quality_warnings                            — list[str]
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_test, preds)

    warnings: list[str] = []
    if auc < MIN_ACCEPTABLE_AUC and not np.isnan(auc):
        warnings.append(
            f"ROC-AUC ({auc:.3f}) is close to random guessing (0.5). "
            "Consider using a different model or more features."
        )
    if acc < MIN_ACCEPTABLE_ACC:
        warnings.append(
            f"Accuracy ({acc:.1%}) is below 50%. The model may be misconfigured."
        )

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "confusion_matrix": cm,
        "predictions": preds,
        "probabilities": proba,
        "n_test": len(y_test),
        "threshold": threshold,
        "quality_warnings": warnings,
    }


def get_metrics_dataframe(eval_result: dict[str, Any]) -> pd.DataFrame:
    """
    Convert evaluate_model() output to a display-ready DataFrame.

    Parameters
    ----------
    eval_result : dict — output of evaluate_model()

    Returns
    -------
    pd.DataFrame with columns: Metric, Value, Description
    """
    metric_info = [
        ("Accuracy", eval_result["accuracy"], "Fraction of all predictions correct"),
        ("Precision", eval_result["precision"], "True positives / all positive predictions"),
        ("Recall", eval_result["recall"], "True positives / all actual positives"),
        ("F1 Score", eval_result["f1"], "Harmonic mean of precision and recall"),
        ("ROC-AUC", eval_result["roc_auc"], "Area under the receiver operating characteristic curve"),
    ]
    return pd.DataFrame(metric_info, columns=["Metric", "Value", "Description"])


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model_result: dict[str, Any], name: str) -> Path:
    """
    Save a trained model result dict to disk using joblib.

    Parameters
    ----------
    model_result : dict — output of train_model()
    name : str — filename stem (e.g. "adult_rf_run_001")

    Returns
    -------
    Path to the saved file.
    """
    # Don't pickle the full numpy arrays, save predictions separately
    save_dict = {k: v for k, v in model_result.items() if k != "model"}
    save_dict["model"] = model_result["model"]

    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(save_dict, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(path: str | Path) -> dict[str, Any]:
    """
    Load a model result dict from disk.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict — as returned by train_model()

    Raises
    ------
    FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
