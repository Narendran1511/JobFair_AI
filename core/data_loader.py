"""
core/data_loader.py
===================
Dataset loaders for JobFair AI.

Supported datasets
------------------
* UCI Adult Income  (protected: sex, race, age_group)
* German Credit     (protected: sex, age_group)
* Custom CSV        (user-supplied column configuration)

Each loader returns a :class:`DataBundle` — a dataclass carrying the cleaned
DataFrame, the feature/label split, and an AIF360-compatible wrapper (or a
lightweight substitute when AIF360 is not installed).

Design notes
------------
- AIF360 is imported lazily; if unavailable, a ``_FallbackDataset`` shim
  provides the same ``.features``, ``.labels``, ``.instance_weights`` API
  so downstream code never needs ``if aif360_available`` branches.
- All encoding is deterministic (ordered category maps, no random state).
- Missing values are imputed with mode (categorical) or median (numeric).
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── AIF360 availability check ────────────────────────────────────────────────
try:
    from aif360.datasets import BinaryLabelDataset  # type: ignore[import]
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    logger.warning("aif360 not available — using lightweight fallback dataset wrapper.")

# ── Paths ────────────────────────────────────────────────────────────────────
_PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = _PACKAGE_ROOT / "data"

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
GERMAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback dataset shim (used when AIF360 is not installed)
# ─────────────────────────────────────────────────────────────────────────────

class _FallbackDataset:
    """
    Lightweight substitute for ``aif360.datasets.BinaryLabelDataset``.

    Provides the same attributes expected by :mod:`core.fairness` so that
    the rest of the codebase works whether or not AIF360 is installed.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_name: str,
        protected_attribute_names: list[str],
        favorable_label: float = 1.0,
        unfavorable_label: float = 0.0,
        privileged_protected_attributes: list[list[float]] | None = None,
        unprivileged_protected_attributes: list[list[float]] | None = None,
        instance_weights: np.ndarray | None = None,
    ) -> None:
        self.label_name = label_name
        self.protected_attribute_names = protected_attribute_names
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.privileged_protected_attributes = privileged_protected_attributes or []
        self.unprivileged_protected_attributes = unprivileged_protected_attributes or []

        self.df = df.copy()
        feature_cols = [c for c in df.columns if c != label_name]
        self.feature_names = feature_cols

        self.features: np.ndarray = df[feature_cols].values.astype(float)
        self.labels: np.ndarray = df[[label_name]].values.astype(float)
        self.instance_weights: np.ndarray = (
            instance_weights if instance_weights is not None else np.ones(len(df))
        )
        self.protected_attributes: np.ndarray = df[protected_attribute_names].values.astype(float)

    def copy(self) -> "_FallbackDataset":
        """Return a deep copy."""
        obj = _FallbackDataset.__new__(_FallbackDataset)
        obj.__dict__.update(self.__dict__)
        obj.df = self.df.copy()
        obj.features = self.features.copy()
        obj.labels = self.labels.copy()
        obj.instance_weights = self.instance_weights.copy()
        obj.protected_attributes = self.protected_attributes.copy()
        return obj

    def convert_to_dataframe(self) -> tuple[pd.DataFrame, dict]:
        """Mirrors AIF360's ``convert_to_dataframe`` API."""
        df = self.df.copy()
        df[self.label_name] = self.labels.ravel()
        meta: dict[str, Any] = {
            "label_maps": [{1.0: ">50K", 0.0: "<=50K"}],
            "protected_attribute_maps": [],
        }
        return df, meta


def _make_dataset(
    df: pd.DataFrame,
    label_name: str,
    protected_attribute_names: list[str],
    favorable_label: float,
    unfavorable_label: float,
    privileged_protected_attributes: list[list[float]],
    unprivileged_protected_attributes: list[list[float]],
) -> Any:
    """
    Construct an AIF360 ``BinaryLabelDataset`` or its fallback shim.

    All callers use this factory so the rest of the codebase is toolkit-agnostic.
    """
    if AIF360_AVAILABLE:
        return BinaryLabelDataset(
            df=df,
            label_names=[label_name],
            protected_attribute_names=protected_attribute_names,
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label,
            privileged_protected_attributes=privileged_protected_attributes,
            unprivileged_protected_attributes=unprivileged_protected_attributes,
        )
    return _FallbackDataset(
        df=df,
        label_name=label_name,
        protected_attribute_names=protected_attribute_names,
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        privileged_protected_attributes=privileged_protected_attributes,
        unprivileged_protected_attributes=unprivileged_protected_attributes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataBundle — the unified return type for all loaders
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataBundle:
    """
    Unified container returned by every dataset loader.

    Attributes
    ----------
    df_raw : pd.DataFrame
        Original data (post-download, pre-encoding) for display in the UI.
    df_encoded : pd.DataFrame
        Numerically encoded version used for model training.
    X : pd.DataFrame
        Feature matrix (encoded, label column removed).
    y : pd.Series
        Binary label (0/1).
    label_col : str
        Name of the target column.
    protected_cols : list[str]
        Names of protected attribute columns present in ``X``.
    privileged_groups : list[dict[str, float]]
        AIF360-style privileged group specification.
    unprivileged_groups : list[dict[str, float]]
        AIF360-style unprivileged group specification.
    favorable_label : float
        The label value considered a positive outcome (e.g. 1.0 = income >50K).
    dataset_obj : BinaryLabelDataset | _FallbackDataset
        AIF360-compatible dataset object used by :mod:`core.fairness`.
    feature_names : list[str]
        Ordered list of feature column names.
    categorical_cols : list[str]
        Subset of feature columns that were originally categorical.
    numeric_cols : list[str]
        Subset of feature columns that were originally numeric.
    dataset_name : str
        Human-readable dataset identifier.
    group_stats : dict[str, pd.DataFrame]
        Pre-computed group statistics per protected attribute (see
        :func:`get_group_stats`).
    """

    df_raw: pd.DataFrame
    df_encoded: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    label_col: str
    protected_cols: list[str]
    privileged_groups: list[dict[str, float]]
    unprivileged_groups: list[dict[str, float]]
    favorable_label: float
    dataset_obj: Any
    feature_names: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    dataset_name: str
    group_stats: dict[str, pd.DataFrame] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _impute(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    """Impute missing values: mode for categorical, median for numeric."""
    df = df.copy()
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def _encode_categoricals(
    df: pd.DataFrame, cat_cols: list[str], encodings: dict[str, dict[str, int]] | None = None
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """
    Ordinal-encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list of column names to encode.
    encodings : existing encoding maps; if None, derive from df.

    Returns
    -------
    (encoded_df, encoding_maps)
    """
    df = df.copy()
    if encodings is None:
        encodings = {}
        for col in cat_cols:
            categories = sorted(df[col].dropna().unique().tolist())
            encodings[col] = {v: i for i, v in enumerate(categories)}
    for col in cat_cols:
        df[col] = df[col].map(encodings.get(col, {})).fillna(-1).astype(int)
    return df, encodings


def get_group_stats(df: pd.DataFrame, protected_col: str, label_col: str) -> pd.DataFrame:
    """
    Compute group representation and base-rate statistics for a protected attribute.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (human-readable) dataframe.
    protected_col : str
        Name of the protected attribute column.
    label_col : str
        Name of the binary target column (must be 0/1 encoded in df, or
        the function will attempt to map the first unique value to 0).

    Returns
    -------
    pd.DataFrame with columns:
        group, count, fraction, base_rate, base_rate_diff_from_overall
    """
    stats_rows = []
    overall_rate = df[label_col].mean() if pd.api.types.is_numeric_dtype(df[label_col]) else np.nan

    for group_val, group_df in df.groupby(protected_col):
        count = len(group_df)
        fraction = count / len(df)
        if pd.api.types.is_numeric_dtype(group_df[label_col]):
            base_rate = group_df[label_col].mean()
        else:
            base_rate = np.nan
        stats_rows.append(
            {
                "group": str(group_val),
                "count": count,
                "fraction": round(fraction, 4),
                "base_rate": round(base_rate, 4) if not np.isnan(base_rate) else None,
                "base_rate_diff_from_overall": (
                    round(base_rate - overall_rate, 4)
                    if not np.isnan(base_rate) and not np.isnan(overall_rate)
                    else None
                ),
            }
        )
    return pd.DataFrame(stats_rows).sort_values("count", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# UCI Adult Income
# ─────────────────────────────────────────────────────────────────────────────

_ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

_ADULT_CAT_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

_ADULT_NUM_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week",
]

_ADULT_LABEL_COL = "income"
_ADULT_FAVORABLE_LABEL = 1.0  # income >50K


def _download_adult() -> pd.DataFrame:
    """Download and concatenate Adult train + test splits from UCI."""
    import requests  # lazy import — not needed at module load time

    def _fetch(url: str, skip_rows: int = 0) -> pd.DataFrame:
        logger.info("Downloading: %s", url)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(
            io.StringIO(resp.text),
            header=None,
            names=_ADULT_COLUMNS,
            skiprows=skip_rows,
            na_values=[" ?", "?", ""],
            skipinitialspace=True,
        )

    df_train = _fetch(ADULT_URL)
    df_test = _fetch(ADULT_TEST_URL, skip_rows=1)
    df_test["income"] = df_test["income"].str.rstrip(".")
    return pd.concat([df_train, df_test], ignore_index=True)


def _load_adult_raw() -> pd.DataFrame:
    """Load raw Adult data from disk cache or download."""
    cache_path = DATA_DIR / "adult" / "adult.csv"
    if cache_path.exists():
        logger.info("Loading Adult dataset from cache: %s", cache_path)
        return pd.read_csv(cache_path, na_values=[" ?", "?", ""])

    logger.info("Adult dataset not cached — downloading from UCI...")
    df = _download_adult()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Cached Adult dataset to %s", cache_path)
    return df


def load_adult(sample_size: int | None = None, seed: int = 42) -> DataBundle:
    """
    Load, clean, encode, and wrap the UCI Adult Income dataset.

    Parameters
    ----------
    sample_size : int or None
        If set, stratified-sample this many rows (useful for quick testing).
    seed : int
        Random seed for sampling.

    Returns
    -------
    DataBundle
    """
    df_raw = _load_adult_raw()

    # ── Clean ────────────────────────────────────────────────────────────────
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Standardise label
    df[_ADULT_LABEL_COL] = df[_ADULT_LABEL_COL].str.rstrip(".")
    df[_ADULT_LABEL_COL] = df[_ADULT_LABEL_COL].map({">50K": 1, "<=50K": 0})

    # Drop rows where label is null
    df = df.dropna(subset=[_ADULT_LABEL_COL]).reset_index(drop=True)

    # Add age group column (for richer grouping in UI)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)

    # Impute
    df = _impute(df, _ADULT_CAT_COLS, _ADULT_NUM_COLS)

    # Keep a human-readable copy for the UI
    df_raw_clean = df.copy()

    # Optional stratified sample
    if sample_size and sample_size < len(df):
        df = (
            df.groupby(_ADULT_LABEL_COL, group_keys=False)
            .apply(lambda g: g.sample(frac=sample_size / len(df), random_state=seed))
            .reset_index(drop=True)
        )

    # ── Encode ───────────────────────────────────────────────────────────────
    encode_cols = _ADULT_CAT_COLS + ["age_group"]
    df_enc, _ = _encode_categoricals(df, encode_cols)
    df_enc[_ADULT_LABEL_COL] = df[_ADULT_LABEL_COL].values

    # Protected attribute numeric encodings
    # sex: Female=0, Male=1  (privileged = Male = 1)
    # race: Non-White=0, White=1
    sex_map = {"Female": 0, "Male": 1}
    race_map: dict[str, int] = {}
    for i, v in enumerate(sorted(df["race"].unique())):
        race_map[v] = 0 if v != "White" else 1

    df_enc["sex"] = df["sex"].map(sex_map).fillna(0).astype(int)
    df_enc["race"] = df["race"].map(race_map).fillna(0).astype(int)

    feature_cols = [c for c in df_enc.columns if c != _ADULT_LABEL_COL]
    X = df_enc[feature_cols]
    y = df_enc[_ADULT_LABEL_COL].astype(int)

    protected_cols = ["sex", "race", "age_group"]
    privileged_groups = [{"sex": 1}, {"race": 1}]
    unprivileged_groups = [{"sex": 0}, {"race": 0}]

    # ── AIF360 dataset object ─────────────────────────────────────────────────
    df_aif = df_enc.copy()
    dataset_obj = _make_dataset(
        df=df_aif,
        label_name=_ADULT_LABEL_COL,
        protected_attribute_names=["sex", "race"],
        favorable_label=_ADULT_FAVORABLE_LABEL,
        unfavorable_label=0.0,
        privileged_protected_attributes=[[1.0], [1.0]],
        unprivileged_protected_attributes=[[0.0], [0.0]],
    )

    # ── Group stats ───────────────────────────────────────────────────────────
    group_stats = {
        "sex": get_group_stats(df, "sex", _ADULT_LABEL_COL),
        "race": get_group_stats(df, "race", _ADULT_LABEL_COL),
        "age_group": get_group_stats(df, "age_group", _ADULT_LABEL_COL),
    }

    return DataBundle(
        df_raw=df_raw_clean,
        df_encoded=df_enc,
        X=X,
        y=y,
        label_col=_ADULT_LABEL_COL,
        protected_cols=protected_cols,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        favorable_label=_ADULT_FAVORABLE_LABEL,
        dataset_obj=dataset_obj,
        feature_names=feature_cols,
        categorical_cols=encode_cols,
        numeric_cols=_ADULT_NUM_COLS,
        dataset_name="UCI Adult Income",
        group_stats=group_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# German Credit
# ─────────────────────────────────────────────────────────────────────────────

_GERMAN_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "credit_risk",
]

_GERMAN_CAT_COLS = [
    "status", "credit_history", "purpose", "savings", "employment",
    "personal_status_sex", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone",
    "foreign_worker",
]

_GERMAN_NUM_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]

_GERMAN_LABEL_COL = "credit_risk"
_GERMAN_FAVORABLE_LABEL = 1.0  # credit risk=1 means good credit

# German Credit personal_status_sex codes
# A91 = male divorced/separated, A92 = female divorced/separated/married
# A93 = male single, A94 = male married/widowed, A95 = female single
_GERMAN_MALE_CODES = {"A91", "A93", "A94"}
_GERMAN_FEMALE_CODES = {"A92", "A95"}


def _download_german() -> pd.DataFrame:
    """Download German Credit data from UCI."""
    import requests

    logger.info("Downloading German Credit dataset from UCI...")
    resp = requests.get(GERMAN_URL, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(
        io.StringIO(resp.text),
        header=None,
        names=_GERMAN_COLUMNS,
        sep=" ",
        na_values=["?", ""],
    )


def _load_german_raw() -> pd.DataFrame:
    """Load raw German Credit data from disk cache or download."""
    cache_path = DATA_DIR / "german" / "german.csv"
    if cache_path.exists():
        logger.info("Loading German Credit dataset from cache: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("German Credit dataset not cached — downloading from UCI...")
    df = _download_german()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Cached German Credit dataset to %s", cache_path)
    return df


def load_german(sample_size: int | None = None, seed: int = 42) -> DataBundle:
    """
    Load, clean, encode, and wrap the German Credit dataset.

    Protective attributes: derived sex (from personal_status_sex) and age group.

    Parameters
    ----------
    sample_size : int or None
        If set, stratified-sample this many rows.
    seed : int

    Returns
    -------
    DataBundle
    """
    df_raw = _load_german_raw()

    # ── Clean ────────────────────────────────────────────────────────────────
    df = df_raw.copy()

    # Standardise label: 1=good, 2=bad → map to 1=good, 0=bad
    df[_GERMAN_LABEL_COL] = df[_GERMAN_LABEL_COL].map({1: 1, 2: 0})
    df = df.dropna(subset=[_GERMAN_LABEL_COL]).reset_index(drop=True)

    # Derive sex column from personal_status_sex
    df["sex"] = df["personal_status_sex"].apply(
        lambda x: 1 if str(x).strip() in _GERMAN_MALE_CODES else 0
    )

    # Add age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)

    # Impute
    df = _impute(df, _GERMAN_CAT_COLS, _GERMAN_NUM_COLS)

    df_raw_clean = df.copy()

    # Optional sample
    if sample_size and sample_size < len(df):
        df = (
            df.groupby(_GERMAN_LABEL_COL, group_keys=False)
            .apply(lambda g: g.sample(frac=sample_size / len(df), random_state=seed))
            .reset_index(drop=True)
        )

    # ── Encode ───────────────────────────────────────────────────────────────
    encode_cols = _GERMAN_CAT_COLS + ["age_group"]
    df_enc, _ = _encode_categoricals(df, encode_cols)
    df_enc[_GERMAN_LABEL_COL] = df[_GERMAN_LABEL_COL].values
    df_enc["sex"] = df["sex"].values

    feature_cols = [c for c in df_enc.columns if c != _GERMAN_LABEL_COL]
    X = df_enc[feature_cols]
    y = df_enc[_GERMAN_LABEL_COL].astype(int)

    protected_cols = ["sex", "age_group"]
    privileged_groups = [{"sex": 1}]
    unprivileged_groups = [{"sex": 0}]

    # ── AIF360 dataset object ─────────────────────────────────────────────────
    df_aif = df_enc.copy()
    dataset_obj = _make_dataset(
        df=df_aif,
        label_name=_GERMAN_LABEL_COL,
        protected_attribute_names=["sex"],
        favorable_label=_GERMAN_FAVORABLE_LABEL,
        unfavorable_label=0.0,
        privileged_protected_attributes=[[1.0]],
        unprivileged_protected_attributes=[[0.0]],
    )

    # ── Group stats ───────────────────────────────────────────────────────────
    group_stats = {
        "sex": get_group_stats(df, "sex", _GERMAN_LABEL_COL),
        "age_group": get_group_stats(df, "age_group", _GERMAN_LABEL_COL),
    }

    return DataBundle(
        df_raw=df_raw_clean,
        df_encoded=df_enc,
        X=X,
        y=y,
        label_col=_GERMAN_LABEL_COL,
        protected_cols=protected_cols,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        favorable_label=_GERMAN_FAVORABLE_LABEL,
        dataset_obj=dataset_obj,
        feature_names=feature_cols,
        categorical_cols=encode_cols,
        numeric_cols=_GERMAN_NUM_COLS,
        dataset_name="German Credit",
        group_stats=group_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_custom(
    path: str | Path,
    target_col: str,
    protected_attrs: list[str],
    favorable_label: str | int | float,
    privileged_vals: dict[str, str | int | float],
    unprivileged_vals: dict[str, str | int | float],
    sample_size: int | None = None,
    seed: int = 42,
) -> DataBundle:
    """
    Load and prepare a user-supplied CSV for fairness analysis.

    Parameters
    ----------
    path : str or Path
        Absolute path to the uploaded CSV file.
    target_col : str
        Name of the binary target column.
    protected_attrs : list[str]
        Column names of protected attributes (must exist in the CSV).
    favorable_label : str, int, or float
        The label value considered a positive/good outcome.
    privileged_vals : dict[str, value]
        Mapping of protected_attr → value that defines the privileged group.
        Example: {"sex": "Male", "race": "White"}
    unprivileged_vals : dict[str, value]
        Mapping of protected_attr → value for the unprivileged group.
    sample_size : int or None
    seed : int

    Returns
    -------
    DataBundle

    Raises
    ------
    ValueError
        If required columns are missing, target column has >2 unique values,
        or protected attribute values are not found in the data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, na_values=["?", "NA", "N/A", ""])

    # ── Validate columns ─────────────────────────────────────────────────────
    missing = set([target_col] + protected_attrs) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")

    # ── Validate target ───────────────────────────────────────────────────────
    unique_labels = df[target_col].dropna().unique()
    if len(unique_labels) > 2:
        raise ValueError(
            f"Target column '{target_col}' has {len(unique_labels)} unique values; "
            "binary classification requires exactly 2."
        )

    # ── Validate protected attribute values ───────────────────────────────────
    for attr in protected_attrs:
        attr_vals = set(df[attr].dropna().unique())
        priv_val = privileged_vals.get(attr)
        unpriv_val = unprivileged_vals.get(attr)
        if priv_val is not None and priv_val not in attr_vals:
            raise ValueError(
                f"Privileged value '{priv_val}' not found in column '{attr}'. "
                f"Available values: {attr_vals}"
            )
        if unpriv_val is not None and unpriv_val not in attr_vals:
            raise ValueError(
                f"Unprivileged value '{unpriv_val}' not found in column '{attr}'. "
                f"Available values: {attr_vals}"
            )

    df_raw_clean = df.copy()

    # ── Encode label ──────────────────────────────────────────────────────────
    df["__label__"] = (df[target_col] == favorable_label).astype(int)
    df = df.dropna(subset=["__label__"]).reset_index(drop=True)

    # Optional sample
    if sample_size and sample_size < len(df):
        df = (
            df.groupby("__label__", group_keys=False)
            .apply(lambda g: g.sample(frac=sample_size / len(df), random_state=seed))
            .reset_index(drop=True)
        )

    # ── Detect column types ───────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target_col and c != "__label__"]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c not in (target_col, "__label__")]

    # ── Impute ────────────────────────────────────────────────────────────────
    df = _impute(df, cat_cols, num_cols)

    # ── Encode protected attributes ───────────────────────────────────────────
    priv_encoded: dict[str, float] = {}
    unpriv_encoded: dict[str, float] = {}
    for attr in protected_attrs:
        priv_val = privileged_vals.get(attr)
        df[f"__{attr}_encoded__"] = (df[attr] == priv_val).astype(float)
        priv_encoded[f"__{attr}_encoded__"] = 1.0
        unpriv_encoded[f"__{attr}_encoded__"] = 0.0

    # ── Encode all categoricals ───────────────────────────────────────────────
    df_enc, _ = _encode_categoricals(df, cat_cols)
    df_enc["__label__"] = df["__label__"].values

    # Add encoded protected attr cols
    for attr in protected_attrs:
        df_enc[f"__{attr}_encoded__"] = df[f"__{attr}_encoded__"].values

    feature_cols = [
        c for c in df_enc.columns
        if c not in ("__label__", target_col)
    ]
    X = df_enc[feature_cols]
    y = df_enc["__label__"].astype(int)

    encoded_protected_cols = [f"__{a}_encoded__" for a in protected_attrs]
    priv_groups_aif = [priv_encoded]
    unpriv_groups_aif = [unpriv_encoded]

    # ── AIF360 dataset object ─────────────────────────────────────────────────
    # AIF360 requires a fully-numeric DataFrame — drop original string target
    # column and any remaining object-dtype columns before passing.
    df_aif = df_enc.copy()
    df_aif["__label__"] = y.values

    # Drop the original (potentially string) target column if it survived encoding
    if target_col in df_aif.columns:
        df_aif = df_aif.drop(columns=[target_col])

    # Drop any remaining object columns (safety guard for partially-encoded data)
    obj_cols = df_aif.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        logger.warning("Dropping non-numeric columns from AIF360 input: %s", obj_cols)
        df_aif = df_aif.drop(columns=obj_cols)

    # Ensure all columns are float-compatible
    df_aif = df_aif.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    dataset_obj = _make_dataset(
        df=df_aif,
        label_name="__label__",
        protected_attribute_names=encoded_protected_cols,
        favorable_label=1.0,
        unfavorable_label=0.0,
        privileged_protected_attributes=[[1.0]] * len(protected_attrs),
        unprivileged_protected_attributes=[[0.0]] * len(protected_attrs),
    )

    # ── Group stats ───────────────────────────────────────────────────────────
    group_stats = {}
    for attr in protected_attrs:
        try:
            group_stats[attr] = get_group_stats(df, attr, "__label__")
        except Exception:
            group_stats[attr] = pd.DataFrame()

    return DataBundle(
        df_raw=df_raw_clean,
        df_encoded=df_enc,
        X=X,
        y=y,
        label_col="__label__",
        protected_cols=encoded_protected_cols,
        privileged_groups=priv_groups_aif,
        unprivileged_groups=unpriv_groups_aif,
        favorable_label=1.0,
        dataset_obj=dataset_obj,
        feature_names=feature_cols,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        dataset_name=f"Custom: {path.name}",
        group_stats=group_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience registry
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "UCI Adult Income": {
        "loader": load_adult,
        "description": (
            "48 842 US Census records (1994). Predict whether income exceeds $50K/yr. "
            "Protected attributes: sex, race, age group."
        ),
        "protected_attrs": ["sex", "race", "age_group"],
        "label": "income",
        "known_issues": (
            "Historical data with known under-representation of women and minorities. "
            "Income threshold ($50K in 1994) is not inflation-adjusted."
        ),
    },
    "German Credit": {
        "loader": load_german,
        "description": (
            "1 000 German credit risk records. Predict creditworthiness (good/bad). "
            "Protected attributes: sex (derived from personal status), age group."
        ),
        "protected_attrs": ["sex", "age_group"],
        "label": "credit_risk",
        "known_issues": (
            "Small dataset (n=1000). Sex is inferred from personal_status_sex field, "
            "which conflates marital status. Age binarization threshold (25) is arbitrary."
        ),
    },
}

