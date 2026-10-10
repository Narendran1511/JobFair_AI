"""
tests/test_data_loader.py
=========================
Phase 1 tests — data loading, schema validation, and group statistics.

All tests run offline: network-dependent tests use synthesised cache files
created by conftest._synthesise_adult_cache().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import (
    DataBundle,
    _FallbackDataset,
    _impute,
    _encode_categoricals,
    get_group_stats,
    load_adult,
    load_german,
    load_custom,
    DATASET_REGISTRY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Adult dataset schema and no-null guarantee
# ─────────────────────────────────────────────────────────────────────────────

class TestAdultDataset:
    """Tests for UCI Adult Income loader."""

    def test_schema_and_no_nulls(self, adult_sample: DataBundle) -> None:
        """
        After loading, the encoded DataFrame and X should have:
        - No NaN values
        - The label column should only contain 0 and 1
        - Protected columns should be present in X
        """
        bundle = adult_sample

        # No NaN in encoded data
        assert bundle.df_encoded.isnull().sum().sum() == 0, (
            "df_encoded contains NaN values after loading"
        )
        assert bundle.X.isnull().sum().sum() == 0, (
            "Feature matrix X contains NaN values"
        )

        # Label is binary
        unique_labels = set(bundle.y.unique())
        assert unique_labels <= {0, 1}, f"Label contains non-binary values: {unique_labels}"

        # Protected columns present in feature matrix
        for col in bundle.protected_cols:
            assert col in bundle.X.columns, f"Protected column '{col}' missing from X"

        # DataBundle fields are well-formed
        assert isinstance(bundle.feature_names, list)
        assert len(bundle.feature_names) > 0
        assert bundle.label_col in bundle.df_encoded.columns
        assert bundle.dataset_name == "UCI Adult Income"

    def test_group_stats_structure(self, adult_sample: DataBundle) -> None:
        """
        group_stats should have entries for 'sex', 'race', and 'age_group'.
        Each entry must be a DataFrame with required columns and non-negative counts.
        """
        bundle = adult_sample
        required_keys = {"sex", "race", "age_group"}
        assert required_keys.issubset(bundle.group_stats.keys()), (
            f"Missing group_stats keys: {required_keys - bundle.group_stats.keys()}"
        )

        for key, stats_df in bundle.group_stats.items():
            assert isinstance(stats_df, pd.DataFrame), f"group_stats['{key}'] is not a DataFrame"
            required_cols = {"group", "count", "fraction", "base_rate"}
            assert required_cols.issubset(stats_df.columns), (
                f"group_stats['{key}'] missing columns: {required_cols - set(stats_df.columns)}"
            )
            assert (stats_df["count"] >= 0).all(), f"Negative counts in group_stats['{key}']"
            # Fractions should sum to approximately 1
            assert abs(stats_df["fraction"].sum() - 1.0) < 0.01, (
                f"group_stats['{key}'] fractions do not sum to 1: {stats_df['fraction'].sum()}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: German Credit loader
# ─────────────────────────────────────────────────────────────────────────────

class TestGermanDataset:
    """Tests for German Credit loader."""

    def test_german_schema(self, tmp_path) -> None:
        """
        German Credit loader should produce a valid DataBundle.
        Uses synthesised cache if UCI is unreachable.
        """
        from core.data_loader import DATA_DIR
        from tests.conftest import _synthesise_adult_cache  # noqa: F401

        # Synthesise German cache if needed
        cache = DATA_DIR / "german" / "german.csv"
        if not cache.exists():
            _synthesise_german_cache(cache)

        bundle = load_german(sample_size=200, seed=0)

        assert bundle.dataset_name == "German Credit"
        assert bundle.label_col == "credit_risk"
        assert "sex" in bundle.protected_cols
        assert "age_group" in bundle.protected_cols
        assert bundle.X.isnull().sum().sum() == 0
        assert set(bundle.y.unique()) <= {0, 1}
        assert bundle.favorable_label == 1.0


def _synthesise_german_cache(cache_path) -> None:
    """Create a synthetic German Credit-like CSV for offline testing."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    n = 500
    personal_codes = ["A91", "A92", "A93", "A94", "A95"]
    df = pd.DataFrame(
        {
            "status": rng.choice(["A11", "A12", "A13", "A14"], size=n),
            "duration": rng.integers(4, 72, size=n),
            "credit_history": rng.choice(["A30", "A31", "A32", "A33", "A34"], size=n),
            "purpose": rng.choice(["A40", "A41", "A42", "A43"], size=n),
            "credit_amount": rng.integers(250, 20000, size=n),
            "savings": rng.choice(["A61", "A62", "A63", "A64", "A65"], size=n),
            "employment": rng.choice(["A71", "A72", "A73", "A74", "A75"], size=n),
            "installment_rate": rng.integers(1, 4, size=n),
            "personal_status_sex": rng.choice(personal_codes, size=n),
            "other_debtors": rng.choice(["A101", "A102", "A103"], size=n),
            "residence_since": rng.integers(1, 4, size=n),
            "property": rng.choice(["A121", "A122", "A123", "A124"], size=n),
            "age": rng.integers(19, 75, size=n),
            "other_installment_plans": rng.choice(["A141", "A142", "A143"], size=n),
            "housing": rng.choice(["A151", "A152", "A153"], size=n),
            "existing_credits": rng.integers(1, 4, size=n),
            "job": rng.choice(["A171", "A172", "A173", "A174"], size=n),
            "num_dependents": rng.integers(1, 2, size=n),
            "telephone": rng.choice(["A191", "A192"], size=n),
            "foreign_worker": rng.choice(["A201", "A202"], size=n),
            "credit_risk": rng.choice([1, 2], p=[0.7, 0.3], size=n),
        }
    )
    df.to_csv(cache_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Custom CSV loader — round-trip validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomLoader:
    """Tests for the generic custom CSV loader."""

    def test_custom_csv_round_trip(self, tiny_custom_csv: str) -> None:
        """
        Loading a minimal custom CSV should:
        - Produce a valid DataBundle
        - Set the label to 0/1 binary based on favorable_label
        - Surface protected columns correctly
        - Not raise any exceptions
        """
        bundle = load_custom(
            path=tiny_custom_csv,
            target_col="approved",
            protected_attrs=["gender"],
            favorable_label="Yes",
            privileged_vals={"gender": "Male"},
            unprivileged_vals={"gender": "Female"},
        )

        assert isinstance(bundle, DataBundle)
        assert set(bundle.y.unique()) <= {0, 1}, f"Unexpected labels: {bundle.y.unique()}"
        assert len(bundle.X) == len(bundle.y)
        assert bundle.X.isnull().sum().sum() == 0

        # Favorable label count: "Yes" appears 3 times in 5 rows
        assert bundle.y.sum() == 3, f"Expected 3 positive labels, got {bundle.y.sum()}"

    def test_custom_csv_missing_column_raises(self, tiny_custom_csv: str) -> None:
        """Loader must raise ValueError if a required column is absent."""
        with pytest.raises(ValueError, match="Columns not found"):
            load_custom(
                path=tiny_custom_csv,
                target_col="approved",
                protected_attrs=["nonexistent_column"],
                favorable_label="Yes",
                privileged_vals={"nonexistent_column": "Male"},
                unprivileged_vals={"nonexistent_column": "Female"},
            )

    def test_custom_csv_bad_privileged_val_raises(self, tiny_custom_csv: str) -> None:
        """Loader must raise ValueError if a privileged value is absent from the column."""
        with pytest.raises(ValueError, match="Privileged value"):
            load_custom(
                path=tiny_custom_csv,
                target_col="approved",
                protected_attrs=["gender"],
                favorable_label="Yes",
                privileged_vals={"gender": "Alien"},  # not in data
                unprivileged_vals={"gender": "Female"},
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: get_group_stats correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupStats:
    """Tests for the get_group_stats utility."""

    def test_group_stats_values(self) -> None:
        """
        Hand-computed case:
        - Group sex=1 (male): 4 rows, label sum = 3, base_rate = 0.75
        - Group sex=0 (female): 6 rows, label sum = 1, base_rate = 0.1667
        - Overall base_rate = 4/10 = 0.4
        """
        df = pd.DataFrame(
            {
                "sex": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                "label": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            }
        )
        stats = get_group_stats(df, protected_col="sex", label_col="label")

        assert len(stats) == 2, "Expected stats for 2 groups"

        male_row = stats[stats["group"] == "1"].iloc[0]
        female_row = stats[stats["group"] == "0"].iloc[0]

        assert male_row["count"] == 4
        assert female_row["count"] == 6
        assert abs(male_row["base_rate"] - 0.75) < 0.001
        assert abs(female_row["base_rate"] - (1 / 6)) < 0.001
        # base_rate_diff_from_overall = group_rate - 0.4
        assert abs(male_row["base_rate_diff_from_overall"] - (0.75 - 0.4)) < 0.001
        assert abs(female_row["base_rate_diff_from_overall"] - ((1 / 6) - 0.4)) < 0.001

    def test_group_stats_fractions_sum_to_one(self) -> None:
        """Fractions across all groups must sum to 1.0."""
        df = pd.DataFrame(
            {
                "race": ["White", "White", "Black", "Asian", "Black", "White"],
                "label": [1, 0, 1, 0, 1, 1],
            }
        )
        stats = get_group_stats(df, protected_col="race", label_col="label")
        assert abs(stats["fraction"].sum() - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Fallback dataset shim
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackDataset:
    """Tests for the _FallbackDataset shim used when AIF360 is absent."""

    def test_fallback_shapes(self) -> None:
        """Verify that _FallbackDataset produces correct shapes and dtypes."""
        df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3, 0.4],
                "sex": [1.0, 0.0, 1.0, 0.0],
                "label": [1.0, 0.0, 1.0, 1.0],
            }
        )
        ds = _FallbackDataset(
            df=df,
            label_name="label",
            protected_attribute_names=["sex"],
            favorable_label=1.0,
            unfavorable_label=0.0,
            privileged_protected_attributes=[[1.0]],
            unprivileged_protected_attributes=[[0.0]],
        )

        assert ds.features.shape == (4, 2), f"Unexpected features shape: {ds.features.shape}"
        assert ds.labels.shape == (4, 1), f"Unexpected labels shape: {ds.labels.shape}"
        assert ds.instance_weights.shape == (4,)
        assert ds.protected_attributes.shape == (4, 1)

    def test_fallback_copy(self) -> None:
        """copy() must return an independent object."""
        df = pd.DataFrame({"feat": [1.0, 2.0], "sex": [1.0, 0.0], "label": [1.0, 0.0]})
        ds = _FallbackDataset(df=df, label_name="label", protected_attribute_names=["sex"])
        ds_copy = ds.copy()
        ds_copy.labels[0, 0] = 99.0
        assert ds.labels[0, 0] != 99.0, "copy() is not deep — mutations leak back to original"
