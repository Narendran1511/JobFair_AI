"""
tests/conftest.py
=================
Shared pytest fixtures for JobFair AI.

Fixtures are designed to be fast (synthetic data, no downloads) so the
test suite can run fully offline without any external resources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Tiny synthetic dataset fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_binary_df() -> pd.DataFrame:
    """
    A minimal 200-row binary-classification DataFrame with:
    - two numeric features (feat_a, feat_b)
    - two categorical features (cat_x, cat_y)
    - one binary protected attribute (sex: 0=female, 1=male)
    - one binary label (label: 0/1)

    The label is correlated with sex so bias metrics are non-trivial.
    """
    rng = np.random.default_rng(42)
    n = 200
    sex = rng.integers(0, 2, size=n)
    feat_a = rng.normal(loc=0.5 + 0.3 * sex, scale=0.5, size=n)
    feat_b = rng.normal(loc=0.0, scale=1.0, size=n)
    cat_x = rng.choice(["A", "B", "C"], size=n)
    cat_y = rng.choice(["X", "Y"], size=n)
    # Label: positively correlated with sex and feat_a
    p = 1 / (1 + np.exp(-(0.5 * sex + 0.8 * feat_a - 0.3)))
    label = (rng.uniform(size=n) < p).astype(int)

    return pd.DataFrame(
        {
            "feat_a": feat_a,
            "feat_b": feat_b,
            "cat_x": cat_x,
            "cat_y": cat_y,
            "sex": sex,
            "label": label,
        }
    )


@pytest.fixture(scope="session")
def biased_df() -> pd.DataFrame:
    """
    A 100-row DataFrame where the privileged group (sex=1) always gets
    label=1 and the unprivileged group (sex=0) always gets label=0.

    SPD should be exactly 1.0, DI should be undefined (0-division → inf or 0).
    Used to verify extreme bias detection.
    """
    n = 100
    sex = np.array([1] * 50 + [0] * 50)
    label = sex.copy()
    feat = np.random.default_rng(0).normal(size=n)
    return pd.DataFrame({"feat": feat, "sex": sex, "label": label})


@pytest.fixture(scope="session")
def tiny_custom_csv(tmp_path_factory) -> str:
    """Write a tiny CSV to a temp file; return its path as str."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [30000, 50000, 70000, 90000, 110000],
            "gender": ["Female", "Male", "Female", "Male", "Male"],
            "approved": ["No", "Yes", "No", "Yes", "Yes"],
        }
    )
    tmp = tmp_path_factory.mktemp("custom") / "test.csv"
    df.to_csv(tmp, index=False)
    return str(tmp)


@pytest.fixture(scope="session")
def adult_sample(tmp_path_factory) -> "DataBundle":  # noqa: F821
    """
    Load a small (500-row) sample of the Adult dataset.
    Uses a cached local file if available; otherwise synthesises a mock.
    """
    import sys
    from pathlib import Path

    # Ensure core/ is importable
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from core.data_loader import DATA_DIR, DataBundle, load_adult

    cache = DATA_DIR / "adult" / "adult.csv"
    if cache.exists():
        return load_adult(sample_size=500, seed=42)

    # Synthesise a minimal Adult-like CSV so tests run offline
    _synthesise_adult_cache(cache)
    return load_adult(sample_size=500, seed=42)


def _synthesise_adult_cache(cache_path) -> None:
    """Create a synthetic Adult-like CSV for offline testing."""
    from pathlib import Path

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame(
        {
            "age": rng.integers(17, 90, size=n),
            "workclass": rng.choice(["Private", "Self-emp", "Gov"], size=n),
            "fnlwgt": rng.integers(10000, 1000000, size=n),
            "education": rng.choice(["Bachelors", "HS-grad", "Masters"], size=n),
            "education_num": rng.integers(1, 16, size=n),
            "marital_status": rng.choice(["Married", "Never-married", "Divorced"], size=n),
            "occupation": rng.choice(["Tech-support", "Sales", "Craft-repair"], size=n),
            "relationship": rng.choice(["Husband", "Not-in-family", "Wife"], size=n),
            "race": rng.choice(["White", "Black", "Asian-Pac-Islander"], size=n),
            "sex": rng.choice(["Male", "Female"], size=n),
            "capital_gain": rng.integers(0, 10000, size=n),
            "capital_loss": rng.integers(0, 2000, size=n),
            "hours_per_week": rng.integers(10, 80, size=n),
            "native_country": rng.choice(["United-States", "Mexico", "India"], size=n),
            "income": rng.choice([">50K", "<=50K"], p=[0.24, 0.76], size=n),
        }
    )
    df.to_csv(cache_path, index=False)
