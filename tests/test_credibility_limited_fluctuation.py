"""Tests for limited-fluctuation (classical) credibility."""

import numpy as np
import pandas as pd
import pytest

from actuarialpy import (
    credibility_weighted_estimate,
    full_credibility_claims,
    limited_fluctuation_z,
)


def test_square_root_rule():
    assert limited_fluctuation_z(270, 1080) == pytest.approx(0.5)
    assert limited_fluctuation_z(1080, 1080) == pytest.approx(1.0)


def test_capped_at_one():
    assert limited_fluctuation_z(5000, 1080) == 1.0


def test_zero_exposure_is_zero():
    assert limited_fluctuation_z(0, 1080) == 0.0


def test_scalar_returns_float():
    assert isinstance(limited_fluctuation_z(270, 1080), float)


def test_series_preserves_index_and_name():
    s = pd.Series([1080, 270, 30], index=["a", "b", "c"], name="n")
    z = limited_fluctuation_z(s, 1080)
    assert isinstance(z, pd.Series)
    assert list(z.index) == ["a", "b", "c"]
    assert z.name == "n"
    assert z.loc["a"] == pytest.approx(1.0)
    assert z.loc["b"] == pytest.approx(0.5)


def test_array_input():
    z = limited_fluctuation_z(np.array([1080, 270]), 1080)
    assert isinstance(z, np.ndarray)
    assert z.tolist() == pytest.approx([1.0, 0.5])


def test_invalid_standard_raises():
    with pytest.raises(ValueError):
        limited_fluctuation_z(100, 0)


def test_full_credibility_classic_value():
    # 90% confidence, 5% tolerance -> ~1082 claims
    assert full_credibility_claims(confidence=0.90, tolerance=0.05) == pytest.approx(1082.2, abs=1.0)


def test_full_credibility_severity_cv_inflation():
    base = full_credibility_claims()
    inflated = full_credibility_claims(severity_cv=2.0)
    assert inflated == pytest.approx(base * (1 + 2.0**2))


def test_full_credibility_invalid_params():
    with pytest.raises(ValueError):
        full_credibility_claims(confidence=1.5)
    with pytest.raises(ValueError):
        full_credibility_claims(tolerance=0)
    with pytest.raises(ValueError):
        full_credibility_claims(severity_cv=-1)


def test_blend_pulls_thin_groups_to_complement():
    df = pd.DataFrame({"n": [1080, 30], "exp": [0.95, 0.60]})
    z = limited_fluctuation_z(df["n"], 1080)
    blended = credibility_weighted_estimate(df["exp"], 0.85, z)
    assert blended.iloc[0] == pytest.approx(0.95)            # full credibility -> own experience
    assert 0.60 < blended.iloc[1] < 0.85                     # thin -> pulled toward 0.85 manual
