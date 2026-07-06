"""Persistency tests: the renewal-probability model and its fit."""

import numpy as np
import pandas as pd
import pytest

from actuarialpy import Persistency, fit_persistency


# --------------------------------------------------------------------------- #
# Persistency.probability
# --------------------------------------------------------------------------- #
def test_probability_scalar_returns_float():
    p = Persistency(base_retention=0.90, rate_elasticity=1.0)
    out = p.probability(0.10)
    assert isinstance(out, float)
    assert out == pytest.approx(0.80)


def test_probability_zero_rate_change_is_base():
    p = Persistency(base_retention=0.92, rate_elasticity=1.5)
    assert p.probability(0.0) == pytest.approx(0.92)


def test_probability_is_clipped():
    p = Persistency(base_retention=0.90, rate_elasticity=1.0)
    assert p.probability(2.00) == pytest.approx(0.0)   # would go negative -> floor
    assert p.probability(-1.0) == pytest.approx(1.0)   # would exceed 1 -> cap


def test_probability_array_and_series():
    p = Persistency(base_retention=0.90, rate_elasticity=1.0)
    arr = p.probability(np.array([0.0, 0.05, 0.10]))
    assert arr == pytest.approx([0.90, 0.85, 0.80])
    ser = p.probability(pd.Series([0.0, 0.20], index=["a", "b"]))
    assert list(ser.index) == ["a", "b"]
    assert ser.to_numpy() == pytest.approx([0.90, 0.70])


def test_invalid_floor_cap_raises():
    with pytest.raises(ValueError):
        Persistency(base_retention=0.9, floor=0.6, cap=0.4)


# --------------------------------------------------------------------------- #
# fit_persistency
# --------------------------------------------------------------------------- #
def test_fit_recovers_line_exactly():
    base, elas = 0.95, 1.2
    rc = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    renewed = base - elas * rc
    fit = fit_persistency(rc, renewed)
    assert fit.base_retention == pytest.approx(base, abs=1e-9)
    assert fit.rate_elasticity == pytest.approx(elas, abs=1e-9)


def test_fit_requires_two_observations():
    with pytest.raises(ValueError):
        fit_persistency([0.05], [0.9])


def test_fit_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fit_persistency([0.0, 0.1], [0.9])
