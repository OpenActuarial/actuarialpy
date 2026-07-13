"""Tests for the shared numeric validators and the guards wired onto them.

The core failure these protect against: a bare ``x <= 0`` test lets ``NaN``
through, because every comparison with ``NaN`` is ``False``. Missing/infinite
data would then reach a log, a division, or a regression and propagate silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from actuarialpy.reserving import ChainLadder
from actuarialpy.trend import fit_trend
from actuarialpy.validation import (
    validate_finite,
    validate_nonnegative,
    validate_positive,
    validate_probability,
    validate_quantile,
    validate_weights,
)


# --------------------------------------------------------------------------- #
# validators
# --------------------------------------------------------------------------- #
def test_validate_finite_rejects_nan_and_inf():
    validate_finite([1.0, 2.0, 3.0])  # no raise
    for bad in ([1.0, np.nan], [np.inf, 1.0], [-np.inf]):
        with pytest.raises(ValueError):
            validate_finite(bad, "x")


def test_validate_positive_rejects_nan_that_slips_past_naive_check():
    arr = np.array([1.0, np.nan, 2.0])
    # the naive check the validator replaces would pass this:
    assert not np.any(arr <= 0)
    with pytest.raises(ValueError):
        validate_positive(arr, "rate")
    with pytest.raises(ValueError):
        validate_positive([1.0, 0.0], "rate")
    with pytest.raises(ValueError):
        validate_positive([1.0, -0.5], "rate")
    validate_positive([0.1, 5.0])  # no raise


def test_validate_nonnegative():
    validate_nonnegative([0.0, 1.0])
    with pytest.raises(ValueError):
        validate_nonnegative([0.0, -1e-9])
    with pytest.raises(ValueError):
        validate_nonnegative([0.0, np.nan])


def test_validate_probability_bounds():
    validate_probability([0.0, 0.5, 1.0])
    validate_probability(0.3, closed="neither")
    for closed, bad in [("both", 1.1), ("neither", 1.0), ("neither", 0.0), ("right", 0.0), ("left", 1.0)]:
        with pytest.raises(ValueError):
            validate_probability(bad, closed=closed)
    with pytest.raises(ValueError):
        validate_probability(np.nan)


def test_validate_quantile_open_interval():
    validate_quantile(0.995)
    for bad in [0.0, 1.0, np.nan]:
        with pytest.raises(ValueError):
            validate_quantile(bad)


def test_validate_weights():
    validate_weights([1.0, 2.0, 0.0])
    with pytest.raises(ValueError):
        validate_weights([0.0, 0.0])  # sums to zero
    with pytest.raises(ValueError):
        validate_weights([1.0, -1.0])
    with pytest.raises(ValueError):
        validate_weights([1.0, np.nan])


# --------------------------------------------------------------------------- #
# fit_trend rejects a non-finite rate (previously produced a silent NaN fit)
# --------------------------------------------------------------------------- #
def _trend_frame(claims):
    dates = pd.date_range("2022-01-01", periods=len(claims), freq="MS")
    return pd.DataFrame({"month": dates, "claims": claims, "mm": [1000.0] * len(claims)})


def test_fit_trend_rejects_non_finite_rate():
    good = _trend_frame([100.0, 110.0, 121.0, 133.0, 146.0])
    fit_trend(good, value_col="claims", date_col="month", exposure_col="mm")  # no raise

    bad = _trend_frame([100.0, 110.0, np.inf, 133.0, 146.0])
    with pytest.raises(ValueError):
        fit_trend(bad, value_col="claims", date_col="month", exposure_col="mm")


# --------------------------------------------------------------------------- #
# reserving denominator guards
# --------------------------------------------------------------------------- #
def _tri(rows):
    return pd.DataFrame(rows, index=[chr(ord("A") + i) for i in range(len(next(iter(rows.values()))))])


def test_chain_ladder_rejects_negative_cumulative_volume():
    tri = pd.DataFrame({0: [-100.0, -200.0], 1: [150.0, 300.0]}, index=["A", "B"])
    with pytest.raises(ValueError, match="non-positive or non-finite cumulative"):
        ChainLadder.fit(tri, method="volume")


def test_chain_ladder_rejects_zero_next_column_factor():
    # next column all zero -> volume link factor 0 -> would give inf completion
    tri = pd.DataFrame({0: [100.0, 200.0], 1: [0.0, 0.0]}, index=["A", "B"])
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        ChainLadder.fit(tri, method="volume")


def test_chain_ladder_simple_rejects_nonpositive_prior_cell():
    tri = pd.DataFrame({0: [100.0, 0.0], 1: [150.0, 300.0]}, index=["A", "B"])
    with pytest.raises(ValueError, match="strictly positive prior cumulatives"):
        ChainLadder.fit(tri, method="simple")


def test_chain_ladder_still_fits_valid_triangle():
    tri = pd.DataFrame(
        {0: [100.0, 200.0, 400.0], 1: [150.0, 300.0, float("nan")], 2: [165.0, float("nan"), float("nan")]},
        index=["A", "B", "C"],
    )
    cl = ChainLadder.fit(tri)
    assert cl.age_to_age[0] == pytest.approx(1.5)
    assert (cl.completion_factors <= 1.0).all()


# --------------------------------------------------------------------------- #
# credibility weight checks now reject NaN
# --------------------------------------------------------------------------- #
def test_credibility_rejects_nan_weight():
    from actuarialpy.credibility import BuhlmannStraub

    BuhlmannStraub(100.0, 50.0, 10.0, np.array([1000.0, 2000.0]))  # finite weights, no raise
    with pytest.raises(ValueError):
        # a NaN weight previously slipped past the bare `np.any(weights <= 0)` check
        BuhlmannStraub(100.0, 50.0, 10.0, np.array([1000.0, np.nan]))
