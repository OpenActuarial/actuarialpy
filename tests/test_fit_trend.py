"""Tests for fit_trend (log-linear trend regression) and TrendFit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from actuarialpy import annualized_trend, fit_trend, trend_factor
from actuarialpy.trend import _inverse_normal_cdf, _student_t_ppf


def _series(annual_trend, months=36, start="2022-01-01", noise=None, seed=0):
    dates = pd.date_range(start, periods=months, freq="MS")
    t = np.asarray((dates - dates[0]).days) / 365.25
    pmpm = 300.0 * (1.0 + annual_trend) ** t
    if noise:
        pmpm = pmpm * np.random.default_rng(seed).normal(1.0, noise, months)
    return pd.DataFrame({"month": dates, "claims": pmpm * 1000.0, "mm": 1000.0}), t


def test_recovers_known_trend_exactly_without_noise():
    df, _ = _series(0.072)
    fit = fit_trend(df, value_col="claims", date_col="month", exposure_col="mm")
    assert abs(fit.annual_trend - 0.072) < 1e-9
    assert fit.r_squared > 0.99999
    assert fit.std_error < 1e-9
    assert fit.n_periods == 36


def test_confidence_interval_contains_truth_under_noise():
    df, _ = _series(0.072, noise=0.03, seed=7)
    fit = fit_trend(df, value_col="claims", date_col="month", exposure_col="mm")
    assert fit.ci_low <= 0.072 <= fit.ci_high
    assert fit.ci_low < fit.annual_trend < fit.ci_high


def test_robust_to_outlier_endpoint_where_two_point_is_not():
    df, _ = _series(0.072)
    pmpm = np.array(df["claims"] / df["mm"], dtype="float64")
    pmpm[-1] *= 1.25  # a spike in the latest month
    spiked = df.assign(claims=pmpm * df["mm"])
    fit = fit_trend(spiked, value_col="claims", date_col="month", exposure_col="mm")
    two_point = annualized_trend(pmpm[-1], pmpm[0], months_between=len(pmpm) - 1)
    # the regression barely moves; the two-point CAGR is thrown right off
    assert abs(fit.annual_trend - 0.072) < abs(two_point - 0.072)
    assert abs(fit.annual_trend - 0.072) < 0.02


def test_fits_the_rate_not_the_level():
    # claims grow from both PMPM trend and membership growth; only the PMPM trend is fitted
    df, t = _series(0.072)
    members = 1000.0 * (1.02) ** t
    pmpm = (df["claims"] / df["mm"]).to_numpy()
    grown = pd.DataFrame({"month": df["month"], "claims": pmpm * members, "mm": members})
    fit = fit_trend(grown, value_col="claims", date_col="month", exposure_col="mm")
    assert abs(fit.annual_trend - 0.072) < 1e-6


def test_without_exposure_fits_value_directly():
    df, _ = _series(0.05)
    fit = fit_trend(df.assign(level=df["claims"] / df["mm"]), value_col="level", date_col="month")
    assert abs(fit.annual_trend - 0.05) < 1e-9


def test_robust_to_a_missing_period():
    df, _ = _series(0.06)
    gapped = df.drop(df.index[10]).reset_index(drop=True)  # one month missing
    fit = fit_trend(gapped, value_col="claims", date_col="month", exposure_col="mm")
    assert abs(fit.annual_trend - 0.06) < 1e-6  # time is measured from real dates


def test_factor_bridges_to_application():
    df, _ = _series(0.08)
    fit = fit_trend(df, value_col="claims", date_col="month", exposure_col="mm")
    assert np.isclose(fit.factor(18), trend_factor(fit.annual_trend, 18))
    assert np.isclose(fit.factor(12), 1.0 + fit.annual_trend)
    assert fit.ci == (fit.ci_low, fit.ci_high)


def test_quarterly_frequency():
    dates = pd.date_range("2020-01-01", periods=18, freq="QS")
    t = np.asarray((dates - dates[0]).days) / 365.25
    df = pd.DataFrame({"q": dates, "v": 500.0 * (1.05) ** t, "e": 1.0})
    fit = fit_trend(df, value_col="v", date_col="q", exposure_col="e", freq="Q")
    assert abs(fit.annual_trend - 0.05) < 1e-6


def test_validation_errors():
    df, _ = _series(0.05)
    with pytest.raises(ValueError):  # too few periods
        fit_trend(df.head(2), value_col="claims", date_col="month", exposure_col="mm")
    with pytest.raises(ValueError):  # non-positive cannot be logged
        fit_trend(pd.DataFrame({"m": df["month"][:4], "v": [1.0, -2.0, 3.0, 4.0]}), value_col="v", date_col="m")
    with pytest.raises(ValueError):  # confidence out of range
        fit_trend(df, value_col="claims", date_col="month", exposure_col="mm", confidence=1.5)


def test_critical_value_helpers_match_tables():
    assert abs(_inverse_normal_cdf(0.975) - 1.959964) < 1e-5
    assert abs(_inverse_normal_cdf(0.95) - 1.644854) < 1e-5
    for df_val, expected in {5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042, 120: 1.980}.items():
        assert abs(_student_t_ppf(0.975, df_val) - expected) < 0.002
