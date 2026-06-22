"""Simple forecast and expected-value helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.compare import variance, variance_pct
from actuarialpy.metrics import actual_to_expected, per_exposure
from actuarialpy.trend import project_forward, trend_factor


def expected_from_rate(rate, exposure):
    """Expected amount from a rate/PMPM and exposure."""
    return rate * exposure


def forecast_from_rate(
    base_rate,
    exposure,
    *,
    annual_trend: float = 0.0,
    months_forward: float = 0.0,
):
    """Forecast an amount from base rate, exposure, trend, and elapsed months."""
    trended_rate = project_forward(base_rate, annual_trend, months_forward)
    return expected_from_rate(trended_rate, exposure)


def forecast_experience(
    df: pd.DataFrame,
    *,
    rate_col: str,
    exposure_col: str,
    annual_trend: float | str = 0.0,
    months_forward: float | str = 0.0,
    forecast_col: str = "expected_expense",
    trended_rate_col: str = "expected_rate",
    copy: bool = True,
) -> pd.DataFrame:
    """Create forecast/expected amounts from rates, exposures, and trend."""
    required = [rate_col, exposure_col]
    if isinstance(annual_trend, str):
        required.append(annual_trend)
    if isinstance(months_forward, str):
        required.append(months_forward)
    validate_columns(df, required)
    result = df.copy() if copy else df
    trend_values = result[annual_trend] if isinstance(annual_trend, str) else annual_trend
    months_values = result[months_forward] if isinstance(months_forward, str) else months_forward
    result[trended_rate_col] = result[rate_col] * ((1 + trend_values) ** (months_values / 12))
    result[forecast_col] = result[trended_rate_col] * result[exposure_col]
    return result


def compare_actual_to_expected(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    on: str | Iterable[str],
    actual_col: str,
    expected_col: str,
    how: str = "left",
) -> pd.DataFrame:
    """Join actual and expected tables and calculate A/E and variance metrics."""
    keys = as_list(on)
    validate_columns(actual, keys + [actual_col])
    validate_columns(expected, keys + [expected_col])
    result = actual.merge(expected[keys + [expected_col]], on=keys, how=how, validate="many_to_one")
    result["actual_to_expected"] = actual_to_expected(result[actual_col], result[expected_col])
    result["variance"] = variance(result[actual_col], result[expected_col])
    result["variance_pct"] = variance_pct(result[actual_col], result[expected_col])
    return result
