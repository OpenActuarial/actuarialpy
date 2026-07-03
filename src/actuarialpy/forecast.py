"""Simple forecast and expected-value helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.compare import variance, variance_pct
from actuarialpy.metrics import actual_to_expected
from actuarialpy.trend import project_forward


def expected_from_rate(rate, exposure):
    """Expected amount from a per-exposure rate and exposure."""
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
    how: Literal["left", "right", "outer", "inner", "cross"] = "left",
    suffixes: tuple[str, str] = ("actual", "expected"),
) -> pd.DataFrame:
    """Join actual and expected tables and calculate A/E and variance metrics.

    The two frames are merged on ``on`` and the actual-to-expected ratio, variance,
    and variance percent are computed. Use ``how="outer"`` so that keys present on
    only one side -- for example forecast months that do not have actuals yet -- are
    kept, with the missing side coming back as ``NaN`` (so an unavailable actual is
    distinguishable from a true zero).

    Column-name collisions are handled automatically. If the actual and expected
    amount columns share a name (e.g. both frames call their value column
    ``"amount"``, which a plain merge would turn into ``amount_x`` / ``amount_y``),
    the output columns are named ``"{actual_col}_{suffixes[0]}"`` and
    ``"{expected_col}_{suffixes[1]}"`` -- by default ``amount_actual`` and
    ``amount_expected``. Pass ``suffixes=("actual", "forecast")`` for
    ``amount_actual`` / ``amount_forecast``. When the two columns already have
    distinct names they are left unchanged.
    """
    keys = as_list(on)
    validate_columns(actual, keys + [actual_col])
    validate_columns(expected, keys + [expected_col])

    actual_suffix, expected_suffix = suffixes
    actual_out, expected_out = actual_col, expected_col
    actual_frame = actual
    expected_subset = expected[keys + [expected_col]]
    other_actual_cols = set(actual.columns) - set(keys)

    if actual_col == expected_col:
        # same name on both sides -> disambiguate both
        actual_out = f"{actual_col}_{actual_suffix}"
        expected_out = f"{expected_col}_{expected_suffix}"
        actual_frame = actual.rename(columns={actual_col: actual_out})
        expected_subset = expected_subset.rename(columns={expected_col: expected_out})
    elif expected_col in other_actual_cols:
        # expected amount name collides with an unrelated actual column -> rename expected only
        expected_out = f"{expected_col}_{expected_suffix}"
        expected_subset = expected_subset.rename(columns={expected_col: expected_out})

    result = actual_frame.merge(
        expected_subset, on=keys, how=how, validate="many_to_one"
    )
    result["variance"] = variance(result[actual_out], result[expected_out])
    result["variance_pct"] = variance_pct(result[actual_out], result[expected_out])
    result["actual_to_expected"] = actual_to_expected(result[actual_out], result[expected_out])
    return result
