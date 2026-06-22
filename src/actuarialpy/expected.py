"""Actual-versus-expected experience summaries."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from actuarialpy.columns import as_list, sum_columns, validate_columns
from actuarialpy.metrics import actual_to_expected as actual_to_expected_ratio, per_exposure, safe_divide


def _per_exposure_name(prefix: str, exposure_col: str) -> str:
    if exposure_col == "member_months":
        return f"{prefix}_pmpm"
    if exposure_col == "subscriber_months":
        return f"{prefix}_pspm"
    if exposure_col == "employee_months":
        return f"{prefix}_pepm"
    return f"{prefix}_per_{exposure_col}"


def summarize_actual_vs_expected(
    df: pd.DataFrame,
    *,
    groupby: str | Iterable[str] | None = None,
    actual_cols: str | Iterable[str],
    expected_cols: str | Iterable[str],
    exposure_cols: str | Iterable[str] | None = None,
    actual_name: str = "actual",
    expected_name: str = "expected",
    ae_name: str = "actual_to_expected",
    variance_name: str = "variance",
    variance_pct_name: str = "variance_pct",
) -> pd.DataFrame:
    """Summarize actual-versus-expected results by optional grouping columns.

    Actual and expected amounts are aggregated before ratios are calculated.
    This makes the function suitable for claim costs, benefits, expenses,
    revenue, or any other actual-versus-expected measure.
    """
    groups = as_list(groupby)
    actuals = as_list(actual_cols)
    expecteds = as_list(expected_cols)
    exposures = as_list(exposure_cols)
    validate_columns(df, groups + actuals + expecteds + exposures)

    amount_cols = list(dict.fromkeys(actuals + expecteds + exposures))
    if groups:
        out = df[groups + amount_cols].groupby(groups, dropna=False, as_index=False).sum(numeric_only=True)
    else:
        out = pd.DataFrame({col: [df[col].sum()] for col in amount_cols})

    out[actual_name] = sum_columns(out, actuals)
    out[expected_name] = sum_columns(out, expecteds)
    out[ae_name] = actual_to_expected_ratio(out[actual_name], out[expected_name])
    out[variance_name] = out[actual_name] - out[expected_name]
    out[variance_pct_name] = safe_divide(out[variance_name], out[expected_name])

    for exposure in exposures:
        out[_per_exposure_name(actual_name, exposure)] = per_exposure(out[actual_name], out[exposure])
        out[_per_exposure_name(expected_name, exposure)] = per_exposure(out[expected_name], out[exposure])
        out[_per_exposure_name(variance_name, exposure)] = per_exposure(out[variance_name], out[exposure])

    return out
