"""Completion-factor, completed-claims, and IBNR tools."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.metrics import safe_divide


def validate_completion_factors(factors: pd.DataFrame, factor_col: str = "completion_factor", *, method: str = "divide") -> None:
    """Validate completion-factor values for a selected convention."""
    validate_columns(factors, [factor_col])
    values = factors[factor_col]
    bad_missing = values.isna()
    if bad_missing.any():
        raise ValueError(f"{bad_missing.sum()} completion factors are missing")
    if method == "divide":
        bad = (values <= 0) | (values > 1)
        if bad.any():
            raise ValueError("divide-method completion factors should generally satisfy 0 < factor <= 1")
    elif method == "multiply":
        bad = values < 1
        if bad.any():
            raise ValueError("multiply-method completion factors should generally satisfy factor >= 1")
    else:
        raise ValueError("method must be either 'divide' or 'multiply'")


def completed_from_factor(paid, factor, *, method: str = "divide"):
    """Calculate completed claims from paid claims and a completion factor."""
    if method == "divide":
        return safe_divide(paid, factor)
    if method == "multiply":
        return paid * factor
    raise ValueError("method must be either 'divide' or 'multiply'")


def ibnr(completed, paid):
    """Calculate IBNR as completed minus paid."""
    return completed - paid


def complete_claims(
    df: pd.DataFrame,
    *,
    paid_col: str = "paid_claims",
    factor_col: str = "completion_factor",
    method: str = "divide",
    completed_col: str = "completed_claims",
    ibnr_col: str = "ibnr",
    validate_factors: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Add completed-claims and IBNR columns for one paid/factor pair.

    This function assumes the factor columns are already present. Use pandas
    ``merge`` directly when joining factor tables, especially when the factor
    table has several factor columns.
    """
    validate_columns(df, [paid_col, factor_col])
    if validate_factors:
        validate_completion_factors(df, factor_col, method=method)
    result = df.copy() if copy else df
    result[completed_col] = completed_from_factor(result[paid_col], result[factor_col], method=method)
    result[ibnr_col] = ibnr(result[completed_col], result[paid_col])
    return result


def complete_claim_components(
    df: pd.DataFrame,
    component_factor_map: dict[str, str],
    *,
    method: str = "divide",
    completed_suffix: str = "_completed",
    ibnr_suffix: str = "_ibnr",
    validate_factors: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Complete several claim components using component-specific factors.

    Example
    -------
    ``{"inpatient_claims": "inpatient_completion_factor"}`` creates
    ``inpatient_claims_completed`` and ``inpatient_claims_ibnr``.
    """
    if not component_factor_map:
        raise ValueError("component_factor_map must contain at least one component/factor pair")
    required = list(component_factor_map.keys()) + list(component_factor_map.values())
    validate_columns(df, required)
    result = df.copy() if copy else df
    for paid_col, factor_col in component_factor_map.items():
        if validate_factors:
            validate_completion_factors(result, factor_col, method=method)
        completed_col = f"{paid_col}{completed_suffix}"
        ibnr_col = f"{paid_col}{ibnr_suffix}"
        result[completed_col] = completed_from_factor(result[paid_col], result[factor_col], method=method)
        result[ibnr_col] = ibnr(result[completed_col], result[paid_col])
    return result


def lag_months(incurred_date, valuation_date):
    """Calculate valuation lag in whole months."""
    incurred = pd.to_datetime(incurred_date)
    valuation = pd.to_datetime(valuation_date)
    return (valuation.dt.year - incurred.dt.year) * 12 + (valuation.dt.month - incurred.dt.month) if hasattr(incurred, "dt") else (valuation.year - incurred.year) * 12 + (valuation.month - incurred.month)


def make_completion_triangle(
    df: pd.DataFrame,
    *,
    origin_col: str,
    valuation_col: str,
    amount_col: str,
    index_name: str = "origin_period",
    lag_name: str = "lag_month",
    cumulative: bool = False,
) -> pd.DataFrame:
    """Create a claims triangle by origin period and valuation lag.

    By default ``amount_col`` is assumed to already hold the value *as of* each
    valuation date (a cumulative-to-date snapshot); amounts are summed within
    each (origin, lag) cell and pivoted as-is. Set ``cumulative=True`` when
    ``amount_col`` holds *incremental* amounts at each lag, in which case the
    incremental values are accumulated across lag to build a cumulative
    triangle. The two conventions give different triangles, so choose the one
    that matches your input.
    """
    validate_columns(df, [origin_col, valuation_col, amount_col])
    temp = df.copy()
    temp[index_name] = pd.to_datetime(temp[origin_col]).dt.to_period("M")
    temp[lag_name] = lag_months(temp[origin_col], temp[valuation_col])
    grouped = temp.groupby([index_name, lag_name], dropna=False)[amount_col].sum().reset_index()
    triangle = grouped.pivot(index=index_name, columns=lag_name, values=amount_col).sort_index(axis=1)
    if cumulative:
        triangle = triangle.cumsum(axis=1)
    return triangle


def completed_experience(
    df: pd.DataFrame,
    *,
    component_factor_map: dict[str, str],
    revenue_cols: str | Iterable[str],
    groupby: str | Iterable[str] | None = None,
    exposure_cols: str | Iterable[str] | None = None,
    additional_expense_cols: str | Iterable[str] | None = None,
    method: str = "divide",
    ratio_col: str | None = None,
    profile: str | None = None,
    validate_factors: bool = True,
) -> pd.DataFrame:
    """Complete claim components, then summarize experience on a completed basis.

    Convenience for the common completion -> experience chain. Each
    ``paid_col -> factor_col`` pair in ``component_factor_map`` is completed; the
    resulting ``*_completed`` columns become the expense numerator, together with
    any ``additional_expense_cols`` (e.g. rebates, non-fee-for-service expense)
    that are summed as-is. The factor columns must already be present on ``df``
    (join your completion-factor table on first). Returns a grouped experience
    summary whose loss ratio is on a completed basis.
    """
    from actuarialpy.experience import summarize_experience

    completed = complete_claim_components(
        df, component_factor_map, method=method, validate_factors=validate_factors
    )
    completed_cols = [f"{paid_col}_completed" for paid_col in component_factor_map]
    expense_cols = completed_cols + as_list(additional_expense_cols)
    return summarize_experience(
        completed,
        groupby=groupby,
        expense_cols=expense_cols,
        revenue_cols=revenue_cols,
        exposure_cols=exposure_cols,
        ratio_col=ratio_col,
        profile=profile,
    )
