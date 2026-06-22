"""Completion-factor, completed-claims, and IBNR tools."""

from __future__ import annotations

import pandas as pd

from actuarialpy.columns import validate_columns
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
) -> pd.DataFrame:
    """Create a simple cumulative claims triangle by origin period and valuation lag."""
    validate_columns(df, [origin_col, valuation_col, amount_col])
    temp = df.copy()
    temp[index_name] = pd.to_datetime(temp[origin_col]).dt.to_period("M")
    temp[lag_name] = lag_months(temp[origin_col], temp[valuation_col])
    grouped = temp.groupby([index_name, lag_name], dropna=False)[amount_col].sum().reset_index()
    return grouped.pivot(index=index_name, columns=lag_name, values=amount_col).sort_index(axis=1)
