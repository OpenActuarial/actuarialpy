"""Trend and projection primitives."""

from __future__ import annotations

from typing import Any

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.compare import relative_change
from actuarialpy.metrics import safe_divide


def period_change(current: Any, prior: Any) -> Any:
    """Calculate period-over-period change: current / prior - 1."""
    return relative_change(current, prior)


def annualized_trend(current: Any, prior: Any, months_between: float) -> Any:
    """Annualize change between two values separated by a number of months."""
    if months_between <= 0:
        raise ValueError("months_between must be positive")
    return safe_divide(current, prior) ** (12 / months_between) - 1


def trend_factor(annual_trend: Any, months: float) -> Any:
    """Convert an annual trend rate into a trend factor over a number of months."""
    return (1 + annual_trend) ** (months / 12)


def project_forward(value: Any, annual_trend: Any, months: float) -> Any:
    """Project a value forward using an annual trend rate."""
    return value * trend_factor(annual_trend, months)


def midpoint_trend_factor(base_midpoint, projection_midpoint, annual_trend: Any) -> Any:
    """Trend factor between base and projection midpoints."""
    base = pd.to_datetime(base_midpoint)
    projection = pd.to_datetime(projection_midpoint)
    months = (projection.year - base.year) * 12 + (projection.month - base.month)
    return trend_factor(annual_trend, months)


def trend_summary(
    df: pd.DataFrame,
    *,
    period_col: str | None = None,
    prior_period=None,
    current_period=None,
    groupby=None,
    amount_col: str,
    exposure_col: str | None = None,
    prior_filter=None,
    current_filter=None,
    prior_label: str = "prior",
    current_label: str = "current",
) -> pd.DataFrame:
    """Summarize current vs prior trend by optional grouping.

    Preferred API:
    ``period_col='year', prior_period=2025, current_period=2026``.

    Advanced API:
    pass boolean ``prior_filter`` and ``current_filter`` masks instead.
    """
    groups = as_list(groupby)
    sum_cols = groups + [amount_col] + ([exposure_col] if exposure_col else [])
    to_validate = sum_cols + ([period_col] if period_col is not None else [])
    validate_columns(df, to_validate)

    period_args_supplied = period_col is not None or prior_period is not None or current_period is not None
    filter_args_supplied = prior_filter is not None or current_filter is not None
    if period_args_supplied and filter_args_supplied:
        raise ValueError("Use either period_col/prior_period/current_period or prior_filter/current_filter, not both.")
    if period_args_supplied:
        if period_col is None or prior_period is None or current_period is None:
            raise ValueError("period_col, prior_period, and current_period must all be supplied together.")
        prior_filter = df[period_col] == prior_period
        current_filter = df[period_col] == current_period
    elif not filter_args_supplied:
        raise ValueError("Provide period_col/prior_period/current_period or prior_filter/current_filter.")
    elif prior_filter is None or current_filter is None:
        raise ValueError("prior_filter and current_filter must be supplied together.")

    def summarize(mask, label):
        temp = df.loc[mask, sum_cols].copy()
        if groups:
            out = temp.groupby(groups, dropna=False, as_index=False).sum(numeric_only=True)
        else:
            out = pd.DataFrame({amount_col: [temp[amount_col].sum()]})
            if exposure_col:
                out[exposure_col] = temp[exposure_col].sum()
        out = out.rename(columns={amount_col: f"{label}_{amount_col}"})
        if exposure_col:
            out = out.rename(columns={exposure_col: f"{label}_{exposure_col}"})
            out[f"{label}_{amount_col}_per_{exposure_col}"] = safe_divide(
                out[f"{label}_{amount_col}"], out[f"{label}_{exposure_col}"]
            )
        return out

    prior = summarize(prior_filter, prior_label)
    current = summarize(current_filter, current_label)
    out = prior.merge(current, on=groups, how="outer") if groups else pd.concat([prior, current], axis=1)
    prior_metric = f"{prior_label}_{amount_col}_per_{exposure_col}" if exposure_col else f"{prior_label}_{amount_col}"
    current_metric = f"{current_label}_{amount_col}_per_{exposure_col}" if exposure_col else f"{current_label}_{amount_col}"
    out["trend"] = period_change(out[current_metric], out[prior_metric])
    if period_args_supplied:
        out.insert(len(groups), "prior_period", prior_period)
        out.insert(len(groups) + 1, "current_period", current_period)
    return out
