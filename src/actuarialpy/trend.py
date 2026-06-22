"""Trend and projection primitives."""

from __future__ import annotations

from typing import Any

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.metrics import safe_divide


def period_change(current: Any, prior: Any) -> Any:
    """Calculate period-over-period change: current / prior - 1."""
    return safe_divide(current, prior) - 1


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



def _date_range_mask(df: pd.DataFrame, date_col: str, start, end) -> pd.Series:
    dates = pd.to_datetime(df[date_col])
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    if end_date < start_date:
        raise ValueError("range end must be greater than or equal to range start")
    return (dates >= start_date) & (dates <= end_date)


def _comparison_masks(
    df: pd.DataFrame,
    *,
    period_col: str | None = None,
    prior_period=None,
    current_period=None,
    date_col: str | None = None,
    prior_start=None,
    prior_end=None,
    current_start=None,
    current_end=None,
    prior_filter=None,
    current_filter=None,
) -> tuple[pd.Series, pd.Series, str]:
    period_args_supplied = period_col is not None or prior_period is not None or current_period is not None
    date_args_supplied = (
        date_col is not None
        or prior_start is not None
        or prior_end is not None
        or current_start is not None
        or current_end is not None
    )
    filter_args_supplied = prior_filter is not None or current_filter is not None
    modes = sum([period_args_supplied, date_args_supplied, filter_args_supplied])
    if modes != 1:
        raise ValueError(
            "Use exactly one comparison mode: period_col/prior_period/current_period, "
            "date_col with prior/current ranges, or prior_filter/current_filter."
        )

    if period_args_supplied:
        if period_col is None or prior_period is None or current_period is None:
            raise ValueError("period_col, prior_period, and current_period must all be supplied together.")
        return df[period_col] == prior_period, df[period_col] == current_period, "period"

    if date_args_supplied:
        if None in (date_col, prior_start, prior_end, current_start, current_end):
            raise ValueError(
                "date_col, prior_start, prior_end, current_start, and current_end must all be supplied together."
            )
        return (
            _date_range_mask(df, date_col, prior_start, prior_end),
            _date_range_mask(df, date_col, current_start, current_end),
            "date",
        )

    if prior_filter is None or current_filter is None:
        raise ValueError("prior_filter and current_filter must be supplied together.")
    return prior_filter, current_filter, "filter"

def trend_summary(
    df: pd.DataFrame,
    *,
    period_col: str | None = None,
    prior_period=None,
    current_period=None,
    date_col: str | None = None,
    prior_start=None,
    prior_end=None,
    current_start=None,
    current_end=None,
    groupby=None,
    amount_col: str,
    exposure_col: str | None = None,
    prior_filter=None,
    current_filter=None,
    prior_label: str = "prior",
    current_label: str = "current",
) -> pd.DataFrame:
    """Summarize current vs prior trend by optional grouping.

    Supported comparison modes:
    - ``period_col='year', prior_period=2025, current_period=2026``
    - ``date_col='incurred_date'`` with prior/current start and end dates
    - explicit boolean ``prior_filter`` and ``current_filter`` masks
    """
    groups = as_list(groupby)
    required = groups + [amount_col] + ([exposure_col] if exposure_col else [])
    if period_col is not None:
        required.append(period_col)
    if date_col is not None:
        required.append(date_col)
    validate_columns(df, required)

    prior_filter, current_filter, mode = _comparison_masks(
        df,
        period_col=period_col,
        prior_period=prior_period,
        current_period=current_period,
        date_col=date_col,
        prior_start=prior_start,
        prior_end=prior_end,
        current_start=current_start,
        current_end=current_end,
        prior_filter=prior_filter,
        current_filter=current_filter,
    )

    def summarize(mask, label):
        # Aggregate only grouping, amount, and exposure columns. The comparison
        # column (for example, ``year``) is used only to select records and must
        # not leak into the final output as a summed numeric column such as
        # ``year_x`` / ``year_y``.
        summary_cols = groups + [amount_col] + ([exposure_col] if exposure_col else [])
        temp = df.loc[mask, summary_cols].copy()
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
    if mode == "period":
        out.insert(len(groups), "prior_period", prior_period)
        out.insert(len(groups) + 1, "current_period", current_period)
    elif mode == "date":
        out.insert(len(groups), "prior_start", pd.to_datetime(prior_start))
        out.insert(len(groups) + 1, "prior_end", pd.to_datetime(prior_end))
        out.insert(len(groups) + 2, "current_start", pd.to_datetime(current_start))
        out.insert(len(groups) + 3, "current_end", pd.to_datetime(current_end))
    return out
