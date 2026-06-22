"""Rolling-period actuarial summaries."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.experience import summarize_experience
from actuarialpy.metrics import loss_ratio, per_exposure


def _per_exposure_column_names(exposure: str) -> tuple[str, str]:
    mapping = {
        "member_months": ("expense_pmpm", "revenue_pmpm"),
        "subscriber_months": ("expense_pspm", "revenue_pspm"),
        "employee_months": ("expense_pepm", "revenue_pepm"),
    }
    return mapping.get(exposure, (f"total_expense_per_{exposure}", f"total_revenue_per_{exposure}"))


def rolling_summary(
    df: pd.DataFrame,
    *,
    date_col: str,
    window: int = 12,
    groupby: str | Iterable[str] | None = None,
    expense_cols: str | Iterable[str],
    revenue_cols: str | Iterable[str],
    exposure_cols: str | Iterable[str] | None = None,
    min_periods: int | None = None,
    drop_incomplete: bool = True,
    ratio_col: str = "mlr",
) -> pd.DataFrame:
    """Calculate rolling sums and ratios by period and optional grouping.

    The output includes ``period_start`` and ``period_end``. By default only
    complete rolling windows are returned; for a 12-month window, the first
    output row appears after 12 months of data are available.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    groups = as_list(groupby)
    expenses = as_list(expense_cols)
    revenues = as_list(revenue_cols)
    exposures = as_list(exposure_cols)
    validate_columns(df, groups + [date_col] + expenses + revenues + exposures)
    min_periods = window if min_periods is None else min_periods

    base = summarize_experience(
        df,
        groupby=groups + [date_col],
        expense_cols=expenses,
        revenue_cols=revenues,
        exposure_cols=exposures,
        ratio_col="period_ratio",
    ).sort_values(groups + [date_col] if groups else [date_col])

    amount_cols = ["total_expense", "total_revenue"] + exposures
    pieces = []
    iterator = base.groupby(groups, dropna=False, sort=False) if groups else [((), base)]

    for _, part in iterator:
        part = part.sort_values(date_col).copy().reset_index(drop=True)
        rolled = part[amount_cols].rolling(window=window, min_periods=min_periods).sum()
        months_available = part["total_expense"].rolling(window=window, min_periods=1).count().astype(int)

        out = part[groups].copy() if groups else pd.DataFrame(index=part.index)
        dates = pd.to_datetime(part[date_col])
        starts = []
        for i in range(len(part)):
            start_i = max(0, i - window + 1)
            starts.append(dates.iloc[start_i])
        out["period_start"] = starts
        out["period_end"] = dates
        out["months_available"] = months_available.values

        for col in amount_cols:
            out[col] = rolled[col].values
        out[ratio_col] = loss_ratio(out["total_expense"], out["total_revenue"])
        for exposure in exposures:
            expense_per, revenue_per = _per_exposure_column_names(exposure)
            out[expense_per] = per_exposure(out["total_expense"], out[exposure])
            out[revenue_per] = per_exposure(out["total_revenue"], out[exposure])

        if drop_incomplete:
            out = out[out["months_available"] >= window].copy()
        pieces.append(out)

    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    if drop_incomplete:
        result = result.drop(columns=["months_available"])
    return result
