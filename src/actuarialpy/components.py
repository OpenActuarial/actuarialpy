"""Component/category summaries and driver analysis."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.metrics import per_exposure, safe_divide
from actuarialpy.trend import _comparison_masks


def _per_exposure_name(component: str, exposure_col: str) -> str:
    if exposure_col == "member_months":
        return f"{component}_pmpm"
    if exposure_col == "subscriber_months":
        return f"{component}_pspm"
    if exposure_col == "employee_months":
        return f"{component}_pepm"
    return f"{component}_per_{exposure_col}"


def summarize_components(
    df: pd.DataFrame,
    *,
    groupby: str | Iterable[str] | None = None,
    component_cols: str | Iterable[str],
    exposure_col: str | None = None,
    total_col: str = "total_expense",
    include_shares: bool = True,
) -> pd.DataFrame:
    """Summarize component/category amounts, per-exposure values, and shares."""
    groups = as_list(groupby)
    components = as_list(component_cols)
    required = groups + components + ([exposure_col] if exposure_col else [])
    validate_columns(df, required)

    amount_cols = components + ([exposure_col] if exposure_col else [])
    if groups:
        summary = df[groups + amount_cols].groupby(groups, dropna=False, as_index=False).sum(numeric_only=True)
    else:
        summary = pd.DataFrame({col: [df[col].sum()] for col in amount_cols})

    summary[total_col] = summary[components].sum(axis=1)
    if exposure_col:
        for component in components:
            summary[_per_exposure_name(component, exposure_col)] = per_exposure(summary[component], summary[exposure_col])
        summary[_per_exposure_name(total_col, exposure_col)] = per_exposure(summary[total_col], summary[exposure_col])
    if include_shares:
        for component in components:
            summary[f"{component}_share"] = safe_divide(summary[component], summary[total_col])
    return summary


def component_driver_analysis(
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
    component_cols: str | Iterable[str],
    exposure_col: str | None = None,
    groupby: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Explain component drivers of change between two periods.

    The primary comparison is based on component totals, or component amount per
    exposure when ``exposure_col`` is supplied. The API matches ``trend_summary``
    and supports period-column, date-range, or explicit-filter comparisons.
    """
    groups = as_list(groupby)
    components = as_list(component_cols)
    required = groups + components + ([exposure_col] if exposure_col else [])
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

    prior_df = df.loc[prior_filter]
    current_df = df.loc[current_filter]

    prior_sum = summarize_components(
        prior_df,
        groupby=groups,
        component_cols=components,
        exposure_col=exposure_col,
        include_shares=False,
    )
    current_sum = summarize_components(
        current_df,
        groupby=groups,
        component_cols=components,
        exposure_col=exposure_col,
        include_shares=False,
    )

    if groups:
        merged = prior_sum.merge(current_sum, on=groups, how="outer", suffixes=("_prior", "_current"))
    else:
        merged = pd.concat([prior_sum.add_suffix("_prior"), current_sum.add_suffix("_current")], axis=1)

    rows = []
    for _, row in merged.iterrows():
        key_data = {g: row[g] for g in groups} if groups else {}
        changes = {}
        total_change = 0
        for comp in components:
            metric = _per_exposure_name(comp, exposure_col) if exposure_col else comp
            prior_val = row.get(f"{metric}_prior", 0)
            current_val = row.get(f"{metric}_current", 0)
            prior_val = 0 if pd.isna(prior_val) else prior_val
            current_val = 0 if pd.isna(current_val) else current_val
            changes[comp] = current_val - prior_val
            total_change += changes[comp]

        for comp in components:
            metric = _per_exposure_name(comp, exposure_col) if exposure_col else comp
            prior_val = row.get(f"{metric}_prior", 0)
            current_val = row.get(f"{metric}_current", 0)
            prior_val = 0 if pd.isna(prior_val) else prior_val
            current_val = 0 if pd.isna(current_val) else current_val
            period_data = {}
            if mode == "period":
                period_data = {"prior_period": prior_period, "current_period": current_period}
            elif mode == "date":
                period_data = {
                    "prior_start": pd.to_datetime(prior_start),
                    "prior_end": pd.to_datetime(prior_end),
                    "current_start": pd.to_datetime(current_start),
                    "current_end": pd.to_datetime(current_end),
                }
            rows.append(
                {
                    **key_data,
                    **period_data,
                    "component": comp,
                    "prior": prior_val,
                    "current": current_val,
                    "change": current_val - prior_val,
                    "trend": safe_divide(current_val, prior_val) - 1,
                    "contribution_to_change": safe_divide(changes[comp], total_change),
                }
            )
    return pd.DataFrame(rows)


def component_trend(*args, **kwargs) -> pd.DataFrame:
    """Alias for ``component_driver_analysis``.

    The preferred name is ``component_driver_analysis`` because the function
    explains drivers of total component change, not just component-specific trend.
    """
    return component_driver_analysis(*args, **kwargs)
