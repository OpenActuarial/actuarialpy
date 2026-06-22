"""Grouped actuarial experience summaries."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from actuarialpy.columns import as_list, sum_columns, validate_columns
from actuarialpy.metrics import loss_ratio, per_exposure
from actuarialpy.profiles import apply_profile_labels, get_profile_defaults

_ID_LIKE_EXPOSURE_NAMES = {"member_id", "subscriber_id", "group_id", "employee_id", "policy_id", "claim_id"}


def _validate_exposures(exposures: list[str]) -> None:
    bad = [col for col in exposures if col.lower() in _ID_LIKE_EXPOSURE_NAMES or col.lower().endswith("_id")]
    if bad:
        raise ValueError(
            "Exposure columns must be numeric exposure measures, not identifiers. "
            f"Invalid exposure column(s): {bad}. For member-level monthly data, create a "
            "member_months column equal to 1 and use exposure_cols='member_months'."
        )


def _per_exposure_column_names(total_expense_name: str, total_revenue_name: str, exposure: str) -> tuple[str, str]:
    mapping = {
        "member_months": ("expense_pmpm", "revenue_pmpm"),
        "subscriber_months": ("expense_pspm", "revenue_pspm"),
        "employee_months": ("expense_pepm", "revenue_pepm"),
    }
    return mapping.get(exposure, (f"{total_expense_name}_per_{exposure}", f"{total_revenue_name}_per_{exposure}"))


def summarize_experience(
    df: pd.DataFrame,
    *,
    groupby: str | Iterable[str] | None = None,
    expense_cols: str | Iterable[str],
    revenue_cols: str | Iterable[str],
    exposure_cols: str | Iterable[str] | None = None,
    ratio_col: str | None = None,
    ratio_name: str | None = None,
    total_expense_name: str = "total_expense",
    total_revenue_name: str = "total_revenue",
    profile: str | None = None,
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize experience by grouping columns.

    Amounts and exposures are aggregated first. Ratios and per-exposure metrics
    are calculated after aggregation, which avoids averaging row-level ratios.

    ``profile`` only supplies light defaults such as ``ratio_col='mlr'`` for
    health. It does not rename total expense or total revenue.
    """
    groups = as_list(groupby)
    expenses = as_list(expense_cols)
    revenues = as_list(revenue_cols)
    exposures = as_list(exposure_cols)
    _validate_exposures(exposures)
    validate_columns(df, groups + expenses + revenues + exposures)

    if ratio_col is not None and ratio_name is not None:
        raise ValueError("Specify only one of ratio_col or ratio_name.")
    if ratio_name is not None:
        ratio_col = ratio_name
    if ratio_col is None:
        ratio_col = get_profile_defaults(profile).get("ratio_col", "expense_revenue_ratio")

    amount_cols = list(dict.fromkeys(expenses + revenues + exposures))
    if groups:
        summary = df[groups + amount_cols].groupby(groups, dropna=False, as_index=False).sum(numeric_only=True)
    else:
        summary = pd.DataFrame({col: [df[col].sum()] for col in amount_cols})

    summary[total_expense_name] = sum_columns(summary, expenses)
    summary[total_revenue_name] = sum_columns(summary, revenues)
    summary[ratio_col] = loss_ratio(summary[total_expense_name], summary[total_revenue_name])

    for exposure in exposures:
        expense_per, revenue_per = _per_exposure_column_names(total_expense_name, total_revenue_name, exposure)
        summary[expense_per] = per_exposure(summary[total_expense_name], summary[exposure])
        summary[revenue_per] = per_exposure(summary[total_revenue_name], summary[exposure])

    return apply_profile_labels(summary, profile=profile, labels=labels)


def summarize_views(
    df: pd.DataFrame,
    *,
    views: dict[str, str | Iterable[str] | None],
    expense_cols: str | Iterable[str],
    revenue_cols: str | Iterable[str],
    exposure_cols: str | Iterable[str] | None = None,
    ratio_col: str | None = None,
    ratio_name: str | None = None,
    total_expense_name: str = "total_expense",
    total_revenue_name: str = "total_revenue",
    profile: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Create multiple experience summary views from the same input data."""
    return {
        name: summarize_experience(
            df,
            groupby=groupby,
            expense_cols=expense_cols,
            revenue_cols=revenue_cols,
            exposure_cols=exposure_cols,
            ratio_col=ratio_col,
            ratio_name=ratio_name,
            total_expense_name=total_expense_name,
            total_revenue_name=total_revenue_name,
            profile=profile,
        )
        for name, groupby in views.items()
    }


def status_summary(
    df: pd.DataFrame,
    *,
    status_col: str,
    entity_col: str | None = None,
    expense_cols: str | Iterable[str],
    revenue_cols: str | Iterable[str],
    exposure_cols: str | Iterable[str] | None = None,
    profile: str | None = None,
) -> pd.DataFrame:
    """Summarize experience by status, optionally adding entity counts."""
    validate_columns(df, [status_col] + ([entity_col] if entity_col else []))
    summary = summarize_experience(
        df,
        groupby=status_col,
        expense_cols=expense_cols,
        revenue_cols=revenue_cols,
        exposure_cols=exposure_cols,
        profile=profile,
    )
    if entity_col:
        counts = df.groupby(status_col, dropna=False)[entity_col].nunique().reset_index(name="entity_count")
        summary = counts.merge(summary, on=status_col, how="right")
    return summary
