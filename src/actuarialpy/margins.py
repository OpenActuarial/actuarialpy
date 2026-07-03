"""Margin primitives.

Underwriting margin is premium net of losses and expense loadings (retention,
commission, overhead, profit provision). Generic across lines: in health this is
the dollars left after medical expense and administrative loadings; in P&C it is
premium less losses and expenses.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from actuarialpy.columns import sum_columns, validate_columns
from actuarialpy.metrics import per_exposure, safe_divide


def margin(premium: Any, expenses: Any) -> Any:
    """Margin = premium - expenses, element-wise.

    ``expenses`` should already be the total of losses plus any loadings.
    """
    return premium - expenses


def margin_ratio(margin_amount: Any, premium: Any) -> Any:
    """Margin as a fraction of premium = margin / premium."""
    return safe_divide(margin_amount, premium)


def add_margin(
    df: pd.DataFrame,
    *,
    premium_col: str,
    expense_cols: str | Iterable[str],
    out_col: str = "margin",
    ratio_col: str | None = None,
    exposure_col: str | None = None,
    per_exposure_col: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Add an underwriting-margin column (premium minus summed expense columns).

    ``expense_cols`` is summed row-wise and may mix losses and loadings (e.g.
    claims, retention, commission, allocated overhead). Optionally also
    add the margin ratio (``ratio_col``) and a per-exposure margin
    (``per_exposure_col``, requires ``exposure_col``) such as margin per exposure unit.
    """
    validate_columns(df, [premium_col])
    result = df.copy() if copy else df
    total_expense = sum_columns(result, expense_cols)
    result[out_col] = result[premium_col] - total_expense
    if ratio_col is not None:
        result[ratio_col] = safe_divide(result[out_col], result[premium_col])
    if per_exposure_col is not None:
        if exposure_col is None:
            raise ValueError("exposure_col is required when per_exposure_col is set.")
        validate_columns(result, [exposure_col])
        result[per_exposure_col] = per_exposure(result[out_col], result[exposure_col])
    return result
