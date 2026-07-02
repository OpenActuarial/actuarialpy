"""Core actuarial metric primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _is_pandas(obj: Any) -> bool:
    return isinstance(obj, (pd.Series, pd.DataFrame))


def safe_divide(numerator: Any, denominator: Any, *, fill_value: float = np.nan) -> Any:
    """Safely divide numerator by denominator.

    The return type mirrors the input: scalars return scalars, array-likes
    return NumPy arrays, and pandas inputs return pandas objects with their
    index (and name) preserved -- so results can be assigned straight back
    onto the source DataFrame. Zero denominators are returned as
    ``fill_value``.
    """
    if isinstance(numerator, (int, float, np.number)) and isinstance(denominator, (int, float, np.number)):
        return fill_value if denominator == 0 else numerator / denominator

    if _is_pandas(numerator) or _is_pandas(denominator):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = numerator / denominator
        if _is_pandas(denominator):
            zero_mask = denominator.eq(0).reindex_like(out).fillna(False)
        else:
            zero_mask = np.broadcast_to(np.asarray(denominator) == 0, np.shape(out))
        return out.mask(zero_mask, fill_value)

    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    numerator_b, denominator_b = np.broadcast_arrays(numerator_arr, denominator_arr)
    return np.divide(
        numerator_b,
        denominator_b,
        out=np.full(numerator_b.shape, fill_value, dtype=float),
        where=denominator_b != 0,
    )


def ratio(numerator: Any, denominator: Any) -> Any:
    """Calculate a generic ratio as numerator divided by denominator."""
    return safe_divide(numerator, denominator)


def loss_ratio(losses_or_expenses: Any, revenue: Any) -> Any:
    """Calculate a loss ratio: losses or expenses divided by revenue."""
    return ratio(losses_or_expenses, revenue)


def medical_loss_ratio(claims: Any, premium: Any) -> Any:
    """Calculate a medical loss ratio: claims divided by premium."""
    return loss_ratio(claims, premium)


def expense_ratio(expenses: Any, revenue: Any) -> Any:
    """Calculate an expense ratio: expenses divided by revenue."""
    return ratio(expenses, revenue)


def combined_ratio(losses: Any, expenses: Any, revenue: Any) -> Any:
    """Calculate combined ratio: (losses + expenses) divided by revenue."""
    if _is_pandas(losses) or _is_pandas(expenses):
        return ratio(losses + expenses, revenue)
    return ratio(np.asarray(losses) + np.asarray(expenses), revenue)


def actual_to_expected(actual: Any, expected: Any) -> Any:
    """Calculate actual-to-expected: actual divided by expected."""
    return ratio(actual, expected)


def per_exposure(amount: Any, exposure: Any) -> Any:
    """Calculate amount per exposure unit."""
    return ratio(amount, exposure)


def pmpm(amount: Any, member_months: Any) -> Any:
    """Calculate amount per member month."""
    return per_exposure(amount, member_months)


def pspm(amount: Any, subscriber_months: Any) -> Any:
    """Calculate amount per subscriber month."""
    return per_exposure(amount, subscriber_months)


def pepm(amount: Any, employee_months: Any) -> Any:
    """Calculate amount per employee month."""
    return per_exposure(amount, employee_months)


def frequency(claim_count: Any, exposure: Any) -> Any:
    """Calculate claim frequency: claim count divided by exposure."""
    return ratio(claim_count, exposure)


def severity(losses: Any, claim_count: Any) -> Any:
    """Calculate severity: losses divided by claim count."""
    return ratio(losses, claim_count)


def pure_premium(losses: Any, exposure: Any) -> Any:
    """Calculate pure premium: losses divided by exposure."""
    return per_exposure(losses, exposure)


def utilization_per_1000(claim_count: Any, exposure: Any, *, annualization: float = 12) -> Any:
    """Annualized utilization per 1,000 members.

    Returns ``claim_count / exposure * annualization * 1000``. With monthly member
    months as ``exposure`` the default ``annualization=12`` yields services (admits,
    visits, scripts, ...) per 1,000 members per year. If ``exposure`` is already in
    member-years, pass ``annualization=1``.
    """
    return ratio(claim_count, exposure) * annualization * 1000


def required_revenue(expense: Any, target_ratio: Any) -> Any:
    """Revenue needed for an expense amount to hit a target ratio."""
    return safe_divide(expense, target_ratio)


def indicated_change(required: Any, current: Any) -> Any:
    """Indicated change from current to required amount."""
    return safe_divide(required, current) - 1


def permissible_loss_ratio(expense_ratio: Any, profit_provision: Any = 0.0) -> Any:
    """Permissible (target / break-even) loss ratio.

    ``PLR = 1 - expense_ratio - profit_provision`` where both loadings are
    expressed as a fraction of premium. Also called the zero-margin or target
    loss ratio: the loss ratio at which premium exactly covers losses, expenses,
    and the profit/contingency provision. Works element-wise on scalars or
    Series. (Shops that load fixed expenses on a loss basis instead use
    ``(1 - V - Q) / (1 + G)``; this implements the premium-basis form.)
    """
    return 1.0 - expense_ratio - profit_provision
