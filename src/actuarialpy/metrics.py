"""Core actuarial metric primitives."""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np


def safe_divide(numerator: Any, denominator: Any, *, fill_value: float = np.nan) -> Any:
    """Safely divide numerator by denominator.

    Scalars return scalars. Array-like inputs return NumPy arrays. Zero denominators
    are returned as ``fill_value``.
    """
    if isinstance(numerator, Number) and isinstance(denominator, Number):
        return fill_value if denominator == 0 else numerator / denominator

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


def required_revenue(expense: Any, target_ratio: Any) -> Any:
    """Revenue needed for an expense amount to hit a target ratio."""
    return safe_divide(expense, target_ratio)


def indicated_change(required: Any, current: Any) -> Any:
    """Indicated change from current to required amount."""
    return safe_divide(required, current) - 1
