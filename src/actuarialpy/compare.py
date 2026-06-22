"""Comparison and variance primitives."""

from __future__ import annotations

from typing import Any

from actuarialpy.metrics import safe_divide


def absolute_change(current: Any, prior: Any) -> Any:
    """Calculate current minus prior."""
    return current - prior


def percent_change(current: Any, prior: Any) -> Any:
    """Calculate percent change: current / prior - 1."""
    return safe_divide(current, prior) - 1


def basis_point_change(current_ratio: Any, prior_ratio: Any) -> Any:
    """Calculate basis point change between two decimal ratios."""
    return (current_ratio - prior_ratio) * 10_000


def variance(actual: Any, expected: Any) -> Any:
    """Calculate actual minus expected."""
    return actual - expected


def variance_pct(actual: Any, expected: Any) -> Any:
    """Calculate variance as percent of expected: actual / expected - 1."""
    return safe_divide(actual, expected) - 1
