"""Shared numeric-input validation.

Small, dependency-free guards used across the calculation modules to reject
non-finite (``NaN``/``inf``) and out-of-domain inputs *before* they reach a
logarithm, a division, or a regression -- where they would otherwise propagate
silently as ``NaN``/``inf`` results rather than raising.

The important subtlety these guard against: a bare ``np.any(x <= 0)`` does *not*
reject ``NaN``, because every comparison with ``NaN`` is ``False``. A rate,
weight, or exposure formed from missing data therefore slips past a naive
non-positivity check and contaminates the downstream estimate. Every validator
here checks finiteness first, so missing data raises instead of being hidden.

These are internal helpers (not part of the public API); import them where a
public entry point accepts numeric data. Each raises ``ValueError`` with a
message naming the offending quantity and leaves its input unchanged.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "validate_finite",
    "validate_positive",
    "validate_nonnegative",
    "validate_probability",
    "validate_weights",
    "validate_quantile",
]


def _as_float_array(x: object) -> np.ndarray:
    """Coerce to a float ndarray for uniform checking (scalars become 0-d)."""
    return np.asarray(x, dtype="float64")


def validate_finite(x: object, name: str = "value") -> None:
    """Raise ``ValueError`` if ``x`` contains any ``NaN`` or infinite entry."""
    arr = _as_float_array(x)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")


def validate_positive(x: object, name: str = "value") -> None:
    """Raise if ``x`` is not finite and strictly positive (``> 0``) throughout."""
    arr = _as_float_array(x)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive (> 0)")


def validate_nonnegative(x: object, name: str = "value") -> None:
    """Raise if ``x`` is not finite and non-negative (``>= 0``) throughout."""
    arr = _as_float_array(x)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative (>= 0)")


def validate_probability(x: object, name: str = "probability", *, closed: str = "both") -> None:
    """Raise if ``x`` is not a finite probability in ``[0, 1]``.

    ``closed`` controls which endpoints are admissible: ``"both"`` for
    ``[0, 1]`` (the default), ``"neither"`` for the open ``(0, 1)``,
    ``"left"`` for ``[0, 1)``, or ``"right"`` for ``(0, 1]``.
    """
    arr = _as_float_array(x)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    lo_ok = arr >= 0.0 if closed in ("both", "left") else arr > 0.0
    hi_ok = arr <= 1.0 if closed in ("both", "right") else arr < 1.0
    if not np.all(lo_ok & hi_ok):
        bounds = {"both": "[0, 1]", "neither": "(0, 1)", "left": "[0, 1)", "right": "(0, 1]"}[closed]
        raise ValueError(f"{name} must lie in {bounds}")


def validate_quantile(q: object, name: str = "quantile") -> None:
    """Raise if ``q`` is not a finite quantile level in the open interval ``(0, 1)``."""
    validate_probability(q, name, closed="neither")


def validate_weights(w: object, name: str = "weights") -> None:
    """Raise if ``w`` is not a finite, non-negative weight vector with positive sum.

    Weights that are all zero (or, with finiteness, sum to zero) cannot form a
    weighted average, so this rejects them rather than dividing by zero.
    """
    arr = _as_float_array(w)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative (>= 0)")
    if arr.sum() <= 0.0:
        raise ValueError(f"{name} must not sum to zero")
