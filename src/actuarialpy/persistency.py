"""Persistency: renewal probability as a function of the rate action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Persistency:
    """Renewal (persistency) model: the probability a policy renews given the rate
    change put through.

    The probability declines linearly with the rate increase and is clipped to
    ``[floor, cap]``::

        P(renew) = clip(base_retention - rate_elasticity * rate_change, floor, cap)

    ``base_retention`` is the renewal probability at a zero rate change and
    ``rate_elasticity`` is the drop in that probability per unit of rate increase
    (the lapse-vs-rate-increase slope). A larger proposed increase lowers
    ``P(renew)``. Fit both from renewal history with :func:`fit_persistency`.
    """

    base_retention: float
    rate_elasticity: float = 0.0
    floor: float = 0.0
    cap: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.floor <= self.cap <= 1.0:
            raise ValueError("require 0 <= floor <= cap <= 1.")

    def probability(self, rate_change=0.0):
        """Renewal probability for a rate change (scalar, Series, or array-like)."""
        if isinstance(rate_change, pd.Series):
            p = self.base_retention - self.rate_elasticity * rate_change
            return p.clip(lower=self.floor, upper=self.cap)
        rc = np.asarray(rate_change, dtype=float)
        p = np.clip(self.base_retention - self.rate_elasticity * rc, self.floor, self.cap)
        return float(p) if p.ndim == 0 else p


def fit_persistency(rate_changes, renewed, *, floor: float = 0.0, cap: float = 1.0) -> Persistency:
    """Fit a :class:`Persistency` model from renewal history by least squares.

    ``renewed`` is the renewal outcome per observation -- a 0/1 indicator, or a
    per-cell retention rate -- regressed on the rate change::

        renewed ~= base_retention - rate_elasticity * rate_change

    The fitted intercept is ``base_retention`` and the negated slope is
    ``rate_elasticity``. Requires at least two observations.
    """
    x = np.asarray(rate_changes, dtype=float)
    y = np.asarray(renewed, dtype=float)
    if x.shape != y.shape:
        raise ValueError("rate_changes and renewed must have the same shape.")
    if x.size < 2:
        raise ValueError("need at least two observations to fit persistency.")
    intercept, slope = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)[0]
    return Persistency(
        base_retention=float(intercept), rate_elasticity=float(-slope), floor=floor, cap=cap
    )
