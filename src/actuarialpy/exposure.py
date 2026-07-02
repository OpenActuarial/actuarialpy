r"""Exposure measurement and age bases for experience studies.

Any actual-to-expected study has to answer "how old is this life, and how much
exposure does this record contribute," and the answer depends on the age basis
and the study window. This module provides age on the standard bases (exact,
age last birthday, age nearest birthday) and exposure-year calculation within a
study period.

This complements :mod:`actuarialpy.lifecycle`, which measures member-month
exposure and in-force status for health blocks; here the focus is the
exposure-year / age-basis machinery used in mortality and lapse studies.
"""
from __future__ import annotations

import pandas as pd

from actuarialpy.financial import year_fraction


def age(date_of_birth, as_of, basis: str = "exact") -> float:
    """Age of a life at a date on a given basis.

    Parameters
    ----------
    date_of_birth, as_of : date-like
        Date of birth and the valuation date.
    basis : {"exact", "last", "nearest"}
        ``"exact"`` returns the fractional age; ``"last"`` is age last birthday
        (completed years, ALB); ``"nearest"`` is age nearest birthday (ANB).

    Returns
    -------
    float or int
        Fractional age for ``"exact"``; an integer age for ``"last"`` and
        ``"nearest"``.
    """
    dob = pd.Timestamp(date_of_birth)
    at = pd.Timestamp(as_of)
    if at < dob:
        raise ValueError("as_of must not precede date_of_birth.")

    completed = at.year - dob.year - (
        1 if (at.month, at.day) < (dob.month, dob.day) else 0
    )

    b = basis.lower()
    if b in ("last", "alb", "age_last", "lastbirthday"):
        return int(completed)

    # fraction of the current age-year elapsed (for exact / nearest)
    try:
        last_bday = dob.replace(year=dob.year + completed)
    except ValueError:  # Feb 29 birthday in a non-leap year
        last_bday = dob.replace(year=dob.year + completed, day=28)
    try:
        next_bday = dob.replace(year=dob.year + completed + 1)
    except ValueError:
        next_bday = dob.replace(year=dob.year + completed + 1, day=28)

    span = (next_bday - last_bday).days
    frac = (at - last_bday).days / span if span else 0.0

    if b in ("exact", "anniversary"):
        return float(completed + frac)
    if b in ("nearest", "anb", "age_nearest", "nearestbirthday"):
        return int(completed + (1 if frac >= 0.5 else 0))
    raise ValueError(f"unknown age basis: {basis!r}.")


def exposure_years(
    entry,
    exit,
    study_start,
    study_end,
    *,
    convention: str = "actual/365",
) -> float:
    """Exposure (in years) a record contributes within a study window.

    The exposure is the overlap of ``[entry, exit]`` with
    ``[study_start, study_end]``, measured under the given day-count convention.
    Returns 0 when the record and study window do not overlap.
    """
    start = max(pd.Timestamp(entry), pd.Timestamp(study_start))
    end = min(pd.Timestamp(exit), pd.Timestamp(study_end))
    if end <= start:
        return 0.0
    return float(year_fraction(start, end, convention))


def add_exposure_column(
    df: pd.DataFrame,
    entry_col: str,
    exit_col: str,
    study_start,
    study_end,
    *,
    exposure_col: str = "exposure_years",
    convention: str = "actual/365",
    copy: bool = True,
) -> pd.DataFrame:
    """Add an exposure-years column for each record over a study window.

    Useful for building the denominator of an actual-to-expected study.
    """
    for col in (entry_col, exit_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    result = df.copy() if copy else df
    ss = pd.Timestamp(study_start)
    se = pd.Timestamp(study_end)
    entry = pd.to_datetime(result[entry_col]).clip(lower=ss)
    exit_ = pd.to_datetime(result[exit_col]).clip(upper=se)
    days = (exit_ - entry).dt.days.clip(lower=0)
    denom = 365.0 if convention.lower().replace(" ", "") in ("actual/365", "act/365") else 360.0
    if convention.lower().replace(" ", "") not in ("actual/365", "act/365", "actual/360", "act/360"):
        # fall back to per-row exact calculation for non-actual/fixed conventions
        result[exposure_col] = [
            exposure_years(e, x, ss, se, convention=convention)
            for e, x in zip(result[entry_col], result[exit_col], strict=True)
        ]
        return result
    result[exposure_col] = days / denom
    return result
