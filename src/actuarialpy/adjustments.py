"""General factor application -- the restatement spine.

Most of experience rating is "take a base amount and carry it through a chain of
factors": completion to ultimate, trend, benefit relativity, area, age/sex or other
demographic loads, network discounts. :func:`adjust` is that move, once: join a factor
to each row by a key (a column already in the frame, optionally within a segment), then
multiply or divide the value by it. :func:`completion <actuarialpy.apply_completion>` and
:func:`deseasonalize <actuarialpy.deseasonalize>` are the same move with the key *derived*
from a date (a development period, a season); ``adjust`` is the general case where the key
is an ordinary column.

The library deliberately does not encode any particular method here: it takes the factors
as input -- a credibility table, an externally-sourced trend, a filed relativity -- and
applies them mechanically, with the same validated join (unique-key / fan-out guard,
surfaced gaps, index-independent) used everywhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from actuarialpy.columns import as_list, factor_lookup, validate_columns


def adjust(
    df: pd.DataFrame,
    factors: float | int | pd.Series | pd.DataFrame,
    *,
    value_col: str,
    on: str | list[str] | None = None,
    by: str | list[str] | None = None,
    how: str = "multiply",
    factor_col: str = "factor",
    out_col: str | None = None,
    audit_col: str | None = None,
    default: float | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Multiply or divide a column by a factor joined on a key.

    The general factor-application primitive behind trend, benefit / area / demographic
    relativities, network discounts -- any per-key multiplier. The factor for each row is
    taken from one of:

    - a **scalar** ``factors`` -- one factor for every row (e.g. a single trend factor);
    - a **Series** indexed by ``on`` -- one key column (e.g. an area factor by region);
    - a tidy **DataFrame** keyed by ``by + on`` with ``factor_col`` -- per-segment factors
      (the shape the ``*_by`` estimators return).

    and applied to ``value_col``: ``how="multiply"`` gives ``value * factor`` (loads,
    trend), ``how="divide"`` gives ``value / factor`` (backing a factor out).

    The join is by value (the frame's index never participates); the factor table must be
    unique on its keys -- a duplicate would fan out the data -- which is enforced. An
    absent key gives ``default`` (``NaN`` when ``default`` is ``None`` -- a surfaced gap,
    never silently filled); pass ``default=1.0`` when a key missing from the table should
    mean "no adjustment". With ``audit_col``, the cumulative *net multiplier* applied to
    ``value_col`` is accumulated there (``factor`` for multiply, ``1 / factor`` for
    divide), so a chain of adjustments leaves a per-row record of total restatement.
    """
    if how not in ("multiply", "divide"):
        raise ValueError("how must be 'multiply' or 'divide'")
    on_cols = as_list(on)
    by_cols = as_list(by)
    validate_columns(df, [value_col] + on_cols + by_cols)
    result = df.copy() if copy else df

    if isinstance(factors, pd.DataFrame):
        keys = by_cols + on_cols
        if not keys:
            raise ValueError("Pass on=... (and optionally by=...) naming the key column(s) for a factor table.")
        factor = factor_lookup(result, factors, keys, factor_col=factor_col, default=default)
    elif isinstance(factors, pd.Series):
        if len(on_cols) != 1:
            raise ValueError("Pass on=<column> (one key) when factors is a Series indexed by that key.")
        if by_cols:
            raise ValueError("by= needs a tidy DataFrame of per-segment factors, not a Series.")
        factor = np.array(result[on_cols[0]].map(factors), dtype="float64")
        if default is not None:
            factor = np.where(np.isnan(factor), float(default), factor)
    elif isinstance(factors, bool):
        raise TypeError("factors must be a number, a Series keyed by `on`, or a tidy DataFrame.")
    elif isinstance(factors, (int, float)):
        factor = np.full(len(result), float(factors))
    else:
        raise TypeError("factors must be a number, a Series keyed by `on`, or a tidy DataFrame.")

    applied = factor if how == "multiply" else 1.0 / factor
    result[out_col or value_col] = result[value_col].to_numpy() * applied
    if audit_col is not None:
        prior = result[audit_col].to_numpy() if audit_col in result.columns else np.ones(len(result))
        result[audit_col] = prior * applied
    return result
