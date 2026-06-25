"""Small DataFrame validation helpers.

ActuarialPy intentionally avoids wrapping ordinary pandas operations unless the
helper adds validation or actuarial-specific safeguards.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def as_list(value: Any) -> list[Any]:
    """Return value as a list. Strings are treated as single values."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def validate_columns(df: pd.DataFrame, cols: str | Iterable[str]) -> None:
    """Raise ValueError if any required columns are missing."""
    required = as_list(cols)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_unique_keys(df: pd.DataFrame, keys: str | Iterable[str], *, name: str = "data") -> None:
    """Raise ValueError if key columns are not unique."""
    key_list = as_list(keys)
    validate_columns(df, key_list)
    duplicates = df[df.duplicated(key_list, keep=False)]
    if not duplicates.empty:
        examples = duplicates[key_list].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"{name} has duplicate keys for {key_list}. Examples: {examples}")


def factor_lookup(
    df: pd.DataFrame,
    factors: pd.DataFrame,
    keys: str | Iterable[str],
    *,
    factor_col: str,
    default: float | None = None,
) -> np.ndarray:
    """Join a factor onto ``df`` by value on one or more existing key columns.

    The single factor-join primitive behind grouped completion, seasonality, and
    :func:`adjust`. ``factors`` is a tidy table containing ``keys`` and ``factor_col``;
    each row of ``df`` is matched on its ``keys`` values. The factor table must be unique
    on ``keys`` -- a duplicate would fan rows out on the join -- so this raises otherwise.
    Returns a float array aligned to ``df``'s row order (the frame's own index never
    participates). An absent key gives ``default`` (``NaN`` when ``default`` is ``None``
    -- a surfaced gap, never silently filled).
    """
    key_cols = as_list(keys)
    if not key_cols:
        raise ValueError("keys must name at least one column")
    validate_columns(factors, key_cols + [factor_col])
    validate_columns(df, key_cols)
    ensure_unique_keys(factors, key_cols, name="factor table")
    if len(key_cols) == 1:
        lookup = factors.set_index(key_cols[0])[factor_col]
        factor = np.array(df[key_cols[0]].map(lookup), dtype="float64")
    else:
        lookup = factors.set_index(key_cols)[factor_col]
        row_keys = pd.MultiIndex.from_frame(df[key_cols])
        factor = np.array(lookup.reindex(row_keys), dtype="float64")
    if default is not None:
        factor = np.where(np.isnan(factor), float(default), factor)
    return factor


def grouped_factor_lookup(
    df: pd.DataFrame,
    factors: pd.DataFrame,
    by: str | Iterable[str],
    key_values: Any,
    *,
    key_col: str,
    factor_col: str,
) -> np.ndarray:
    """Look up a per-segment factor by ``(group..., key)``, joining by value.

    Thin wrapper over :func:`factor_lookup` for the case where the key is a *derived*
    quantity (``key_values``, positional in row order) rather than an existing column --
    e.g. a season extracted from a date, or a development period. ``factors`` is a tidy
    table with grouping column(s) ``by``, a key column (``key_col``) and ``factor_col``;
    it must be unique on ``by + [key_col]``. Returns a float array with ``NaN`` where the
    ``(group, key)`` pair is absent; order is preserved regardless of index.
    """
    by_cols = as_list(by)
    if not by_cols:
        raise ValueError("Pass by=... naming the grouping column(s) for a per-segment factor table.")
    key_frame = df[by_cols].reset_index(drop=True).copy()
    key_frame[key_col] = key_values
    return factor_lookup(key_frame, factors, by_cols + [key_col], factor_col=factor_col)


def sum_columns(df: pd.DataFrame, cols: str | Iterable[str], *, min_count: int = 1) -> pd.Series:
    """Validate and sum one or more DataFrame columns row-wise.

    This is kept as a small internal-friendly utility because many actuarial
    functions accept several expense or revenue columns. For simple user code,
    pandas syntax such as ``df[cols].sum(axis=1)`` is usually sufficient.
    """
    cols_list = as_list(cols)
    if not cols_list:
        raise ValueError("cols must contain at least one column")
    validate_columns(df, cols_list)
    return df[cols_list].sum(axis=1, min_count=min_count)


_DATE_NAME_TOKENS = {"date", "month", "period", "year", "quarter", "week", "yearmonth", "yyyymm"}
_DATE_AFFIX_TOKENS = ("date", "month", "period", "quarter", "week", "year")


def is_date_like(series: pd.Series, name: str) -> bool:
    """Heuristic test for a date/time column.

    Returns True if the column has a datetime or period dtype, or its name matches a
    common date token (e.g. ``month``, ``paid_month``, ``effective_date``). Used to
    place date columns first in summary output.
    """
    if pd.api.types.is_datetime64_any_dtype(series) or isinstance(series.dtype, pd.PeriodDtype):
        return True
    lowered = name.lower()
    if lowered in _DATE_NAME_TOKENS:
        return True
    return any(lowered.startswith(tok + "_") or lowered.endswith("_" + tok) for tok in _DATE_AFFIX_TOKENS)
