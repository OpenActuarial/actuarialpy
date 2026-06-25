"""Frequency-severity and PMPM trend decomposition.

Splits a per-member cost (PMPM / pure premium) into its utilization and unit-cost
drivers, and decomposes the change between two periods into a utilization effect and
a unit-cost effect -- the standard "how much of the trend is utilization vs unit
cost" exhibit. Decomposing requires a claim (or service) count alongside losses and
exposure.

Passing ``mix_by`` to :func:`decompose_pmpm_trend` adds a third **mix** component
(utilization x unit cost x mix), separating the effect of the membership composition
shifting across cells from genuine within-cell utilization and unit-cost movement.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.metrics import frequency, per_exposure, safe_divide, severity, utilization_per_1000


def frequency_severity_summary(
    df: pd.DataFrame,
    *,
    count_col: str,
    loss_col: str,
    exposure_col: str,
    groupby: str | Iterable[str] | None = None,
    annualization: float = 12,
) -> pd.DataFrame:
    """Per-group claim frequency, severity, and PMPM.

    Counts, losses, and exposure are aggregated first, then the rates are derived
    after aggregation (avoiding averaging row-level rates). The identity
    ``pmpm == frequency * severity`` holds for every row. ``frequency`` is claims per
    exposure unit (per member month for monthly data), ``severity`` is loss per claim,
    ``util_per_1000`` is annualized claims per 1,000 members, and ``pmpm`` is loss per
    exposure unit.
    """
    groups = as_list(groupby)
    validate_columns(df, groups + [count_col, loss_col, exposure_col])
    amount_cols = [count_col, loss_col, exposure_col]
    if groups:
        summary = df[groups + amount_cols].groupby(groups, dropna=False, as_index=False).sum(numeric_only=True)
    else:
        summary = pd.DataFrame({col: [df[col].sum()] for col in amount_cols})

    summary["frequency"] = frequency(summary[count_col], summary[exposure_col])
    summary["severity"] = severity(summary[loss_col], summary[count_col])
    summary["util_per_1000"] = utilization_per_1000(summary[count_col], summary[exposure_col], annualization=annualization)
    summary["pmpm"] = per_exposure(summary[loss_col], summary[exposure_col])

    ordered = groups + [exposure_col, count_col, loss_col, "frequency", "severity", "util_per_1000", "pmpm"]
    return summary[[col for col in ordered if col in summary.columns]]


def _logarithmic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise logarithmic mean ``L(a, b) = (a - b) / (ln a - ln b)``, with ``L(a, a) = a``.

    Defined for strictly positive inputs. This is the weight kernel behind the LMDI
    (logarithmic mean Divisia index) decomposition, which reconciles exactly with no
    residual term.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    close = np.isclose(a, b)
    log_diff = np.where(close, 1.0, np.log(a) - np.log(b))
    return np.where(close, a, (a - b) / log_diff)


def _aggregate_cells(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Sum ``cols`` over ``keys`` (or the whole frame when ``keys`` is empty)."""
    if keys:
        return df[keys + cols].groupby(keys, dropna=False, as_index=False).sum(numeric_only=True)
    return pd.DataFrame({col: [df[col].sum()] for col in cols})


def _lmdi_three_way(
    m0: np.ndarray, n0: np.ndarray, a0: np.ndarray,
    m1: np.ndarray, n1: np.ndarray, a1: np.ndarray,
) -> dict[str, float]:
    """LMDI utilization / unit-cost / mix split for one reporting group.

    Each argument is an array over the mix cells: ``m`` exposure, ``n`` count, ``a``
    dollars; suffix ``0`` prior and ``1`` current. Returns the multiplicative factors
    (``util_trend * cost_trend * mix_trend == pmpm_trend``) and the additive dollar
    effects (``util_effect + cost_effect + mix_effect == pmpm_change``); both exact.
    """
    big_m0, big_m1 = m0.sum(), m1.sum()
    u0, c0, w0 = n0 / m0, a0 / n0, m0 / big_m0
    u1, c1, w1 = n1 / m1, a1 / n1, m1 / big_m1
    v0, v1 = a0 / big_m0, a1 / big_m1            # cell contribution to group PMPM (== w*u*c)
    p0, p1 = float(v0.sum()), float(v1.sum())    # group PMPM each period
    l_cell = _logarithmic_mean(v1, v0)
    l_tot = float(_logarithmic_mean(np.array([p1]), np.array([p0]))[0])
    omega = l_cell / l_tot
    ln_u, ln_c, ln_w = np.log(u1 / u0), np.log(c1 / c0), np.log(w1 / w0)
    return {
        "pmpm_prior": p0,
        "pmpm_current": p1,
        "pmpm_trend": p1 / p0,
        "util_trend": float(np.exp(np.sum(omega * ln_u))),
        "cost_trend": float(np.exp(np.sum(omega * ln_c))),
        "mix_trend": float(np.exp(np.sum(omega * ln_w))),
        "pmpm_change": p1 - p0,
        "util_effect": float(np.sum(l_cell * ln_u)),
        "cost_effect": float(np.sum(l_cell * ln_c)),
        "mix_effect": float(np.sum(l_cell * ln_w)),
    }


def _decompose_pmpm_trend_mix(
    prior: pd.DataFrame,
    current: pd.DataFrame,
    *,
    count_col: str,
    loss_col: str,
    exposure_col: str,
    on: list[str],
    mix_by: str | Iterable[str],
) -> pd.DataFrame:
    """Three-way (utilization x unit cost x mix) PMPM decomposition via LMDI."""
    mix_keys = as_list(mix_by)
    overlap = [k for k in mix_keys if k in on]
    if overlap:
        raise ValueError(
            f"on and mix_by must be distinct dimensions; shared column(s): {overlap}. "
            "Mix is undefined when the mix dimension is also a reporting group."
        )
    cell_keys = on + mix_keys
    cols = [count_col, loss_col, exposure_col]
    validate_columns(prior, cell_keys + cols)
    validate_columns(current, cell_keys + cols)

    p_cells = _aggregate_cells(prior, cell_keys, cols)
    c_cells = _aggregate_cells(current, cell_keys, cols)
    if cell_keys:
        merged = p_cells.merge(c_cells, on=cell_keys, how="outer", suffixes=("_prior", "_current"))
    else:
        merged = pd.concat(
            [p_cells.add_suffix("_prior").reset_index(drop=True),
             c_cells.add_suffix("_current").reset_index(drop=True)],
            axis=1,
        )

    period_cols = [f"{col}_{per}" for per in ("prior", "current") for col in cols]
    invalid = merged[period_cols].isna().any(axis=1) | (merged[period_cols] <= 0).any(axis=1)
    if bool(invalid.any()):
        shown = merged.loc[invalid, cell_keys] if cell_keys else merged.loc[invalid, period_cols]
        raise ValueError(
            "decompose_pmpm_trend(mix_by=...) requires every mix cell to have positive "
            f"{exposure_col!r}, {count_col!r}, and {loss_col!r} in BOTH periods; the "
            "within-cell utilization x unit cost x mix split is undefined otherwise. "
            "Combine sparse cells or filter cells that enter/exit between periods. "
            f"Offending cell(s):\n{shown.to_string(index=False)}"
        )

    e0, n0, l0 = f"{exposure_col}_prior", f"{count_col}_prior", f"{loss_col}_prior"
    e1, n1, l1 = f"{exposure_col}_current", f"{count_col}_current", f"{loss_col}_current"

    def _group_record(sub: pd.DataFrame) -> dict[str, float]:
        return _lmdi_three_way(
            sub[e0].to_numpy(), sub[n0].to_numpy(), sub[l0].to_numpy(),
            sub[e1].to_numpy(), sub[n1].to_numpy(), sub[l1].to_numpy(),
        )

    records: list[dict] = []
    if on:
        for group_vals, sub in merged.groupby(on, dropna=False, sort=False):
            group_vals = group_vals if isinstance(group_vals, tuple) else (group_vals,)
            records.append({**dict(zip(on, group_vals)), **_group_record(sub)})
    else:
        records.append(_group_record(merged))

    out = pd.DataFrame(records)
    ordered = on + [
        "pmpm_prior", "pmpm_current", "pmpm_trend",
        "util_trend", "cost_trend", "mix_trend",
        "pmpm_change", "util_effect", "cost_effect", "mix_effect",
    ]
    return out[[col for col in ordered if col in out.columns]]


def decompose_pmpm_trend(
    prior: pd.DataFrame,
    current: pd.DataFrame,
    *,
    count_col: str,
    loss_col: str,
    exposure_col: str,
    on: str | Iterable[str] | None = None,
    mix_by: str | Iterable[str] | None = None,
    annualization: float = 12,
) -> pd.DataFrame:
    """Decompose the PMPM change from ``prior`` to ``current``.

    With ``mix_by`` omitted this is the two-way split: both frames are summarized with
    :func:`frequency_severity_summary` (optionally by the ``on`` keys), aligned, and the
    change reported two exact ways:

    - **Multiplicative trend**: ``pmpm_trend == util_trend * cost_trend``, where
      ``util_trend`` is the frequency ratio and ``cost_trend`` the severity ratio.
    - **Additive dollars**: ``pmpm_change == util_effect + cost_effect`` via a symmetric
      (midpoint) split, so the contributions sum exactly to the PMPM change.

    Pass ``mix_by`` (a column or list of columns) to add a third **mix** component. PMPM
    is then decomposed into utilization, unit cost, and the effect of the membership
    composition shifting across the ``mix_by`` cells. Utilization and unit cost are
    measured *within* each cell (free of composition), and mix captures the aggregate
    movement that comes purely from the cell weights changing -- the piece the two-way
    otherwise misattributes to utilization and unit cost. The split uses the LMDI
    (logarithmic mean Divisia index) convention, which is order-free and reconciles
    exactly: ``pmpm_trend == util_trend * cost_trend * mix_trend`` and
    ``pmpm_change == util_effect + cost_effect + mix_effect``.

    A list of columns in ``mix_by`` defines the cells as their cross -- one blended mix
    term, not a per-column attribution; to attribute mix to each dimension separately,
    run the decomposition once per dimension. ``on`` and ``mix_by`` are orthogonal:
    ``on`` groups the output rows, ``mix_by`` defines the mix cells within each group.
    Every cell must have positive count, loss, and exposure in both periods.
    """
    keys = as_list(on)
    if mix_by is not None:
        return _decompose_pmpm_trend_mix(
            prior, current,
            count_col=count_col, loss_col=loss_col, exposure_col=exposure_col,
            on=keys, mix_by=mix_by,
        )

    p = frequency_severity_summary(
        prior, count_col=count_col, loss_col=loss_col, exposure_col=exposure_col,
        groupby=on, annualization=annualization,
    )
    c = frequency_severity_summary(
        current, count_col=count_col, loss_col=loss_col, exposure_col=exposure_col,
        groupby=on, annualization=annualization,
    )
    keep = ["frequency", "severity", "pmpm"]
    if keys:
        merged = p[keys + keep].merge(c[keys + keep], on=keys, how="outer", suffixes=("_prior", "_current"))
    else:
        merged = pd.concat(
            [p[keep].add_suffix("_prior").reset_index(drop=True),
             c[keep].add_suffix("_current").reset_index(drop=True)],
            axis=1,
        )

    merged["util_trend"] = safe_divide(merged["frequency_current"], merged["frequency_prior"])
    merged["cost_trend"] = safe_divide(merged["severity_current"], merged["severity_prior"])
    merged["pmpm_trend"] = safe_divide(merged["pmpm_current"], merged["pmpm_prior"])

    freq_mean = (merged["frequency_prior"] + merged["frequency_current"]) / 2
    sev_mean = (merged["severity_prior"] + merged["severity_current"]) / 2
    merged["pmpm_change"] = merged["pmpm_current"] - merged["pmpm_prior"]
    merged["util_effect"] = (merged["frequency_current"] - merged["frequency_prior"]) * sev_mean
    merged["cost_effect"] = (merged["severity_current"] - merged["severity_prior"]) * freq_mean

    ordered = keys + [
        "pmpm_prior", "pmpm_current", "pmpm_trend", "util_trend", "cost_trend",
        "pmpm_change", "util_effect", "cost_effect",
        "frequency_prior", "frequency_current", "severity_prior", "severity_current",
    ]
    return merged[[col for col in ordered if col in merged.columns]]
