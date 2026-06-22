"""Claimant and large-risk concentration summaries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from actuarialpy.columns import as_list, sum_columns, validate_columns
from actuarialpy.metrics import per_exposure, safe_divide


def _per_exposure_name(amount_name: str, exposure_col: str) -> str:
    if exposure_col == "member_months":
        return f"{amount_name}_pmpm"
    if exposure_col == "subscriber_months":
        return f"{amount_name}_pspm"
    if exposure_col == "employee_months":
        return f"{amount_name}_pepm"
    return f"{amount_name}_per_{exposure_col}"


def summarize_claimants(
    df: pd.DataFrame,
    *,
    claimant_col: str,
    amount_cols: str | Iterable[str],
    groupby: str | Iterable[str] | None = None,
    exposure_col: str | None = None,
    amount_name: str = "total_expense",
) -> pd.DataFrame:
    """Aggregate experience to claimant/member/risk level.

    ``claimant_col`` can be a member ID, policy ID, claim group ID, or another
    entity identifier. The function is descriptive; it does not cap, pool, or
    otherwise adjust the underlying amounts.
    """
    groups = as_list(groupby)
    amounts = as_list(amount_cols)
    required = groups + [claimant_col] + amounts + ([exposure_col] if exposure_col else [])
    validate_columns(df, required)

    agg_cols = list(dict.fromkeys(amounts + ([exposure_col] if exposure_col else [])))
    out = df[groups + [claimant_col] + agg_cols].groupby(
        groups + [claimant_col], dropna=False, as_index=False
    ).sum(numeric_only=True)
    out[amount_name] = sum_columns(out, amounts)
    if exposure_col:
        out[_per_exposure_name(amount_name, exposure_col)] = per_exposure(out[amount_name], out[exposure_col])
    return out


def top_claimants(
    df: pd.DataFrame,
    *,
    claimant_col: str,
    amount_cols: str | Iterable[str] | None = None,
    amount_col: str | None = None,
    groupby: str | Iterable[str] | None = None,
    n: int = 25,
    amount_name: str = "total_expense",
) -> pd.DataFrame:
    """Return the top claimants by amount, optionally within each group."""
    if n <= 0:
        raise ValueError("n must be positive")
    groups = as_list(groupby)

    if amount_col is None:
        if amount_cols is None:
            raise ValueError("Pass either amount_col or amount_cols.")
        base = summarize_claimants(
            df,
            claimant_col=claimant_col,
            amount_cols=amount_cols,
            groupby=groups,
            amount_name=amount_name,
        )
        amount_col = amount_name
    else:
        validate_columns(df, groups + [claimant_col, amount_col])
        base = df[groups + [claimant_col, amount_col]].copy()

    sort_cols = groups + [amount_col] if groups else [amount_col]
    ascending = [True] * len(groups) + [False]
    base = base.sort_values(sort_cols, ascending=ascending).copy()
    if groups:
        base["rank"] = base.groupby(groups, dropna=False)[amount_col].rank(method="first", ascending=False).astype(int)
        totals = base.groupby(groups, dropna=False)[amount_col].sum().reset_index(name="_group_total")
        base = base.merge(totals, on=groups, how="left")
        base = base[base["rank"] <= n].copy()
        base["share_of_total"] = safe_divide(base[amount_col], base["_group_total"])
        base["cumulative_share"] = base.groupby(groups, dropna=False)["share_of_total"].cumsum()
        return base.drop(columns=["_group_total"])

    base["rank"] = range(1, len(base) + 1)
    total = base[amount_col].sum()
    base = base[base["rank"] <= n].copy()
    base["share_of_total"] = safe_divide(base[amount_col], total)
    base["cumulative_share"] = base["share_of_total"].cumsum()
    return base


def large_claimant_flags(
    df: pd.DataFrame,
    *,
    amount_col: str = "total_expense",
    thresholds: Sequence[float] = (50_000, 100_000, 250_000),
) -> pd.DataFrame:
    """Add boolean flags for claimants above one or more amount thresholds."""
    validate_columns(df, [amount_col])
    out = df.copy()
    for threshold in thresholds:
        label = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "_")
        out[f"is_over_{label}"] = out[amount_col] >= threshold
    return out


def claim_concentration(
    df: pd.DataFrame,
    *,
    amount_col: str = "total_expense",
    groupby: str | Iterable[str] | None = None,
    top_n: Sequence[int] = (10, 25),
    thresholds: Sequence[float] = (50_000, 100_000, 250_000),
) -> pd.DataFrame:
    """Summarize how concentrated total amounts are among top claimants.

    The input should generally be one row per claimant within the requested
    grouping level, such as the output of ``summarize_claimants``.
    """
    groups = as_list(groupby)
    validate_columns(df, groups + [amount_col])

    def summarize(part: pd.DataFrame) -> dict[str, float]:
        sorted_part = part.sort_values(amount_col, ascending=False)
        total = sorted_part[amount_col].sum()
        row: dict[str, float] = {
            "claimant_count": len(sorted_part),
            "total_amount": total,
        }
        for n in top_n:
            top_amount = sorted_part.head(n)[amount_col].sum()
            row[f"top_{n}_amount"] = top_amount
            row[f"top_{n}_share"] = safe_divide(top_amount, total)
        for threshold in thresholds:
            label = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "_")
            mask = sorted_part[amount_col] >= threshold
            threshold_amount = sorted_part.loc[mask, amount_col].sum()
            row[f"count_over_{label}"] = int(mask.sum())
            row[f"amount_over_{label}"] = threshold_amount
            row[f"share_over_{label}"] = safe_divide(threshold_amount, total)
        return row

    if groups:
        rows = []
        for keys, part in df.groupby(groups, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rows.append({**dict(zip(groups, keys)), **summarize(part)})
        return pd.DataFrame(rows)
    return pd.DataFrame([summarize(df)])
