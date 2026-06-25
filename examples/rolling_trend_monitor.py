"""Rolling-12 trend monitoring, by composition (no new library surface).

A monthly *change-in-trend* monitor. Each evaluation month compares the trailing
12-month window to the same window one year earlier, and ``decompose_pmpm_trend``
reports both the observed year-over-year trend and whether it is utilization- or
unit-cost-driven. Trailing-12 windows are the right basis because each window holds
every calendar month exactly once, so seasonality cancels in the comparison.

This is a diagnostic, not a trend assumption -- and it needs nothing new from the
library. The whole monitor is three moves: slice two 12-month windows, decompose,
tabulate across evaluation months. For a large-claim split, pool first
(``pool_losses`` / ``excess_over_threshold``) and run the same monitor on the capped
and excess series; for a faster read on acceleration, fit a slope with ``fit_trend``
on the deseasonalized series instead of waiting for the trailing average to bend.

    pip install actuarialpy
    python rolling_trend_monitor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from actuarialpy import decompose_pmpm_trend  # noqa: E402


def build_monthly_panel() -> pd.DataFrame:
    """A monthly incurred panel: real seasonality, a 2024 utilization acceleration,
    steady ~6% unit-cost trend, and one large-claim spike month. Columns: ``month``
    (period), ``member_months``, ``claim_count``, ``allowed``.
    """
    months = pd.period_range("2022-01", "2024-12", freq="M")
    seasonal = {1: 1.18, 2: 1.12, 3: 1.05, 4: 0.98, 5: 0.95, 6: 0.92,
                7: 0.90, 8: 0.93, 9: 0.98, 10: 1.02, 11: 1.07, 12: 1.10}
    base_freq, base_sev, base_mm = 1.30, 350.0, 12000.0
    first_2024 = 24  # months from the start to 2024-01

    rows: list[dict] = []
    for t, m in enumerate(months):
        if m.year < 2024:                                  # utilization +2%/yr ...
            freq_level = base_freq * (1.02 ** (t / 12))
        else:                                              # ... then accelerates to +6%/yr in 2024
            base_24 = base_freq * (1.02 ** (first_2024 / 12))
            freq_level = base_24 * (1.06 ** ((t - first_2024) / 12))
        severity = base_sev * (1.06 ** (t / 12))           # unit cost +6%/yr throughout
        member_months = base_mm * (1.003 ** t)
        claim_count = freq_level * seasonal[m.month] * member_months
        allowed = severity * claim_count
        if m == pd.Period("2024-03", "M"):                 # a single large claimant lands
            allowed *= 1.18
        rows.append({"month": m, "member_months": member_months,
                     "claim_count": claim_count, "allowed": allowed})
    return pd.DataFrame(rows)


def trailing_window(df: pd.DataFrame, end: pd.Period, length: int = 12) -> pd.DataFrame:
    start = end - (length - 1)
    return df[(df["month"] >= start) & (df["month"] <= end)]


def main() -> None:
    df = build_monthly_panel()
    cols = dict(count_col="claim_count", loss_col="allowed", exposure_col="member_months")

    records: list[dict] = []
    for eval_month in pd.period_range("2024-01", "2024-12", freq="M"):
        current = trailing_window(df, eval_month)            # trailing 12 ending at eval_month
        prior = trailing_window(df, eval_month - 12)         # same window one year earlier
        d = decompose_pmpm_trend(prior, current, **cols).iloc[0]

        # contrast: single-month year-over-year (same calendar month, one month each)
        this_pmpm = df.loc[df["month"] == eval_month, "allowed"].iloc[0] / df.loc[df["month"] == eval_month, "member_months"].iloc[0]
        year_ago_pmpm = df.loc[df["month"] == eval_month - 12, "allowed"].iloc[0] / df.loc[df["month"] == eval_month - 12, "member_months"].iloc[0]

        records.append({
            "eval_month": str(eval_month),
            "r12_pmpm": d["pmpm_current"],
            "r12_yoy": d["pmpm_trend"] - 1,        # the monitor
            "util": d["util_trend"] - 1,           # ... split into its drivers
            "cost": d["cost_trend"] - 1,
            "1mo_yoy": this_pmpm / year_ago_pmpm - 1,
        })

    out = pd.DataFrame(records)
    print("Rolling-12 observed trend, refreshed monthly (each row compares two annual windows)\n")
    show = out.copy()
    show["r12_pmpm"] = show["r12_pmpm"].map(lambda v: f"{v:,.0f}")
    for c in ("r12_yoy", "util", "cost", "1mo_yoy"):
        show[c] = show[c].map(lambda v: f"{v:+.1%}")
    print(show.to_string(index=False))

    print("\nReading it:")
    print("- r12_yoy is the monitor: it climbs ~8% -> ~12% across 2024. Because a trailing")
    print("  average ramps a change in slowly, the utilization acceleration that began in")
    print("  January only reads ~+3.8% by December -- the signal is real but lagged.")
    print("- The util/cost split separates the story: utilization drifting up is broad-based,")
    print("  while unit cost jumping to ~+7.7% the moment March enters the window is the one")
    print("  large claim, not pricing. That is the cue to pool and confirm.")
    print("- 1mo_yoy spikes to ~+28% in March on that same claim, then settles -- which is")
    print("  exactly why the rolling window, not a single month, is the monitor.")
    print("\nLarge-claim split, same pattern: pool_losses / excess_over_threshold to cap the")
    print("series, then run this monitor on the capped and excess pieces separately.")


if __name__ == "__main__":
    main()
