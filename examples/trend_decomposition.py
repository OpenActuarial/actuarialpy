"""Trend decomposition: utilization x unit cost, and adding a mix term.

The standard "how much of the PMPM trend is utilization vs unit cost" exhibit, and
why a third *mix* term matters once your book is a blend of cells. With ``mix_by``
omitted you get the exact two-way identity (frequency x severity). Pass ``mix_by`` and
PMPM is split into within-cell utilization, within-cell unit cost, and the effect of
the membership composition shifting across those cells -- the piece the two-way
otherwise smears into utilization and unit cost. The split uses LMDI, so all three
reconcile exactly to the total, both multiplicatively and in dollars.

    pip install actuarialpy
    python trend_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from actuarialpy import Experience, decompose_pmpm_trend  # noqa: E402
from _sample_data import sample_trend_cells  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def pct(factor: float) -> str:
    return f"{factor - 1:+.1%}"


def main() -> None:
    panel = sample_trend_cells()
    prior = panel[panel["period"] == "2024"]
    current = panel[panel["period"] == "2025"]
    cols = dict(count_col="claim_count", loss_col="allowed", exposure_col="member_months")

    section("Two-way: utilization x unit cost (mix_by omitted)")
    two = decompose_pmpm_trend(prior, current, **cols).iloc[0]
    print(f"PMPM {two['pmpm_prior']:.2f} -> {two['pmpm_current']:.2f}   trend {pct(two['pmpm_trend'])}")
    print(f"  utilization {pct(two['util_trend'])}   unit cost {pct(two['cost_trend'])}")
    print("  exact identity: util_trend * cost_trend == pmpm_trend.")
    print("  But these are book-wide -- the enrollment shift toward the High segment")
    print("  inflates both, since sicker members use more AND cost more per service.")

    section("Three-way: add a mix term (mix_by='segment')")
    three = decompose_pmpm_trend(prior, current, mix_by="segment", **cols).iloc[0]
    prod = three["util_trend"] * three["cost_trend"] * three["mix_trend"]
    dollars = three["util_effect"] + three["cost_effect"] + three["mix_effect"]
    print(f"PMPM trend {pct(three['pmpm_trend'])}")
    print(f"  utilization {pct(three['util_trend'])}   unit cost {pct(three['cost_trend'])}   mix {pct(three['mix_trend'])}")
    print(f"  multiplicative: {three['util_trend']:.4f} * {three['cost_trend']:.4f} * {three['mix_trend']:.4f} = {prod:.4f}")
    print(f"  dollars:        util {three['util_effect']:+.2f} + cost {three['cost_effect']:+.2f} "
          f"+ mix {three['mix_effect']:+.2f} = {dollars:+.2f}  (PMPM change {three['pmpm_change']:+.2f})")
    print("  Within every cell utilization trends +3% and unit cost +4% -- exactly what")
    print("  the three-way recovers. The remaining ~mix is the population getting sicker.")
    print(f"  Separating mix pulls utilization {pct(two['util_trend'])} -> {pct(three['util_trend'])} "
          f"and unit cost {pct(two['cost_trend'])} -> {pct(three['cost_trend'])}.")

    section("Mix over a different cell set, and the cross")
    by_region = decompose_pmpm_trend(prior, current, mix_by="region", **cols).iloc[0]
    cross = decompose_pmpm_trend(prior, current, mix_by=["segment", "region"], **cols).iloc[0]
    print(f"  mix_by='segment'            -> mix {pct(three['mix_trend'])}")
    print(f"  mix_by='region'             -> mix {pct(by_region['mix_trend'])}")
    print(f"  mix_by=['segment','region'] -> mix {pct(cross['mix_trend'])}   (the joint shift, one blended term)")
    print("  The cross is not the sum of the two single-dimension mixes -- the gap is how")
    print("  segment and region co-move. For separate attribution, run one per dimension.")

    section("Report by one axis, mix over another (on='region', mix_by='segment')")
    out = decompose_pmpm_trend(prior, current, on="region", mix_by="segment", **cols)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(out[["region", "pmpm_trend", "util_trend", "cost_trend", "mix_trend"]].to_string(index=False))
    print("  on= groups the output rows; mix_by= defines the mix cells within each group.")

    section("Same split on an Experience (columns bound once)")
    exp = Experience(panel, expense="allowed", revenue="premium",
                     exposure="member_months", count="claim_count")
    fac = exp.decompose_trend(
        period_col="period", prior_period="2024", current_period="2025", mix_by="segment",
    ).iloc[0]
    print(f"  exp.decompose_trend(period_col='period', ...) -> util {pct(fac['util_trend'])}  "
          f"cost {pct(fac['cost_trend'])}  mix {pct(fac['mix_trend'])}")
    print("  Identical to mix_by='segment' above -- but the Experience holds count/loss/exposure,")
    print("  so you bind the columns once and the period split works like exp.trend.")


if __name__ == "__main__":
    main()
