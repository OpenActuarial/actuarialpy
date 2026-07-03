"""Trend decomposition: frequency x severity, and adding a mix term.

The standard "how much of the Per-exposure trend is frequency vs severity" exhibit, and
why a third *mix* term matters once your book is a blend of cells. With ``mix_by``
omitted you get the exact two-way identity (frequency x severity). Pass ``mix_by`` and
The per-exposure loss is split into within-cell frequency, within-cell severity, and the effect of
the membership composition shifting across those cells -- the piece the two-way
otherwise smears into frequency and severity. The split uses LMDI, so all three
reconcile exactly to the total, both multiplicatively and in dollars.

    pip install actuarialpy
    python trend_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from actuarialpy import Experience, decompose_per_exposure_trend  # noqa: E402
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

    section("Two-way: frequency x severity (mix_by omitted)")
    two = decompose_per_exposure_trend(prior, current, **cols).iloc[0]
    print(f"Loss/exposure {two['loss_per_exposure_prior']:.2f} -> {two['loss_per_exposure_current']:.2f}   trend {pct(two['loss_per_exposure_trend'])}")
    print(f"  frequency {pct(two['frequency_trend'])}   severity {pct(two['severity_trend'])}")
    print("  exact identity: frequency_trend * severity_trend == loss_per_exposure_trend.")
    print("  But these are book-wide -- the enrollment shift toward the High segment")
    print("  inflates both, since sicker members use more AND cost more per service.")

    section("Three-way: add a mix term (mix_by='segment')")
    three = decompose_per_exposure_trend(prior, current, mix_by="segment", **cols).iloc[0]
    prod = three["frequency_trend"] * three["severity_trend"] * three["mix_trend"]
    dollars = three["frequency_effect"] + three["severity_effect"] + three["mix_effect"]
    print(f"Per-exposure trend {pct(three['loss_per_exposure_trend'])}")
    print(f"  frequency {pct(three['frequency_trend'])}   severity {pct(three['severity_trend'])}   mix {pct(three['mix_trend'])}")
    print(f"  multiplicative: {three['frequency_trend']:.4f} * {three['severity_trend']:.4f} * {three['mix_trend']:.4f} = {prod:.4f}")
    print(f"  dollars:        freq {three['frequency_effect']:+.2f} + sev {three['severity_effect']:+.2f} "
          f"+ mix {three['mix_effect']:+.2f} = {dollars:+.2f}  (per-exposure change {three['loss_per_exposure_change']:+.2f})")
    print("  Within every cell frequency trends +3% and severity +4% -- exactly what")
    print("  the three-way recovers. The remaining ~mix is the population getting sicker.")
    print(f"  Separating mix pulls frequency {pct(two['frequency_trend'])} -> {pct(three['frequency_trend'])} "
          f"and severity {pct(two['severity_trend'])} -> {pct(three['severity_trend'])}.")

    section("Mix over a different cell set, and the cross")
    by_region = decompose_per_exposure_trend(prior, current, mix_by="region", **cols).iloc[0]
    cross = decompose_per_exposure_trend(prior, current, mix_by=["segment", "region"], **cols).iloc[0]
    print(f"  mix_by='segment'            -> mix {pct(three['mix_trend'])}")
    print(f"  mix_by='region'             -> mix {pct(by_region['mix_trend'])}")
    print(f"  mix_by=['segment','region'] -> mix {pct(cross['mix_trend'])}   (the joint shift, one blended term)")
    print("  The cross is not the sum of the two single-dimension mixes -- the gap is how")
    print("  segment and region co-move. For separate attribution, run one per dimension.")

    section("Report by one axis, mix over another (on='region', mix_by='segment')")
    out = decompose_per_exposure_trend(prior, current, on="region", mix_by="segment", **cols)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(out[["region", "loss_per_exposure_trend", "frequency_trend", "severity_trend", "mix_trend"]].to_string(index=False))
    print("  on= groups the output rows; mix_by= defines the mix cells within each group.")

    section("Same split on an Experience (columns bound once)")
    exp = Experience(panel, expense="allowed", revenue="premium",
                     exposure="member_months", count="claim_count")
    fac = exp.decompose_trend(
        period_col="period", prior_period="2024", current_period="2025", mix_by="segment",
    ).iloc[0]
    print(f"  exp.decompose_trend(period_col='period', ...) -> freq {pct(fac['frequency_trend'])}  "
          f"sev {pct(fac['severity_trend'])}  mix {pct(fac['mix_trend'])}")
    print("  Identical to mix_by='segment' above -- but the Experience holds count/loss/exposure,")
    print("  so you bind the columns once and the period split works like exp.trend.")


if __name__ == "__main__":
    main()
