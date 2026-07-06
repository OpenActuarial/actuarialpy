"""Credibility with the free functions: blend a group's own experience with the book.

Two classical approaches, both on plain aggregates using the free functions:

  1. Limited-fluctuation (classical) credibility -- the square-root rule.
     ``full_credibility_claims`` gives the full-credibility standard, then
     ``limited_fluctuation_z`` turns exposure into Z, and
     ``credibility_weighted_estimate`` blends the group with its complement.
  2. Greatest-accuracy (Buhlmann-Straub) credibility -- Z is derived from the
     expected process variance (EPV) and the variance of hypothetical means (VHM),
     via k = EPV / VHM, instead of a filed rule. (``Buhlmann.fit`` and
     ``BuhlmannStraub.from_frame`` estimate EPV and VHM from data.)

    pip install actuarialpy
    python credibility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import actuarialpy as ap  # noqa: E402
from _sample_data import sample_member_months  # noqa: E402

# A filed full-credibility standard, in member-months (illustrative). Many shops
# file a volume standard like this rather than deriving one from claim counts.
FILED_STANDARD_MM = 12_000.0


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    df = sample_member_months()

    # The complement of credibility: the book-wide loss ratio every group is
    # blended toward.
    book_lr = float(df["total_claims"].sum() / df["premium"].sum())

    # One row per group: summed claims, premium, and exposure, plus the group's
    # own loss ratio. Plain pandas -- credibility needs only aggregates.
    by_group = (
        df.groupby("group_id")
        .agg(total_claims=("total_claims", "sum"),
             premium=("premium", "sum"),
             member_months=("member_months", "sum"))
        .reset_index()
    )
    by_group["loss_ratio"] = by_group["total_claims"] / by_group["premium"]

    section("The full-credibility standard (limited fluctuation)")
    freq_standard = ap.full_credibility_claims(confidence=0.90, tolerance=0.05)
    agg_standard = ap.full_credibility_claims(confidence=0.90, tolerance=0.05, severity_cv=1.5)
    print(f"claims for full credibility, frequency (90% / 5%) : {freq_standard:,.0f}")
    print(f"claims for full credibility, aggregate (CV 1.5)   : {agg_standard:,.0f}")
    print(f"filed standard used below (member-months)         : {FILED_STANDARD_MM:,.0f}")

    section("1. Limited-fluctuation credibility, per group (vectorized)")
    # Series in -> Series out: one Z and one blended loss ratio per group.
    by_group["Z"] = ap.limited_fluctuation_z(by_group["member_months"], FILED_STANDARD_MM)
    by_group["cred_lr"] = ap.credibility_weighted_estimate(
        observed=by_group["loss_ratio"], complement=book_lr, z=by_group["Z"],
    )
    show = by_group[["group_id", "member_months", "loss_ratio", "Z", "cred_lr"]].copy()
    show["group_id"] = show["group_id"].astype(int)
    show["member_months"] = show["member_months"].astype(int)
    print(f"book loss ratio (complement): {book_lr:.3f}\n")
    print(show.round({"loss_ratio": 3, "Z": 2, "cred_lr": 3}).to_string(index=False))
    print("\ncred_lr = Z * group loss ratio + (1 - Z) * book loss ratio")

    section("2. Greatest-accuracy (Buhlmann-Straub) credibility")
    # Here Z is not a filed square-root rule -- it comes from the variance
    # structure: k = EPV / VHM, then Z = exposure / (exposure + k). We state the
    # structural parameters so the mechanics are visible; in practice they are
    # estimated (Buhlmann.fit for equal exposures, BuhlmannStraub.from_frame for
    # long data). Units are loss per member-month, but the identity is the same
    # for any base.
    OVERALL_PMM = 350.0   # grand mean loss per member-month (the complement)
    EPV = 90_000.0        # expected process variance -- noise within a risk
    VHM = 200.0           # variance of the true risk means -- signal between risks

    # Each risk: its own observed mean PMM and its exposure in member-months.
    risks = [("A", 395.0, 3000), ("B", 470.0, 1200), ("C", 280.0, 450), ("D", 250.0, 150)]
    model = ap.BuhlmannStraub(
        overall_mean=OVERALL_PMM, epv=EPV, vhm=VHM, weights=[w for _, _, w in risks],
    )
    print(f"k = EPV / VHM = {model.k:,.0f}   (the exposure at which Z = 0.5)\n")
    print(f"{'risk':>4}  {'exposure':>9}  {'own PMM':>8}  {'Z':>5}  {'cred PMM':>9}")
    for label, own_mean, weight in risks:
        z = float(model.z(weight))
        cred = float(model.premium(own_mean, weight))
        print(f"{label:>4}  {weight:>9,.0f}  {own_mean:>8,.0f}  {z:>5.2f}  {cred:>9,.1f}")
    print("\ncred PMM = Z * risk's own PMM + (1 - Z) * overall mean PMM")
    print("Bigger risks earn a higher Z; small risks are pulled toward the 350 mean.")


if __name__ == "__main__":
    main()
