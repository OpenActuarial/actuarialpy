"""End-to-end: a group renewal from experience to indicated rate change.

The whole arc in one flow, showing that the pieces compose into a single study:

    experience -> complete immature months -> trend -> apply relativities
    -> pool large claimants -> credibility-blend with the manual
    -> load for retention -> indicated rate change

Each step prints the projected claims PMPM as it builds, so the renewal reads as one
exhibit. Factors (completion, trend, relativities, the manual rate, the pooling charge)
are supplied as inputs -- the library applies them; it does not set them.

    pip install actuarialpy
    python renewal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import actuarialpy as ap  # noqa: E402
from _sample_data import sample_member_months  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def pmpm(claims: float, member_months: float) -> float:
    return claims / member_months


def main() -> None:
    # ---- inputs the renewal takes as given -------------------------------------
    VALUATION = "2025-12-31"          # claims valued through here; recent months immature
    COMPLETION = pd.Series([0.78, 0.90, 0.96, 0.99, 1.0], index=[0, 1, 2, 3, 4])  # by months of maturity
    ANNUAL_TREND = 0.07
    BASE_MIDPOINT, RATING_MIDPOINT = "2025-01-01", "2026-07-01"
    AREA = pd.Series({"West": 1.06, "Central": 1.00, "East": 0.95}, name="factor")
    BENEFIT = pd.Series({"A": 1.04, "B": 0.98, "C": 1.09}, name="factor")
    POOLING_POINT = 75_000.0          # per-member annual cap (catches the genuine tail)
    POOLING_CHARGE_PMPM = 32.0        # expected cost above the pool (manual stop-loss load)
    MANUAL_CLAIMS_PMPM = 365.0        # book/manual expected claims PMPM (the complement)
    FULL_CREDIBILITY_MM = 12_000.0    # member-months for full credibility
    TARGET_LOSS_RATIO = 0.85          # claims as a share of premium
    EXPENSE_RATIO = 0.10              # admin / commission load as a share of premium

    df = sample_member_months()
    df["region"] = df["group_id"].map(lambda g: ("West", "Central", "East")[g % 3])
    member_months = df["member_months"].sum()
    exp = ap.Experience(df, expense="total_claims", revenue="premium",
                        exposure="member_months", date="incurred_date")

    section("0. Experience as filed (paid through valuation)")
    current_claims = df["total_claims"].sum()
    current_premium_pmpm = pmpm(df["premium"].sum(), member_months)
    print(f"member-months        : {member_months:,.0f}")
    print(f"incurred claims PMPM : {pmpm(current_claims, member_months):,.2f}")
    print(f"current premium PMPM : {current_premium_pmpm:,.2f}")
    print(f"current loss ratio   : {current_claims / df['premium'].sum():.3f}")

    section("1.-3. Complete -> trend -> relativities (frame-level, audited)")
    trend_f = ap.midpoint_trend_factor(BASE_MIDPOINT, RATING_MIDPOINT, ANNUAL_TREND)
    projected = (
        exp.complete(COMPLETION, valuation_date=VALUATION)                      # immature months to ultimate
           .adjust(trend_f, audit_col="restatement")                           # trend to rating midpoint
           .adjust(AREA, on="region", audit_col="restatement")                 # area relativity
           .adjust(BENEFIT, on="line_of_business", audit_col="restatement")    # benefit relativity
    )
    pdata = projected.data
    print(f"trend factor ({BASE_MIDPOINT} -> {RATING_MIDPOINT}, {ANNUAL_TREND:.0%}): {trend_f:.4f}")
    print(f"completed claims PMPM            : {pmpm(pdata['total_claims'].sum(), member_months):,.2f}")
    print(f"avg restatement multiplier/row   : {pdata['restatement'].mean():.4f}")

    section("4. Pool large claimants (per-member annual cap)")
    member_totals = pdata.groupby("member_id")["total_claims"].sum().reset_index()
    pooled = ap.pool_losses(member_totals, "total_claims", POOLING_POINT)
    pooled_claims = pooled["pooled_loss"].sum()
    excess = pooled["excess_loss"].sum()
    n_over = (pooled["excess_loss"] > 0).sum()
    print(f"claimants over ${POOLING_POINT:,.0f}: {n_over}  |  excess removed: ${excess:,.0f}")
    print(f"pooled claims PMPM               : {pmpm(pooled_claims, member_months):,.2f}")
    print(f"+ pooling charge (manual)        : {POOLING_CHARGE_PMPM:,.2f}")

    section("5. Credibility-blend pooled experience with the manual")
    group_pmpm = pmpm(pooled_claims, member_months)
    z = ap.limited_fluctuation_z(member_months, FULL_CREDIBILITY_MM)
    blended = ap.credibility_weighted_estimate(group_pmpm, MANUAL_CLAIMS_PMPM, z)
    print(f"group pooled PMPM {group_pmpm:,.2f}  vs  manual {MANUAL_CLAIMS_PMPM:,.2f}")
    print(f"credibility Z (member-months / {FULL_CREDIBILITY_MM:,.0f}): {z:.3f}")
    print(f"credibility-weighted PMPM        : {blended:,.2f}")
    projected_claims_pmpm = blended + POOLING_CHARGE_PMPM
    print(f"projected claims PMPM (+ pooling): {projected_claims_pmpm:,.2f}")

    section("6. Load for retention -> required premium -> indicated change")
    required_premium_pmpm = projected_claims_pmpm / TARGET_LOSS_RATIO
    admin_expense_pmpm = EXPENSE_RATIO * required_premium_pmpm
    profit_margin_pmpm = ap.margin(required_premium_pmpm, projected_claims_pmpm + admin_expense_pmpm)
    margin_pct = ap.margin_ratio(profit_margin_pmpm, required_premium_pmpm)
    indicated = ap.indicated_change(required_premium_pmpm, current_premium_pmpm)
    print(f"projected claims PMPM    : {projected_claims_pmpm:,.2f}  (target loss ratio {TARGET_LOSS_RATIO:.0%})")
    print(f"+ admin / commission     : {admin_expense_pmpm:,.2f}  ({EXPENSE_RATIO:.0%} of premium)")
    print(f"+ profit margin          : {profit_margin_pmpm:,.2f}  ({margin_pct:.1%} of premium)")
    print(f"required premium PMPM     : {required_premium_pmpm:,.2f}")
    print(f"current premium PMPM      : {current_premium_pmpm:,.2f}")
    print(f"\nINDICATED RATE CHANGE     : {indicated:+.1%}")


if __name__ == "__main__":
    main()
