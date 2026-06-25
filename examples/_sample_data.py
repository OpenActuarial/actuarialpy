"""Synthetic sample data for the examples.

Small, deterministic generators so every example is self-contained and runs with
nothing but ``pip install actuarialpy``:

* ``sample_member_months`` -- a member-month experience frame (one row per member
  per incurred month) with claim components, rebates, non-FFS expense, premium,
  per-group effective/termination dates, and a group subscriber count. This is
  the grain ``actuarialpy`` is built around.
* ``sample_claim_payments`` -- a long (line of business, origin, valuation,
  incremental-paid) frame for the reserving / completion-factor example, with a
  distinct payment pattern per line of business.
* ``sample_seasonal_panel`` -- a monthly claims panel by line of business and
  product spanning four years, with a real month-of-year seasonal pattern,
  membership growth, and an underlying cost trend, for the seasonality example.

None of this is part of the library; it exists only to feed the examples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOBS = ("A", "B", "C")
CLAIM_COMPONENTS = ("inpatient_claims", "outpatient_claims", "professional_claims", "pharmacy_claims")


def sample_member_months(seed: int = 7) -> pd.DataFrame:
    """Return a small member-month experience frame spanning 2024-2025."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")

    rows: list[dict] = []
    group_id = 1000
    for g in range(6):
        group_id += int(rng.integers(40, 280))
        lob = LOBS[g % len(LOBS)]
        n_members = int(rng.integers(8, 22))
        subscriber_count = n_members

        group_eff = pd.Timestamp("2024-01-01") + pd.DateOffset(months=int(rng.integers(0, 6)))
        group_term = pd.NaT
        if rng.random() < 0.30:
            group_term = group_eff + pd.DateOffset(months=int(rng.integers(12, 22)))

        for m in range(n_members):
            member_id = group_id * 100 + m
            member_eff = group_eff + pd.DateOffset(months=int(rng.integers(0, 4)))
            member_term = group_term
            base = float(rng.lognormal(mean=np.log(350.0), sigma=0.5))  # monthly claim level

            for d in months:
                if d < member_eff:
                    continue
                if pd.notna(member_term) and d >= member_term:
                    continue

                trend = 1.0 if d.year == 2024 else 1.07
                lam = base * trend

                inpatient = float(rng.gamma(0.2, lam * 3.0)) if rng.random() < 0.08 else 0.0
                # rare catastrophic inpatient month -> creates large annual claimants
                if rng.random() < 0.010:
                    inpatient += float(rng.gamma(2.0, 22_000.0))
                outpatient = float(rng.gamma(1.5, lam * 0.30))
                professional = float(rng.gamma(1.2, lam * 0.30))
                pharmacy = float(rng.gamma(1.0, lam * 0.25))
                total = inpatient + outpatient + professional + pharmacy

                rows.append(
                    {
                        "incurred_date": d,
                        "group_id": group_id,
                        "line_of_business": lob,
                        "member_id": member_id,
                        "inpatient_claims": round(inpatient, 2),
                        "outpatient_claims": round(outpatient, 2),
                        "professional_claims": round(professional, 2),
                        "pharmacy_claims": round(pharmacy, 2),
                        "total_claims": round(total, 2),
                        "pharmacy_rebates": round(-0.10 * pharmacy, 2),
                        "non_ffs_expenses": round(float(rng.gamma(2.0, 8.0)), 2),
                        "premium": round(float(rng.normal(520.0, 40.0)) * trend, 2),
                        "group_subscriber_count": subscriber_count,
                        "effective_date": member_eff,
                        "termination_date": member_term,
                        "member_months": 1,
                    }
                )

    df = pd.DataFrame(rows)
    df["year"] = df["incurred_date"].dt.year
    return df


def sample_claim_payments(seed: int = 7) -> pd.DataFrame:
    """Return a long (line of business, origin, valuation, incremental-paid) frame.

    Twelve monthly origin periods per line of business develop over up to twelve
    development periods, each line following its own payment pattern (B settles
    faster, C slower), truncated into a triangle at the latest valuation.
    """
    rng = np.random.default_rng(seed)
    patterns = {
        "A": np.array([0.34, 0.22, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002]),
        "B": np.array([0.45, 0.25, 0.13, 0.07, 0.04, 0.025, 0.015, 0.01, 0.005, 0.003, 0.001, 0.001]),
        "C": np.array([0.25, 0.18, 0.15, 0.12, 0.09, 0.07, 0.05, 0.04, 0.025, 0.015, 0.007, 0.003]),
    }
    origins = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    latest = origins[-1]

    rows: list[dict] = []
    for lob, pattern in patterns.items():
        pattern = pattern / pattern.sum()
        for origin in origins:
            ultimate = float(rng.normal(1_000_000, 120_000))
            incremental = ultimate * pattern * rng.normal(1.0, 0.05, size=pattern.size)
            for dev, inc in enumerate(incremental):
                valuation = origin + pd.DateOffset(months=dev)
                if valuation > latest:
                    break  # not yet observed -> triangle shape
                rows.append(
                    {
                        "line_of_business": lob,
                        "origin_month": origin,
                        "valuation_month": valuation,
                        "paid": round(float(inc), 2),
                    }
                )
    return pd.DataFrame(rows)


def sample_seasonal_panel(seed: int = 7) -> pd.DataFrame:
    """Return a monthly claims panel by line of business and product (2021-2024).

    Each line of business carries its own month-of-year seasonal shape (A swings
    hard with winter, B is mild), on top of steady membership growth and an
    underlying cost trend. Columns: ``line_of_business``, ``product``, ``month``,
    ``claims``, ``member_months``, ``premium``. The seasonal pattern is real, so
    ``seasonality_factors`` recovers it and deseasonalizing flattens it out.
    """
    rng = np.random.default_rng(seed)
    shapes = {
        "A": np.array([1.25, 1.16, 1.06, 0.97, 0.92, 0.87, 0.85, 0.88, 0.96, 1.03, 1.07, 1.18]),
        "B": np.array([1.06, 1.05, 1.02, 1.00, 0.99, 0.98, 0.97, 0.98, 1.00, 1.01, 1.02, 1.04]),
    }
    months = pd.date_range("2021-01-01", "2024-12-01", freq="MS")

    rows: list[dict] = []
    for lob, shape in shapes.items():
        shape = shape / shape.mean()
        members0 = 9000 if lob == "A" else 6000
        for product in ("PPO", "HMO"):
            base_pmpm = 320.0 if product == "PPO" else 285.0
            product_scale = 1.0 if product == "PPO" else 0.7
            for i, d in enumerate(months):
                member_months = int(round(members0 * (1.004 ** i) * product_scale))
                pmpm = base_pmpm * (1.004 ** i) * shape[d.month - 1] * float(rng.normal(1.0, 0.01))
                claims = pmpm * member_months
                rows.append(
                    {
                        "line_of_business": lob,
                        "product": product,
                        "month": d,
                        "claims": round(claims, 2),
                        "member_months": member_months,
                        "premium": round(claims * 1.18, 2),
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # quick smoke check
    mm = sample_member_months()
    print("member_months:", mm.shape, "| members:", mm["member_id"].nunique(), "| groups:", mm["group_id"].nunique())
    print("annual claimants > $50k:",
          (mm.groupby("member_id")["total_claims"].sum() > 50_000).sum())
    pay = sample_claim_payments()
    print("payment rows:", pay.shape, "| origins:", pay["origin_month"].nunique(),
          "| lines of business:", sorted(pay["line_of_business"].unique()))
    panel = sample_seasonal_panel()
    print("seasonal panel:", panel.shape, "| months:", panel["month"].nunique(),
          "| segments:", panel.groupby(["line_of_business", "product"]).ngroups)


def sample_trend_cells() -> pd.DataFrame:
    """Two-period claims panel by morbidity segment and region, for trend decomposition.

    A deterministic prior/current snapshot (labelled 2024 and 2025) split into
    segment (Low/High morbidity) x region (North/South) cells -- a clean member-level
    partition, so member months add up across cells without double counting. Within
    every cell utilization trends +3% and unit cost +4% (uniform, so the "true"
    within-cell trends are obvious); meanwhile enrollment shifts toward the High
    segment, so the book-wide two-way overstates utilization and unit cost while the
    mix term recovers the difference. Columns: ``period``, ``segment``, ``region``,
    ``member_months``, ``claim_count``, ``allowed``, ``premium``.
    """
    prior = {                                  # (segment, region): (member_months, utilization, unit_cost)
        ("Low", "North"): (40000, 0.45, 420.0),
        ("Low", "South"): (24000, 0.45, 440.0),
        ("High", "North"): (20000, 0.95, 820.0),
        ("High", "South"): (16000, 0.95, 860.0),
    }
    current_mm = {                             # enrollment tilts toward High (and slightly South)
        ("Low", "North"): 33000,
        ("Low", "South"): 22000,
        ("High", "North"): 24000,
        ("High", "South"): 21000,
    }
    util_trend, cost_trend = 1.03, 1.04        # uniform within every cell
    rows: list[dict] = []
    for (seg, reg), (mm0, u0, c0) in prior.items():
        allowed0 = c0 * u0 * mm0
        rows.append({"period": "2024", "segment": seg, "region": reg, "member_months": float(mm0),
                     "claim_count": u0 * mm0, "allowed": allowed0, "premium": 1.18 * allowed0})
        mm1 = current_mm[(seg, reg)]
        u1, c1 = u0 * util_trend, c0 * cost_trend
        allowed1 = c1 * u1 * mm1
        rows.append({"period": "2025", "segment": seg, "region": reg, "member_months": float(mm1),
                     "claim_count": u1 * mm1, "allowed": allowed1, "premium": 1.18 * allowed1})
    return pd.DataFrame(rows)
