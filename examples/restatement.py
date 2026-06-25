"""Restatement: carry base claims through a chain of factors with an audit trail.

The everyday experience-rating move -- develop to ultimate, then apply trend and a
few relativities -- as one composable chain on the Experience object. ``adjust`` joins
each factor by its key (a scalar, a per-key Series, or a per-segment table) and an
``audit_col`` records the cumulative restatement applied to every row.

    pip install actuarialpy
    python restatement.py
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


def main() -> None:
    df = sample_member_months()
    # a couple of rating dimensions to key relativities on
    df["region"] = df["group_id"].map(lambda g: ("West", "Central", "East")[g % 3])
    exp = ap.Experience(
        df, expense="total_claims", revenue="premium",
        exposure="member_months", date="incurred_date",
    )

    section("Factors to apply (supplied as input -- the library does not derive them)")
    trend = 1.072  # one annual trend factor
    area = pd.Series({"West": 1.06, "Central": 1.00, "East": 0.95}, name="factor")  # by region
    benefit = pd.Series({"A": 1.04, "B": 0.98, "C": 1.09}, name="factor")           # by line of business
    print(f"trend (scalar)        : {trend}")
    print(f"area (by region)      : {area.to_dict()}")
    print(f"benefit (by line)     : {benefit.to_dict()}")

    section("Restate claims as one chain, accumulating the audit multiplier")
    restated = (
        exp.adjust(trend, audit_col="restatement")
           .adjust(area, on="region", audit_col="restatement")
           .adjust(benefit, on="line_of_business", audit_col="restatement")
    )

    # the per-row restatement multiplier is just trend * area * benefit
    check = restated.data.assign(
        expected=trend
        * restated.data["region"].map(area)
        * restated.data["line_of_business"].map(benefit)
    )
    matches = (check["restatement"] - check["expected"]).abs().lt(1e-9).all()
    print(f"audit multiplier == trend * area * benefit for every row: {matches}")

    section("Effect on the book: claims and loss ratio, before vs after")
    base_claims = df["total_claims"].sum()
    restated_claims = restated.data["total_claims"].sum()
    base_lr = base_claims / df["premium"].sum()
    restated_lr = restated_claims / restated.data["premium"].sum()
    print(f"claims     : ${base_claims:,.0f}  ->  ${restated_claims:,.0f}"
          f"  ({restated_claims / base_claims - 1:+.1%})")
    print(f"loss ratio : {base_lr:.3f}  ->  {restated_lr:.3f}")

    section("Restated loss ratio by line of business")
    by_lob = restated.by(groupby="line_of_business")
    cols = [c for c in ("line_of_business", "total_claims", "premium", "loss_ratio") if c in by_lob.columns]
    print(by_lob[cols].to_string(index=False, formatters={
        "total_claims": "{:,.0f}".format, "premium": "{:,.0f}".format, "loss_ratio": "{:.3f}".format,
    }))


if __name__ == "__main__":
    main()
