"""Grouped development: each line by its own pattern, with its own bands.

    long (line, origin, valuation, paid) frame -> chain_ladder_by
    -> per-line ChainLadder -> mack_standard_errors per line

Pooling two lines with different volatility into one triangle averages
their patterns and hides their difference; fitting per line puts a CV on
each -- and the difference IS the finding.

Run with:  python examples/per_line_development.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from actuarialpy.reserving import ChainLadder, chain_ladder_by, make_completion_triangle


def _book(seed=21):
    rng = np.random.default_rng(seed)
    rows = []
    for line, sig in (("primary", 0.02), ("excess", 0.15)):
        for origin_year in range(2019, 2025):
            origin = pd.Timestamp(f"{origin_year}-01-01")
            c = float(rng.uniform(900.0, 1_100.0))
            rows.append((line, origin, origin, round(c, 2)))  # first payment
            for lag in range(1, 2025 - origin_year):
                factor = 1.6 if lag == 1 else 1.15 if lag == 2 else 1.05
                new_cum = c * factor * (1.0 + rng.normal(0.0, sig))
                rows.append((line, origin,
                             origin + pd.DateOffset(months=12 * lag),
                             round(new_cum - c, 2)))  # incremental paid
                c = new_cum
    return pd.DataFrame(rows, columns=["line", "origin", "valuation", "paid"])


def main() -> None:
    df = _book()
    patterns = chain_ladder_by(
        df, groupby="line", origin_col="origin",
        valuation_col="valuation", amount_col="paid",
    )
    for line, cl in patterns.items():
        tri = make_completion_triangle(
            df[df["line"] == line], origin_col="origin",
            valuation_col="valuation", amount_col="paid",
        )
        out = cl.mack_standard_errors(tri)
        print(f"=== {line} ===")
        print("age-to-age:", cl.age_to_age.round(3).tolist())
        with pd.option_context("display.float_format", "{:,.0f}".format):
            print(out.loc[["Total"], ["latest", "ultimate", "ibnr", "se"]]
                  .to_string())
        print(f"total CV: {out.loc['Total', 'cv']:.1%}\n")
    print("Same development story, very different confidence -- and a pooled")
    print("triangle would have reported one blended CV that is true of neither.")


if __name__ == "__main__":
    main()
