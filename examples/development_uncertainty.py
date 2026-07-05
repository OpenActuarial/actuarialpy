"""Chain-ladder development with Mack standard errors.

    cumulative triangle -> ChainLadder.fit -> project (ultimate, IBNR)
    -> mack_sigma_squared -> mack_standard_errors (per origin + total)

The triangle is Taylor & Ashe (1983), the dataset Mack (1993) used --
so the Total row reproduces the published reserve 18,680,856 and
standard error 2,447,095.

Run with:  python examples/development_uncertainty.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from actuarialpy.reserving import ChainLadder

_TA = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085, None],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315, None, None],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268, None, None, None],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311, None, None, None, None],
    [396132, 1333217, 2180715, 2985752, 3691712, None, None, None, None, None],
    [440832, 1288463, 2419861, 3483130, None, None, None, None, None, None],
    [359480, 1421128, 2864498, None, None, None, None, None, None, None],
    [376686, 1363294, None, None, None, None, None, None, None, None],
    [344014, None, None, None, None, None, None, None, None, None],
]


def main() -> None:
    triangle = pd.DataFrame(
        _TA, index=pd.Index(range(1, 11), name="origin"), columns=range(1, 11)
    ).astype(float)

    cl = ChainLadder.fit(triangle)  # volume-weighted, Mack's estimator
    print("=== Age-to-age factors ===")
    print(cl.age_to_age.round(4).to_string())

    print("\n=== Variance parameters (sigma^2 per period) ===")
    print(cl.mack_sigma_squared(triangle).round(1).to_string())

    print("\n=== Ultimates and IBNR with Mack standard errors ===")
    out = cl.mack_standard_errors(triangle)
    with pd.option_context("display.float_format", "{:,.0f}".format):
        print(out[["latest", "ultimate", "ibnr", "se"]].to_string())
    print(f"\ntotal IBNR : {out.loc['Total', 'ibnr']:,.0f}")
    print(f"total s.e. : {out.loc['Total', 'se']:,.0f}"
          f"  (cv {out.loc['Total', 'cv']:.1%})")
    print("\nA reserve is an estimate; now it says how good an estimate.")
    assert np.isclose(out.loc["Total", "se"], 2_447_095, rtol=5e-3)


if __name__ == "__main__":
    main()
