"""Seasonality: working-day counts and seasonal factors.

Count working days per month, learn one seasonal multiplier per calendar month
from multi-year history, deseasonalize recent claims so the underlying trend is
visible, then fit factors per line of business and deseasonalize each line by its
own pattern in a single call.

    pip install actuarialpy
    python seasonality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from _sample_data import sample_seasonal_panel  # noqa: E402

import actuarialpy as ap  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    panel = sample_seasonal_panel()

    section("1. business_days_in_period: weekdays minus US federal holidays")
    months_2024 = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    bd = ap.business_days_in_period(months_2024)
    print(bd.to_frame("business_days").to_string())

    section("2. seasonality_factors: one multiplier per calendar month (PMPM basis)")
    # Estimated on the whole book; seasonality_factors aggregates across line and
    # product to the month grain and works on the rate (claims / member_months).
    factors = ap.seasonality_factors(
        panel, date_col="month", value_col="claims", exposure_col="member_months",
    )
    print(factors.round(4).to_string())

    section("3. deseasonalize: recent claims with the seasonal pattern removed")
    pool = (
        panel.groupby("month", as_index=False)
        .agg(claims=("claims", "sum"), member_months=("member_months", "sum"))
    )
    recent = pool.tail(6).copy()
    clean = ap.deseasonalize(recent, factors, date_col="month", value_col="claims")
    clean["pmpm_raw"] = clean["claims"] / clean["member_months"]
    clean["pmpm_deseasonalized"] = clean["claims_deseasonalized"] / clean["member_months"]
    view = clean.assign(month=clean["month"].dt.strftime("%Y-%m"))[
        ["month", "pmpm_raw", "pmpm_deseasonalized"]
    ]
    print(view.to_string(index=False, formatters={
        "pmpm_raw": "{:,.2f}".format, "pmpm_deseasonalized": "{:,.2f}".format,
    }))
    print("(raw PMPM swings with winter; deseasonalized sits on the underlying trend)")

    section("4. seasonality_factors_by + grouped deseasonalize(by=)")
    sf_by_lob = ap.seasonality_factors_by(
        panel, groupby="line_of_business",
        date_col="month", value_col="claims", exposure_col="member_months",
    )
    winter_summer = (
        sf_by_lob[sf_by_lob["season"].isin([7, 12])]
        .pivot(index="line_of_business", columns="season", values="seasonal_factor")
        .rename(columns={7: "Jul", 12: "Dec"})
    )
    print("seasonal factor, summer vs winter, by line (A swings, B is mild):")
    print(winter_summer.round(3).to_string())

    # Develop the deseasonalization across the whole panel, each line by its own
    # factors, in one call -- joined on (line_of_business, season).
    deseasonalized = ap.deseasonalize(
        panel, sf_by_lob, date_col="month", value_col="claims", by="line_of_business",
    )
    y2024 = deseasonalized[deseasonalized["month"].dt.year == 2024].copy()
    by_month = (
        y2024.groupby(["line_of_business", y2024["month"].dt.month])
        .agg(claims=("claims", "sum"),
             des=("claims_deseasonalized", "sum"),
             mm=("member_months", "sum"))
    )
    by_month["pmpm_raw"] = by_month["claims"] / by_month["mm"]
    by_month["pmpm_des"] = by_month["des"] / by_month["mm"]
    swing = pd.DataFrame({
        "raw max/min": by_month.groupby(level=0)["pmpm_raw"].agg(lambda s: s.max() / s.min()),
        "deseasonalized max/min": by_month.groupby(level=0)["pmpm_des"].agg(lambda s: s.max() / s.min()),
    })
    print("\nwithin-year PMPM swing (max month / min month) in 2024, by line:")
    print(swing.round(3).to_string())


if __name__ == "__main__":
    main()
