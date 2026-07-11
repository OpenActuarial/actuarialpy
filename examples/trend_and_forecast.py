"""Trend and forecast: measure prior-vs-current trend and project forward.

Compare a prior period to a current period on an exposure-adjusted basis with
``trend_summary``, then turn the observed change into an annualized trend and
project a base per-exposure rate forward with ``project_forward`` (PMPM in this
health-flavored walkthrough).

    pip install actuarialpy
    python trend_and_forecast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sample_data import sample_member_months, sample_seasonal_panel  # noqa: E402

import actuarialpy as ap  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    df = sample_member_months()

    section("1. trend_summary: 2024 vs 2025 claims PMPM by LOB")
    by_lob = ap.trend_summary(
        df,
        period_col="year",
        prior_period=2024,
        current_period=2025,
        amount_col="total_claims",
        exposure_col="member_months",
        groupby="line_of_business",
    )
    print(by_lob.to_string(index=False))

    section("2. trend_summary: book total (no groupby)")
    book = ap.trend_summary(
        df,
        period_col="year",
        prior_period=2024,
        current_period=2025,
        amount_col="total_claims",
        exposure_col="member_months",
    )
    print(book.to_string(index=False))

    section("3. annualize the observed change and project 12 months forward")
    prior_pmpm = float(book["prior_total_claims_per_member_months"].iloc[0])
    current_pmpm = float(book["current_total_claims_per_member_months"].iloc[0])
    annual = ap.annualized_trend(current_pmpm, prior_pmpm, months_between=12)
    projected = ap.project_forward(current_pmpm, annual, months=12)

    print(f"prior PMPM (2024)      : {prior_pmpm:,.2f}")
    print(f"current PMPM (2025)    : {current_pmpm:,.2f}")
    print(f"trend_summary 'trend'  : {float(book['trend'].iloc[0]):.3%}")
    print(f"annualized trend       : {annual:.3%}")
    print(f"12-month trend factor  : {ap.trend_factor(annual, 12):.4f}")
    print(f"projected PMPM (+12mo) : {projected:,.2f}")

    section("4. fit_trend: develop the trend from history (deseasonalize first)")
    # A monthly panel with a real underlying trend plus seasonality; pool to one series.
    panel = sample_seasonal_panel()
    monthly = (
        panel.groupby("month")
        .agg(claims=("claims", "sum"), member_months=("member_months", "sum"))
        .reset_index()
    )
    raw = ap.fit_trend(monthly, value_col="claims", date_col="month", exposure_col="member_months")
    # the recommended pipeline runs the fit on deseasonalized history
    sf = ap.seasonality_factors(panel, date_col="month", value_col="claims", exposure_col="member_months")
    clean = ap.deseasonalize(monthly, sf, date_col="month", value_col="claims")
    fit = ap.fit_trend(clean, value_col="claims_deseasonalized", date_col="month", exposure_col="member_months")
    print(f"raw series       : trend {raw.annual_trend:.2%}, R^2 {raw.r_squared:.3f}   (seasonality adds scatter)")
    print(f"deseasonalized   : trend {fit.annual_trend:.2%}, R^2 {fit.r_squared:.3f}   (sharper fit)")
    print(f"{int(fit.confidence * 100)}% CI         : [{fit.ci_low:.2%}, {fit.ci_high:.2%}]  (n={fit.n_periods})")
    print(f"18-month factor  : {fit.factor(18):.4f}")
    print(repr(fit))

    section("5. why a fit, not two points: robustness to one odd month")
    pmpm = (clean["claims_deseasonalized"] / clean["member_months"]).to_numpy()
    spiked_pmpm = pmpm.copy()
    spiked_pmpm[-1] *= 1.25  # the latest month comes in 25% high
    spiked = clean.assign(claims_deseasonalized=spiked_pmpm * clean["member_months"])
    two_point = ap.annualized_trend(spiked_pmpm[-1], spiked_pmpm[0], months_between=len(pmpm) - 1)
    fit_spiked = ap.fit_trend(spiked, value_col="claims_deseasonalized", date_col="month", exposure_col="member_months")
    print(f"clean fitted trend         : {fit.annual_trend:.2%}")
    print("after one +25% month:")
    print(f"  two-point annualized     : {two_point:.2%}   <- swings hard on the endpoint")
    print(f"  fit_trend (full series)  : {fit_spiked.annual_trend:.2%}   <- barely moves")


if __name__ == "__main__":
    main()
