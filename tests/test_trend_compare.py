import pandas as pd
import pytest

from actuarialpy.compare import basis_point_change, variance_pct
from actuarialpy.trend import (
    annualized_trend,
    midpoint_trend_factor,
    period_change,
    project_forward,
    trend_factor,
    trend_summary,
)


def test_compare_trend_primitives():
    assert basis_point_change(0.88, 0.85) == pytest.approx(300)
    assert variance_pct(110, 100) == pytest.approx(0.10)
    assert period_change(110, 100) == pytest.approx(0.10)
    assert round(annualized_trend(121, 100, 24), 2) == 0.10
    assert trend_factor(0.10, 12) == pytest.approx(1.10)
    assert project_forward(100, 0.10, 12) == pytest.approx(110)
    assert midpoint_trend_factor("2025-01-01", "2026-01-01", 0.1) == pytest.approx(1.1)


def test_trend_summary_period_api():
    df = pd.DataFrame({"product": ["PPO", "PPO"], "year": [2025, 2026], "claims": [100, 121], "mm": [10, 11]})
    out = trend_summary(df, groupby="product", period_col="year", prior_period=2025, current_period=2026, amount_col="claims", exposure_col="mm")
    assert out.loc[0, "trend"] == pytest.approx((121 / 11) / (100 / 10) - 1)
    assert out.loc[0, "prior_period"] == 2025


def test_trend_summary_filter_api_still_available():
    df = pd.DataFrame({"product": ["PPO", "PPO"], "year": [2025, 2026], "claims": [100, 121], "mm": [10, 11]})
    out = trend_summary(df, groupby="product", prior_filter=df["year"] == 2025, current_filter=df["year"] == 2026, amount_col="claims", exposure_col="mm")
    assert out.loc[0, "trend"] == pytest.approx((121 / 11) / (100 / 10) - 1)


def test_trend_summary_period_does_not_leak_numeric_period_column():
    df = pd.DataFrame(
        {
            "lob": ["A", "A", "B", "B"],
            "year": [2025, 2026, 2025, 2026],
            "total_expense": [903356.13, 3371625.00, 798382.04, 3349350.00],
            "member_months": [1665, 6214, 1654, 6939],
        }
    )
    out = trend_summary(
        df,
        groupby="lob",
        period_col="year",
        prior_period=2025,
        current_period=2026,
        amount_col="total_expense",
        exposure_col="member_months",
    )
    assert "year_x" not in out.columns
    assert "year_y" not in out.columns
    assert "current_total_expense" in out.columns
    assert out.loc[out["lob"] == "A", "current_total_expense"].iloc[0] == pytest.approx(3371625.00)
