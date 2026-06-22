import pandas as pd
import pytest

import actuarialpy as ap
from actuarialpy.components import component_driver_analysis
from actuarialpy.trend import trend_summary


def sample():
    return pd.DataFrame(
        {
            "group": ["A", "A", "A", "A"],
            "product": ["PPO", "PPO", "PPO", "PPO"],
            "incurred_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2026-01-01", "2026-02-01"]),
            "claims": [100, 100, 120, 130],
            "premium": [200, 200, 220, 230],
            "mm": [10, 10, 10, 10],
            "ip": [50, 50, 70, 80],
            "op": [50, 50, 50, 50],
        }
    )


def test_experience_facade_by_and_trend_date_range():
    exp = ap.Experience(sample(), expense="claims", revenue="premium", exposure="mm", date="incurred_date")
    by = exp.by("product", ratio_name="loss_ratio")
    assert by.loc[0, "loss_ratio"] == pytest.approx(450 / 850)

    trend = exp.trend(
        prior_start="2025-01-01",
        prior_end="2025-12-31",
        current_start="2026-01-01",
        current_end="2026-12-31",
        groupby="product",
    )
    assert trend.loc[0, "trend"] == pytest.approx((250 / 20) / (200 / 20) - 1)
    assert "prior_start" in trend.columns


def test_trend_summary_date_range():
    out = trend_summary(
        sample(),
        date_col="incurred_date",
        prior_start="2025-01-01",
        prior_end="2025-12-31",
        current_start="2026-01-01",
        current_end="2026-12-31",
        groupby="product",
        amount_col="claims",
        exposure_col="mm",
    )
    assert out.loc[0, "trend"] == pytest.approx(0.25)


def test_component_driver_date_range():
    out = component_driver_analysis(
        sample(),
        date_col="incurred_date",
        prior_start="2025-01-01",
        prior_end="2025-12-31",
        current_start="2026-01-01",
        current_end="2026-12-31",
        component_cols=["ip", "op"],
        exposure_col="mm",
    )
    ip = out[out["component"] == "ip"].iloc[0]
    assert ip["change"] == pytest.approx(2.5)
    assert "current_end" in out.columns


def test_experience_trend_period_col_with_bound_date_does_not_conflict_or_leak_year_columns():
    data = sample().copy()
    data["year"] = data["incurred_date"].dt.year
    exp = ap.Experience(data, expense="claims", revenue="premium", exposure="mm", date="incurred_date")
    out = exp.trend(period_col="year", prior_period=2025, current_period=2026, groupby="product")
    assert "year_x" not in out.columns
    assert "year_y" not in out.columns
    assert "prior_period" in out.columns
    assert out.loc[0, "trend"] == pytest.approx((250 / 20) / (200 / 20) - 1)
