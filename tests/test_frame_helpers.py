import pandas as pd

from actuarialpy import Experience
from actuarialpy.completion import completed_experience
from actuarialpy.experience import summarize_experience
from actuarialpy.rolling import rolling_summary
from actuarialpy.trend import trend_summary


def sample():
    return pd.DataFrame({
        "lob": ["A", "A", "B", "B"],
        "year": [2025, 2026, 2025, 2026],
        "claims": [100, 120, 200, 260],
        "premium": [200, 200, 400, 400],
        "mm": [10, 10, 20, 20],
    })


def monthly():
    return pd.DataFrame({
        "group_id": ["G"] * 4,
        "month": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]),
        "claims": [100, 100, 100, 100],
        "premium": [200, 200, 200, 200],
        "mm": [10, 10, 10, 10],
    })


# --------------------------------------------------------------------------- #
# Experience facade delegates identically to the free functions
# --------------------------------------------------------------------------- #
def test_experience_by_matches_free_function():
    df = sample()
    exp = Experience(df, expense="claims", revenue="premium", exposure="mm")
    direct = summarize_experience(df, groupby="lob", expense_cols="claims", revenue_cols="premium", exposure_cols="mm")
    pd.testing.assert_frame_equal(exp.by("lob"), direct)


def test_experience_rolling_matches_free_function():
    df = monthly()
    exp = Experience(df, expense="claims", revenue="premium", exposure="mm")
    direct = rolling_summary(
        df, date_col="month", window=3, groupby="group_id",
        expense_cols="claims", revenue_cols="premium", exposure_cols="mm", freq="MS",
    )
    pd.testing.assert_frame_equal(exp.rolling(3, date_col="month", groupby="group_id", freq="MS"), direct)


def test_experience_profile_and_per_call_override():
    df = sample()
    exp = Experience(df, expense="claims", revenue="premium", exposure="mm", profile="health")
    assert "mlr" in exp.by("lob").columns
    # override the bound profile back to general for one call
    out = exp.by("lob", profile=None)
    assert "loss_ratio" in out.columns
    assert "mlr" not in out.columns


def test_experience_expense_override():
    df = sample().assign(other=[1, 2, 3, 4])
    exp = Experience(df, expense="claims", revenue="premium")
    overridden = exp.by("lob", expense="other")
    direct = summarize_experience(df, groupby="lob", expense_cols="other", revenue_cols="premium")
    pd.testing.assert_frame_equal(overridden, direct)


def test_experience_repr():
    assert "Experience(" in repr(Experience(sample(), expense="claims", revenue="premium"))


# --------------------------------------------------------------------------- #
# trend_summary no longer leaks the period column
# --------------------------------------------------------------------------- #
def test_trend_no_period_column_leak():
    out = trend_summary(
        sample(), period_col="year", prior_period=2025, current_period=2026,
        groupby="lob", amount_col="claims", exposure_col="mm",
    )
    assert "year_x" not in out.columns
    assert "year_y" not in out.columns
    assert "year" not in out.columns
    a = out[out["lob"] == "A"].iloc[0]
    assert abs(a["trend"] - 0.2) < 1e-9  # 12/10 vs 10/10 -> +20%


def test_experience_trend_delegates_and_is_clean():
    exp = Experience(sample(), expense="claims", revenue="premium", exposure="mm")
    out = exp.trend(amount_col="claims", period_col="year", prior_period=2025, current_period=2026, groupby="lob")
    assert "year_x" not in out.columns
    b = out[out["lob"] == "B"].iloc[0]
    assert abs(b["trend"] - 0.3) < 1e-9  # 13/10 vs 10/10 -> +30%


# --------------------------------------------------------------------------- #
# completed_experience: completion -> experience in one call
# --------------------------------------------------------------------------- #
def test_completed_experience():
    df = pd.DataFrame({
        "lob": ["A", "B"],
        "inpatient_claims": [90, 180],
        "factor": [0.9, 0.9],
        "rebates": [-10, -20],
        "premium": [200, 400],
        "mm": [10, 20],
    })
    out = completed_experience(
        df,
        component_factor_map={"inpatient_claims": "factor"},
        revenue_cols="premium",
        groupby="lob",
        exposure_cols="mm",
        additional_expense_cols="rebates",
    )
    by_lob = out.set_index("lob")
    # A: completed 90/0.9=100, plus rebate -10 -> expense 90; LR 90/200 = 0.45
    assert by_lob.loc["A", "total_expense"] == 90
    assert by_lob.loc["A", "loss_ratio"] == 0.45
    assert by_lob.loc["B", "loss_ratio"] == 0.45
