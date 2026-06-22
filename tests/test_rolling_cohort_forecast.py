import pandas as pd
import pytest

from actuarialpy.cohorts import cohort_summary, cohort_summary_by_period, duration_summary
from actuarialpy.forecast import compare_actual_to_expected, expected_from_rate, forecast_experience, forecast_from_rate
from actuarialpy.rolling import rolling_summary


def monthly_df():
    return pd.DataFrame({
        "group_id": ["G1"] * 4,
        "month": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]),
        "effective_date": pd.to_datetime(["2026-01-01"] * 4),
        "claims": [100, 200, 300, 400],
        "premium": [200, 200, 200, 200],
        "member_months": [10, 10, 10, 10],
    })


def test_rolling_summary_full_windows_default():
    out = rolling_summary(monthly_df(), date_col="month", window=3, groupby="group_id", expense_cols="claims", revenue_cols="premium", exposure_cols="member_months")
    assert len(out) == 2
    first = out.iloc[0]
    assert first["period_start"] == pd.Timestamp("2026-01-01")
    assert first["period_end"] == pd.Timestamp("2026-03-01")
    last = out.iloc[-1]
    assert last["total_expense"] == 900
    assert last["mlr"] == 900 / 600
    assert "months_available" not in out.columns


def test_rolling_summary_partial_windows_allowed():
    out = rolling_summary(monthly_df(), date_col="month", window=3, groupby="group_id", expense_cols="claims", revenue_cols="premium", exposure_cols="member_months", min_periods=1, drop_incomplete=False)
    assert len(out) == 4
    assert out.loc[0, "months_available"] == 1


def test_cohort_and_duration():
    cohort = cohort_summary(monthly_df(), entity_col="group_id", date_col="month", start_date_col="effective_date", duration_months=3, expense_cols="claims", revenue_cols="premium", exposure_cols="member_months")
    assert cohort.loc[0, "total_expense"] == 600
    by_q = cohort_summary_by_period(cohort, entity_col="group_id", exposure_cols="member_months")
    assert by_q.loc[0, "entity_count"] == 1
    dur = duration_summary(monthly_df(), entity_col="group_id", date_col="month", start_date_col="effective_date", expense_cols="claims", revenue_cols="premium", max_duration_month=2)
    assert dur["duration_month"].tolist() == [1, 2]


def test_forecast_helpers():
    assert expected_from_rate(500, 10) == 5000
    assert forecast_from_rate(500, 10, annual_trend=0.1, months_forward=12) == pytest.approx(5500)
    df = pd.DataFrame({"rate": [500], "exposure": [10], "months": [12]})
    forecast = forecast_experience(df, rate_col="rate", exposure_col="exposure", annual_trend=0.1, months_forward="months")
    assert forecast.loc[0, "expected_expense"] == pytest.approx(5500)

    actual = pd.DataFrame({"id": [1], "actual": [6000]})
    expected = pd.DataFrame({"id": [1], "expected": [5500]})
    comp = compare_actual_to_expected(actual, expected, on="id", actual_col="actual", expected_col="expected")
    assert comp.loc[0, "actual_to_expected"] == pytest.approx(6000 / 5500)
