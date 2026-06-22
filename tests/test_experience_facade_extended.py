import pandas as pd
import pytest

import actuarialpy as ap
from actuarialpy.claimants import claim_concentration, summarize_claimants, top_claimants
from actuarialpy.expected import summarize_actual_vs_expected


def sample_df():
    return pd.DataFrame(
        {
            "incurred_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-01",
                    "2025-02-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-02-01",
                ]
            ),
            "group_id": ["A", "A", "B", "A", "A", "B"],
            "product_code": ["P1", "P1", "P2", "P1", "P1", "P2"],
            "member_id": ["m1", "m2", "m3", "m1", "m4", "m3"],
            "claims": [100, 200, 300, 120, 500, 360],
            "premium": [200, 300, 600, 240, 600, 720],
            "expected_claims": [90, 210, 300, 100, 450, 400],
            "member_months": [1, 1, 1, 1, 1, 1],
            "start_date": pd.to_datetime(["2025-01-01"] * 6),
        }
    )


def test_experience_filter_returns_new_experience():
    exp = ap.Experience(sample_df(), expense="claims", revenue="premium", exposure="member_months", date="incurred_date")
    filtered = exp.filter(query="group_id == 'A'")
    assert isinstance(filtered, ap.Experience)
    assert len(filtered.data) == 4
    out = filtered.by("group_id")
    assert out.loc[0, "total_expense"] == 920


def test_experience_validates_id_like_exposure():
    with pytest.raises(ValueError, match="Exposure columns"):
        ap.Experience(sample_df(), expense="claims", revenue="premium", exposure="member_id", date="incurred_date")


def test_actual_vs_expected_free_function_and_facade():
    df = sample_df()
    free = summarize_actual_vs_expected(
        df,
        groupby="product_code",
        actual_cols="claims",
        expected_cols="expected_claims",
        exposure_cols="member_months",
    )
    assert set(["actual", "expected", "actual_to_expected", "variance", "actual_pmpm"]).issubset(free.columns)

    exp = ap.Experience(df, expense="claims", revenue="premium", exposure="member_months", date="incurred_date")
    facade = exp.actual_vs_expected(expected="expected_claims", groupby="product_code")
    p1 = facade.loc[facade["product_code"] == "P1"].iloc[0]
    assert p1["actual"] == 920
    assert p1["expected"] == 850


def test_claimants_free_functions_and_facade():
    df = sample_df()
    claimants = summarize_claimants(df, claimant_col="member_id", amount_cols="claims", groupby="group_id")
    assert {"member_id", "total_expense"}.issubset(claimants.columns)

    top = top_claimants(df, claimant_col="member_id", amount_cols="claims", groupby="group_id", n=1)
    assert (top.groupby("group_id").size() <= 1).all()
    assert {"rank", "share_of_total", "cumulative_share"}.issubset(top.columns)

    concentration = claim_concentration(claimants, groupby="group_id", thresholds=(250,))
    assert "top_10_share" in concentration.columns
    assert "count_over_250" in concentration.columns

    exp = ap.Experience(df, expense="claims", revenue="premium", exposure="member_months", date="incurred_date")
    facade_top = exp.top_claimants("member_id", groupby="group_id", n=1)
    assert (facade_top.groupby("group_id").size() <= 1).all()


def test_cohort_duration_facade_methods():
    df = sample_df()
    exp = ap.Experience(df, expense="claims", revenue="premium", exposure="member_months", date="incurred_date")
    cohort = exp.cohort(entity_col="group_id", start_date_col="start_date", duration_months=12)
    duration = exp.duration(entity_col="group_id", start_date_col="start_date", max_duration_month=14)
    assert "total_expense" in cohort.columns
    assert "duration_month" in duration.columns
