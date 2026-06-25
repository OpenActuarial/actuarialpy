"""Tests for cohort coverage columns (months_observed, last_month, complete)."""

import pandas as pd

from actuarialpy.cohorts import cohort_summary


def _book(as_of="2026-06-01", joins=None):
    joins = joins or {"old": "2024-01-01", "mid": "2025-09-01", "new": "2026-01-01"}
    as_of = pd.Timestamp(as_of)
    rows = []
    for g, start in joins.items():
        for m in pd.date_range(start, as_of, freq="MS"):
            rows.append(dict(group=g, eff=start, month=m, claims=100, premium=120, member_months=500))
    df = pd.DataFrame(rows)
    df["eff"] = pd.to_datetime(df["eff"])
    df["month"] = pd.to_datetime(df["month"])
    return df


def _cohort(df, **kw):
    return cohort_summary(df, entity_col="group", date_col="month", start_date_col="eff",
                          expense_cols="claims", revenue_cols="premium",
                          exposure_cols="member_months", **kw)


def test_coverage_columns_present():
    cy = _cohort(_book())
    assert {"months_observed", "last_month", "complete"}.issubset(cy.columns)


def test_complete_true_for_full_window():
    cy = _cohort(_book()).set_index("group")
    assert bool(cy.loc["old", "complete"]) is True
    assert cy.loc["old", "months_observed"] == 12


def test_complete_false_for_partial_window():
    cy = _cohort(_book()).set_index("group")
    assert bool(cy.loc["new", "complete"]) is False   # joined 2026-01, only 6 months by as-of
    assert cy.loc["new", "months_observed"] == 6


def test_last_month_gives_range_end():
    cy = _cohort(_book()).set_index("group")
    assert str(cy.loc["new", "last_month"]) == "2026-06"
    assert str(cy.loc["old", "last_month"]) == "2024-12"  # clipped to first year


def test_complete_filter_keeps_only_mature():
    cy = _cohort(_book())
    mature = cy[cy["complete"]]
    assert sorted(mature["group"]) == ["old"]


def test_duration_months_changes_completeness():
    cy = _cohort(_book(), duration_months=6).set_index("group")
    # with a 6-month window, the "new" group (6 months) is now complete
    assert bool(cy.loc["new", "complete"]) is True
    assert cy.loc["new", "months_observed"] == 6


def test_gap_in_data_marks_incomplete():
    df = _book(joins={"g": "2024-01-01"})
    df = df[~(df["month"] == pd.Timestamp("2024-03-01"))]  # drop one interior month
    cy = _cohort(df[df["month"] <= pd.Timestamp("2024-12-01")]).set_index("group")
    assert cy.loc["g", "months_observed"] == 11
    assert bool(cy.loc["g", "complete"]) is False


def test_coverage_columns_precede_metrics():
    cols = list(_cohort(_book()).columns)
    assert cols.index("complete") < cols.index("member_months")
    assert cols.index("group") < cols.index("months_observed")  # entity identity first
