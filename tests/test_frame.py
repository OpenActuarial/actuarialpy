"""Tests for the canonical Experience container and its design invariant."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

import actuarialpy as ap
from actuarialpy import Experience

# ------------------------------------------------------------------ #
# The design invariant: transformations in, consumers out.            #
# ------------------------------------------------------------------ #

#: Public methods allowed to return something other than Experience.
#: Adding a name here requires a visible diff -- that is the point.
ALLOWED_NON_EXPERIENCE: frozenset[str] = frozenset()


def test_every_public_method_returns_experience():
    for name, fn in inspect.getmembers(Experience, inspect.isfunction):
        if name.startswith("_") or name in ALLOWED_NON_EXPERIENCE:
            continue
        annotation = inspect.signature(fn).return_annotation
        if isinstance(annotation, str):
            annotation = annotation.strip("\"'")
        assert annotation in ("Experience", Experience), (
            f"Experience.{name} returns {annotation!r}. Transformations return "
            "Experience; analytical consumers are functions in downstream "
            "packages. If this method is a deliberate accessor, add it to "
            "ALLOWED_NON_EXPERIENCE."
        )


# ------------------------------------------------------------------ #
# Construction and roles.                                              #
# ------------------------------------------------------------------ #


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": ["A", "A", "B", "B"],
            "claim_type": ["ip", "op", "ip", "op"],
            "month": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01", "2026-02-01"]),
            "paid_claims": [100.0, 50.0, 80.0, 40.0],
            "premium": [200.0, 200.0, 150.0, 150.0],
            "member_months": [1.0, 1.0, 1.0, 1.0],
            "claim_count": [2, 1, 2, 1],
        }
    )


def test_roles_normalize_to_tuples():
    exp = Experience(_frame(), expense="paid_claims", revenue=["premium"], exposure="member_months")
    assert exp.expense == ("paid_claims",)
    assert exp.revenue == ("premium",)
    assert exp.exposure == ("member_months",)
    assert exp.count == ()


def test_at_least_one_measure_role_required():
    with pytest.raises(ValueError, match="at least one measure role"):
        Experience(_frame(), exposure="member_months", date="month")


def test_no_particular_measure_role_is_mandatory():
    # Claims-only, premium-only, and count-only bindings are all legal.
    Experience(_frame(), expense="paid_claims")
    Experience(_frame(), revenue="premium")
    Experience(_frame(), count="claim_count")


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Missing required columns"):
        Experience(_frame(), expense="not_a_column")


def test_non_numeric_measures_raise():
    df = _frame().assign(paid_claims=lambda d: d["paid_claims"].astype(str))
    with pytest.raises(ValueError, match="numeric"):
        Experience(df, expense="paid_claims")


def test_id_like_exposure_names_rejected():
    with pytest.raises(ValueError, match="identifiers"):
        Experience(_frame(), expense="paid_claims", exposure="group_id")


def test_dimensions_and_exposure_keys_validated_present():
    with pytest.raises(ValueError, match="Missing required columns"):
        Experience(_frame(), expense="paid_claims", dimensions=["nope"])
    with pytest.raises(ValueError, match="Missing required columns"):
        Experience(_frame(), expense="paid_claims", exposure_keys=["nope"])


def test_valuation_date_normalized_to_timestamp():
    exp = Experience(_frame(), expense="paid_claims", valuation_date="2026-06-30")
    assert exp.valuation_date == pd.Timestamp("2026-06-30")


# ------------------------------------------------------------------ #
# Grain guard.                                                         #
# ------------------------------------------------------------------ #


def test_exposure_keys_guard_rejects_repeated_units():
    # Service-line grain: the same member-month appears twice.
    df = pd.DataFrame(
        {
            "member_id": [1, 1, 2],
            "month": ["2026-01", "2026-01", "2026-01"],
            "service_line": ["office", "lab", "office"],
            "paid_claims": [100.0, 50.0, 80.0],
            "member_months": [1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="repeat an exposure unit"):
        Experience(df, expense="paid_claims", exposure="member_months",
                   exposure_keys=["member_id", "month"])


def test_exposure_keys_guard_passes_unit_grain():
    exp = Experience(
        _frame(),
        expense="paid_claims",
        exposure="member_months",
        exposure_keys=["group_id", "claim_type", "month"],
    )
    assert exp.exposure_keys == ("group_id", "claim_type", "month")


def test_dimensions_alone_claim_no_grain_safety():
    # Duplicates on dimensions are fine -- dimensions are segmentation,
    # not grain. Only exposure_keys triggers the guard.
    df = _frame()
    exp = Experience(df, expense="paid_claims", dimensions=["group_id"])
    assert exp.dimensions == ("group_id",)


# ------------------------------------------------------------------ #
# Basis: prerequisite state for transformations.                      #
# ------------------------------------------------------------------ #


def _monthly_paid() -> Experience:
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]),
            "paid_claims": [100.0, 100.0, 90.0, 60.0],
            "member_months": [10.0, 10.0, 10.0, 10.0],
        }
    )
    return Experience(df, expense="paid_claims", exposure="member_months",
                      date="month", valuation_date="2026-04-30")


def _flat_factors() -> pd.Series:
    return pd.Series({0: 0.60, 1: 0.90, 2: 1.00, 3: 1.00})


def test_complete_defaults_valuation_date_from_object():
    exp = _monthly_paid()
    done = exp.complete(_flat_factors())
    # April (development month 0) grosses up by 1/0.60.
    assert done.data["paid_claims"].iloc[3] == pytest.approx(60.0 / 0.60)
    # March (development month 1) grosses up by 1/0.90.
    assert done.data["paid_claims"].iloc[2] == pytest.approx(90.0 / 0.90)


def test_complete_marks_ultimate_and_refuses_double_completion():
    exp = _monthly_paid()
    assert exp.basis.get("paid_claims") is None
    done = exp.complete(_flat_factors())
    assert done.basis["paid_claims"] == "ultimate"
    with pytest.raises(ValueError, match="already on an ultimate basis"):
        done.complete(_flat_factors())


def test_basis_survives_other_transformations():
    done = _monthly_paid().complete(_flat_factors())
    trended = done.adjust(1.05)
    assert trended.basis["paid_claims"] == "ultimate"
    filtered = trended.filter(query="paid_claims > 0")
    assert filtered.basis["paid_claims"] == "ultimate"


def test_basis_rejects_unknown_columns():
    with pytest.raises(ValueError, match="basis references columns"):
        Experience(_frame(), expense="paid_claims", basis={"nope": "ultimate"})


def test_declared_ultimate_basis_blocks_completion():
    exp = Experience(
        _monthly_paid().data,
        expense="paid_claims",
        exposure="member_months",
        date="month",
        basis={"paid_claims": "ultimate"},
    )
    with pytest.raises(ValueError, match="already on an ultimate basis"):
        exp.complete(_flat_factors())


# ------------------------------------------------------------------ #
# Transformations chain and preserve immutability.                     #
# ------------------------------------------------------------------ #


def test_filter_returns_new_experience_and_leaves_original():
    exp = Experience(_frame(), expense="paid_claims", revenue="premium",
                     exposure="member_months", date="month", dimensions=["group_id"])
    sub = exp.filter(query="group_id == 'A'")
    assert len(sub.data) == 2
    assert len(exp.data) == 4
    assert sub.dimensions == exp.dimensions
    assert sub.revenue == ("premium",)


def test_adjust_multiplies_expense_in_place_under_same_name():
    exp = Experience(_frame(), expense="paid_claims")
    up = exp.adjust(2.0)
    assert (up.data["paid_claims"].to_numpy() == (_frame()["paid_claims"] * 2.0).to_numpy()).all()


def test_with_roles_updates_and_revalidates():
    exp = Experience(_frame(), expense="paid_claims")
    both = exp.with_roles(revenue="premium", dimensions=["group_id", "claim_type"])
    assert both.revenue == ("premium",)
    assert both.dimensions == ("group_id", "claim_type")
    with pytest.raises(ValueError):
        exp.with_roles(expense="missing_col")


def test_with_status_adds_column_and_returns_experience():
    df = _frame().assign(
        effective=pd.Timestamp("2025-01-01"),
        termination=pd.NaT,
    )
    exp = Experience(df, expense="paid_claims")
    out = exp.with_status(effective_col="effective", as_of="2026-01-31",
                          termination_col="termination")
    assert isinstance(out, Experience)
    assert "status" in out.data.columns


# ------------------------------------------------------------------ #
# Experience-native free functions.                                    #
# ------------------------------------------------------------------ #


def test_fit_trend_accepts_experience():
    n = 24
    dates = pd.date_range("2024-01-01", periods=n, freq="MS")
    claims = 100.0 * (1.06 ** (pd.Series(range(n)) / 12.0))
    df = pd.DataFrame({"month": dates, "paid_claims": claims, "member_months": 10.0})
    exp = Experience(df, expense="paid_claims", exposure="member_months", date="month")
    fit = ap.fit_trend(exp)
    assert fit.annual_trend == pytest.approx(0.06, abs=1e-3)


def test_fit_trend_dataframe_path_still_requires_columns():
    df = _monthly_paid().data
    with pytest.raises(TypeError, match="value_col and date_col are required"):
        ap.fit_trend(df)


def test_trend_summary_accepts_experience():
    df = pd.DataFrame(
        {
            "year": [2025, 2025, 2026, 2026],
            "paid_claims": [100.0, 100.0, 110.0, 110.0],
            "member_months": [10.0, 10.0, 10.0, 10.0],
        }
    )
    exp = Experience(df, expense="paid_claims", exposure="member_months")
    out = ap.trend_summary(exp, period_col="year", prior_period=2025, current_period=2026)
    assert len(out) == 1


def test_single_role_helpers():
    exp = Experience(_frame(), expense=["paid_claims"], revenue=["premium"])
    assert ap.single_role(exp.expense, "expense") == "paid_claims"
    assert ap.single_role_or_none(()) is None
    with pytest.raises(ValueError, match="No expense column"):
        ap.single_role((), "expense")
    with pytest.raises(ValueError, match="Multiple"):
        ap.single_role(("a", "b"), "expense")


def test_role_typo_suggests_the_close_match():
    df = pd.DataFrame({"paid_claims": [1.0], "month": ["2025-01-01"]})
    with pytest.raises(ValueError, match="did you mean 'paid_claims'"):
        Experience(df, expense="paid_claim", date="month")
