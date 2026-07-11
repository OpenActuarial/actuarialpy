import pandas as pd

from actuarialpy.lifecycle import (
    STATUS_ACTIVE,
    STATUS_FIRST_YEAR,
    STATUS_TERMED,
    add_months_in_force,
    add_tenure,
    derive_status,
    earned_exposure,
    is_in_force,
)

AS_OF = "2026-03-31"
PERIOD_START = "2026-01-01"
PERIOD_END = "2026-03-01"


def lifecycle_df():
    return pd.DataFrame({
        "group_id": ["G1", "G2", "G3", "G4", "G5"],
        "effective_date": pd.to_datetime([
            "2020-01-01",  # long-tenured -> active
            "2025-10-01",  # 5 months -> first_year
            "2022-01-01",  # termed before as_of
            "2025-06-01",  # 9 months, future-dated term -> first_year
            "2026-02-15",  # mid-period entrant -> first_year
        ]),
        "termination_date": pd.to_datetime([
            None,
            None,
            "2025-06-01",   # <= as_of
            "2027-01-01",   # after as_of (still active)
            None,
        ]),
        "exposure": [12, 12, 12, 12, 12],
    })


def test_derive_status_three_cohorts():
    out = derive_status(
        lifecycle_df(),
        effective_col="effective_date",
        termination_col="termination_date",
        as_of=AS_OF,
        first_year_months=12,
    )
    by_id = out.set_index("group_id")["status"]
    assert by_id["G1"] == STATUS_ACTIVE
    assert by_id["G2"] == STATUS_FIRST_YEAR
    assert by_id["G3"] == STATUS_TERMED
    assert by_id["G4"] == STATUS_FIRST_YEAR
    assert by_id["G5"] == STATUS_FIRST_YEAR


def test_derive_status_custom_labels():
    out = derive_status(
        lifecycle_df(),
        effective_col="effective_date",
        termination_col="termination_date",
        as_of=AS_OF,
        labels={"termed": "Term", "first_year": "First Year Account"},
    )
    statuses = set(out["status"])
    assert "Term" in statuses
    assert "First Year Account" in statuses


def test_add_tenure():
    out = add_tenure(lifecycle_df(), "effective_date", AS_OF)
    by_id = out.set_index("group_id")["tenure_months"]
    assert by_id["G1"] == 74  # (2026-2020)*12 + (3-1)
    assert by_id["G2"] == 5
    assert by_id["G3"] == 50
    assert by_id["G5"] == 1


def test_is_in_force():
    mask = is_in_force(
        lifecycle_df(),
        effective_col="effective_date",
        termination_col="termination_date",
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )
    flags = dict(zip(lifecycle_df()["group_id"], mask, strict=True))
    assert flags["G1"] is True or flags["G1"]
    assert flags["G2"]
    assert not flags["G3"]  # terminated before the period
    assert flags["G4"]
    assert flags["G5"]


def test_months_in_force():
    out = add_months_in_force(
        lifecycle_df(),
        effective_col="effective_date",
        termination_col="termination_date",
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )
    by_id = out.set_index("group_id")["months_in_force"]
    assert by_id["G1"] == 3   # full Jan-Mar
    assert by_id["G2"] == 3
    assert by_id["G3"] == 0   # termed before period
    assert by_id["G4"] == 3
    assert by_id["G5"] == 2   # entered mid-period (Feb, Mar)


def test_earned_exposure_prorates():
    out = earned_exposure(
        lifecycle_df(),
        "exposure",
        effective_col="effective_date",
        termination_col="termination_date",
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )
    by_id = out.set_index("group_id")["earned_exposure"]
    assert by_id["G1"] == 12          # full period
    assert by_id["G5"] == 12 * 2 / 3  # two of three months
    assert by_id["G3"] == 0
