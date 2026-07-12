"""from_tables, aggregate, and melt: multi-table doorway and structural reshapes."""
import pandas as pd
import pytest

from actuarialpy import Experience, Source


def _membership():
    months = pd.date_range("2025-01-01", periods=3, freq="MS")
    return pd.DataFrame([
        {"member_id": m, "month": t, "group_id": "A" if m in ("M1", "M2") else "B",
         "member_months": 1.0, "stop_loss_deductible": 250_000.0}
        for m in ("M1", "M2", "M3") for t in months
    ])


def _claim_lines():
    return pd.DataFrame([
        # member, incurred date (mid-month), type, paid
        ("M1", "2025-01-14", "inpatient", 900.0),
        ("M1", "2025-01-20", "outpatient", 100.0),
        ("M1", "2025-02-03", "outpatient", 150.0),
        ("M2", "2025-03-09", "inpatient", 400.0),
        ("M9", "2025-01-05", "inpatient", 999.0),   # orphan: not in membership
    ], columns=["member_id", "incurred_date", "claim_type", "paid_amount"])


def _billing():
    months = pd.date_range("2025-01-01", periods=3, freq="MS")
    return pd.DataFrame([
        {"member_id": m, "month": t, "billed_premium": 450.0}
        for m in ("M1", "M2", "M3") for t in months
    ])


def _build(unmatched="warn"):
    return Experience.from_tables(
        _membership(),
        grain=["member_id", "month"],
        exposure="member_months",
        sources=[
            Source(_claim_lines(), expense="paid_amount",
                     wide_by="claim_type", date="incurred_date"),
            Source(_claim_lines(), count="paid_amount", agg="count",
                     rename={"paid_amount": "claim_count"}, date="incurred_date"),
            Source(_billing(), revenue="billed_premium"),
        ],
        date="month", period="M",
        dimensions="group_id", valuation_date="2025-03-31",
        unmatched=unmatched,
    )


def test_from_tables_joins_pivots_and_records_provenance():
    with pytest.warns(UserWarning, match="not present in the grain table"):
        exp = _build()
    assert exp.expense == ("inpatient", "outpatient")
    assert exp.revenue == ("billed_premium",)
    assert exp.count == ("claim_count",)
    assert exp.exposure_keys == ("member_id", "month")
    row = exp.data.set_index(["member_id", "month"]).loc[("M1", pd.Timestamp("2025-01-01"))]
    assert row["inpatient"] == 900.0 and row["outpatient"] == 100.0
    assert row["claim_count"] == 2          # two claim lines counted
    assert row["billed_premium"] == 450.0
    # empty cell -> structural zero, orphan M9 excluded
    m3 = exp.data.set_index(["member_id", "month"]).loc[("M3", pd.Timestamp("2025-02-01"))]
    assert m3["inpatient"] == 0.0
    assert float(exp.data["inpatient"].sum()) == 1_300.0
    (pivot,) = exp.pivots
    assert pivot.by == "claim_type" and pivot.value == "paid_amount"
    assert pivot.columns == ("inpatient", "outpatient")
    # entity attributes ride along
    assert "stop_loss_deductible" in exp.data.columns


def test_from_tables_refuses_coarser_tables_and_duplicate_grain():
    group_month = pd.DataFrame({"month": [pd.Timestamp("2025-01-01")], "fee": [10.0]})
    with pytest.raises(ValueError, match="never allocated downward"):
        Experience.from_tables(_membership(), grain=["member_id", "month"],
                               exposure="member_months",
                               sources=[Source(group_month, expense="fee")],
                               date="month")
    dup = pd.concat([_membership(), _membership().head(1)])
    with pytest.raises(ValueError, match="repeat an exposure unit"):
        Experience.from_tables(dup, grain=["member_id", "month"],
                               exposure="member_months", date="month")


def test_from_tables_raise_mode_and_cardinality_guard():
    with pytest.raises(ValueError, match="not present in the grain table"):
        _build(unmatched="raise")
    ids = _claim_lines().assign(claim_type=lambda d: "c" + d.index.astype(str) * 20)
    big = pd.concat([ids] * 20).reset_index(drop=True)
    big["claim_type"] = "c" + big.index.astype(str)
    with pytest.raises(ValueError, match="looks like an identifier"):
        Experience.from_tables(_membership(), grain=["member_id", "month"],
                               exposure="member_months", date="month", period="M",
                               sources=[Source(big, expense="paid_amount",
                                                wide_by="claim_type", date="incurred_date")])


def test_aggregate_sums_to_new_grain_and_requires_grain_proof():
    with pytest.warns(UserWarning):
        exp = _build()
    monthly = exp.aggregate(by="group_id", freq="MS")
    assert monthly.exposure_keys == ("group_id", "month")
    a_jan = monthly.data.set_index(["group_id", "month"]).loc[("A", pd.Timestamp("2025-01-01"))]
    assert a_jan["member_months"] == 2.0 and a_jan["inpatient"] == 900.0
    assert monthly.dimensions == ("group_id",)
    assert monthly.pivots == exp.pivots            # pivot columns survive summation
    unguarded = Experience(exp.data, expense="inpatient", exposure="member_months")
    with pytest.raises(ValueError, match="bind exposure_keys"):
        unguarded.aggregate(by="group_id")


def test_melt_inverts_the_recorded_pivot():
    with pytest.warns(UserWarning):
        exp = _build()
    long = exp.melt()
    assert long.expense == ("paid_amount",)
    assert "claim_type" in long.dimensions
    assert long.exposure_keys == ()                # exposure repeats per category
    assert long.pivots == ()
    total_wide = float(exp.data[["inpatient", "outpatient"]].to_numpy().sum())
    assert float(long.data["paid_amount"].sum()) == pytest.approx(total_wide)
    plain = Experience(exp.data, expense="inpatient", date="month")
    with pytest.raises(ValueError, match="no recorded pivot"):
        plain.melt()


def test_measures_keys_maps_differently_named_join_columns():
    lines = _claim_lines().rename(columns={"member_id": "mbr_id"})
    with pytest.warns(UserWarning, match="not present in the grain table"):
        exp = Experience.from_tables(
            _membership(), grain=["member_id", "month"], exposure="member_months",
            sources=[Source(lines, expense="paid_amount", wide_by="claim_type",
                             date="incurred_date", keys={"mbr_id": "member_id"})],
            date="month", period="M")
    assert float(exp.data["inpatient"].sum()) == 1_300.0
    with pytest.raises(ValueError, match="Missing required columns"):
        Source(lines, expense="paid_amount", keys={"member_id": "member_id"})
