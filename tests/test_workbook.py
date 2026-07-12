"""ExperienceSet: one construction call, grain-honest members."""
import pandas as pd
import pytest

from actuarialpy import Experience, ExperienceSet, Source


def _sources():
    months = pd.date_range("2025-01-01", periods=3, freq="MS")
    membership = pd.DataFrame([
        {"member_id": m, "month": t, "group_id": "A" if m != "M3" else "B",
         "member_months": 1.0}
        for m in ("M1", "M2", "M3") for t in months])
    claims = pd.DataFrame([
        ("M1", "2025-01-14", "inpatient", 900.0),
        ("M1", "2025-02-03", "outpatient", 150.0),
        ("M2", "2025-03-09", "inpatient", 400.0),
    ], columns=["member_id", "incurred_date", "claim_type", "paid_amount"])
    claims["incurred_date"] = pd.to_datetime(claims["incurred_date"])
    billing = pd.DataFrame([
        {"member_id": m, "month": t, "premium": 450.0}
        for m in ("M1", "M2", "M3") for t in months])
    return membership, claims, billing


def _book():
    membership, claims, billing = _sources()
    return ExperienceSet.from_tables(
        membership, grain=["member_id", "month"], exposure="member_months",
        sources=[
            Source(claims, expense="paid_amount", wide_by="claim_type",
                     date="incurred_date", name="claims"),
            Source(billing, revenue="premium"),
        ],
        date="month", period="M", dimensions="group_id")


def test_one_call_two_grain_honest_members():
    book = _book()
    assert isinstance(book.tab, Experience)
    assert book.tab.exposure_keys == ("member_id", "month")
    assert book.member_names == ("tab", "claims")
    listing = book["claims"]
    assert listing.expense == ("paid_amount",)
    assert listing.date == "incurred_date"
    assert "claim_type" in listing.dimensions
    assert len(listing.data) == 3          # the source rows, untouched
    with pytest.raises(KeyError, match="named listings"):
        book["premium"]


def test_cohort_rederives_every_member():
    small = _book().cohort(group_id="A")
    assert set(small.tab.data["group_id"]) == {"A"}
    assert set(small["claims"].data["member_id"]) <= {"M1", "M2"}
    assert len(small.tab.data) == 6        # 2 members x 3 months
    # totals still tie after reconstruction
    assert bool(small.reconcile()["ties"].all())
    with pytest.raises(ValueError, match="grain-table columns"):
        _book().cohort(claim_type="inpatient")


def test_reconcile_ties_listing_to_tab():
    rec = _book().reconcile()
    row = rec[rec["measure"] == "paid_amount"].iloc[0]
    assert row["source_total"] == pytest.approx(1_450.0)
    assert bool(row["ties"])


def test_consumers_route_the_set_to_the_right_member():
    import experiencestudies as es
    from lossmodels.integrations.actuarialpy import claim_amounts
    book = _book()
    assert es.summary(book, "group_id").equals(es.summary(book.tab, "group_id"))
    assert claim_amounts(book) == pytest.approx(
        claim_amounts(book["claims"]))
