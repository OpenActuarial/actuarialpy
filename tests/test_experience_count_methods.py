"""Experience facade: the count role and the frequency_severity / decompose_trend / fit_trend methods."""

import pandas as pd
import pytest

import actuarialpy as ap
from actuarialpy import decompose_per_exposure_trend, fit_trend, frequency_severity_summary


def _panel():
    """Monthly two-year panel by morbidity segment and region; enrollment shifts toward High in 2025.

    Within-region frequency and severity are uniform across regions, so aggregating
    over region reproduces the segment-level totals exactly.
    """
    rows = []
    specs = {"Low": (0.45, 420.0), "High": (0.95, 820.0)}   # (frequency, severity)
    regions = {"North": 0.6, "South": 0.4}                  # split of member months
    for year in (2024, 2025):
        growth = 1.04 if year == 2025 else 1.0              # unit-cost trend
        for month in range(1, 13):
            for seg, (u, c) in specs.items():
                mm_total = 3000.0 if seg == "Low" else (1800.0 if year == 2025 else 1500.0)
                for region, share in regions.items():
                    mm = mm_total * share
                    cnt = u * mm
                    rows.append({
                        "date": pd.Timestamp(year=year, month=month, day=1),
                        "year": year, "segment": seg, "region": region,
                        "member_months": mm, "claim_count": cnt, "allowed": c * growth * cnt,
                    })
    return pd.DataFrame(rows)


def _exp(df=None, **over):
    df = _panel() if df is None else df
    kw = dict(expense="allowed", revenue="allowed", exposure="member_months", date="date", count="claim_count")
    kw.update(over)
    return ap.Experience(df, **kw)


def test_count_role_validates_numeric():
    df = _panel().assign(bad_count="x")
    _exp(df)  # numeric claim_count is fine
    with pytest.raises(ValueError, match="Count columns must be numeric"):
        ap.Experience(df, expense="allowed", revenue="allowed", exposure="member_months", count="bad_count")


def test_count_role_missing_column_raises():
    with pytest.raises(ValueError):
        ap.Experience(_panel(), expense="allowed", revenue="allowed", exposure="member_months", count="nope")


def test_frequency_severity_matches_free_function():
    df = _panel()
    fac = _exp(df).frequency_severity(groupby="segment")
    free = frequency_severity_summary(df, count_col="claim_count", loss_col="allowed",
                                      exposure_col="member_months", groupby="segment")
    pd.testing.assert_frame_equal(fac, free)


def test_decompose_trend_period_mode_matches_free_and_reconciles():
    df = _panel()
    fac = _exp(df).decompose_trend(period_col="year", prior_period=2024, current_period=2025, mix_by="segment")
    free = decompose_per_exposure_trend(df[df["year"] == 2024], df[df["year"] == 2025],
                                count_col="claim_count", loss_col="allowed",
                                exposure_col="member_months", mix_by="segment")
    pd.testing.assert_frame_equal(fac, free)
    r = fac.iloc[0]
    assert r["frequency_trend"] * r["severity_trend"] * r["mix_trend"] == pytest.approx(r["loss_per_exposure_trend"])


def test_decompose_trend_two_way_without_mix():
    fac = _exp().decompose_trend(period_col="year", prior_period=2024, current_period=2025)
    assert "mix_trend" not in fac.columns
    r = fac.iloc[0]
    assert r["frequency_trend"] * r["severity_trend"] == pytest.approx(r["loss_per_exposure_trend"])


def test_decompose_trend_uses_bound_date_with_ranges():
    df = _panel()
    fac = _exp(df).decompose_trend(
        prior_start="2024-01-01", prior_end="2024-12-31",
        current_start="2025-01-01", current_end="2025-12-31", mix_by="segment",
    )
    pri = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")]
    cur = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-12-31")]
    free = decompose_per_exposure_trend(pri, cur, count_col="claim_count", loss_col="allowed",
                                exposure_col="member_months", mix_by="segment")
    pd.testing.assert_frame_equal(fac, free)


def test_decompose_trend_on_groups_one_row_per_group():
    out = _exp().decompose_trend(period_col="year", prior_period=2024, current_period=2025,
                                 mix_by="segment", groupby="region")
    assert set(out["region"]) == {"North", "South"}
    for _, r in out.iterrows():
        assert r["frequency_trend"] * r["severity_trend"] * r["mix_trend"] == pytest.approx(r["loss_per_exposure_trend"])


def test_decompose_trend_on_and_mix_by_must_differ():
    with pytest.raises(ValueError, match="distinct dimensions"):
        _exp().decompose_trend(period_col="year", prior_period=2024, current_period=2025,
                               mix_by="segment", groupby="segment")


def test_fit_trend_matches_free_function():
    df = _panel()
    fac = _exp(df).fit_trend()
    free = fit_trend(df, value_col="allowed", date_col="date", exposure_col="member_months")
    assert fac.annual_trend == pytest.approx(free.annual_trend)
    assert fac.r_squared == pytest.approx(free.r_squared)


def test_methods_require_count_when_unbound():
    exp = ap.Experience(_panel(), expense="allowed", revenue="allowed", exposure="member_months", date="date")
    with pytest.raises(ValueError, match="count column is required"):
        exp.frequency_severity()
    with pytest.raises(ValueError, match="count column is required"):
        exp.decompose_trend(period_col="year", prior_period=2024, current_period=2025)


def test_count_survives_filter_and_with_roles():
    filtered = _exp().filter(query="segment == 'Low'")
    assert filtered.count == "claim_count"
    out = filtered.frequency_severity()
    assert out["loss_per_exposure"].iloc[0] == pytest.approx(out["frequency"].iloc[0] * out["severity"].iloc[0])
    assert _exp(count=None).with_roles(count="claim_count").count == "claim_count"
