"""Tests for working-day counting and seasonal factors."""

import numpy as np
import pandas as pd
import pytest

from actuarialpy import (
    add_business_days,
    apply_seasonality,
    business_days_in_period,
    deseasonalize,
    seasonality_factors,
)
from actuarialpy.reserving import InsufficientDataWarning

# True monthly seasonal shape (flu-heavy winter, light summer), normalized to mean 1.0.
_TRUE = np.array([1.20, 1.15, 1.05, 0.98, 0.95, 0.90, 0.88, 0.90, 0.97, 1.02, 1.05, 1.15])
_TRUE = _TRUE / _TRUE.mean()


def _book(start="2019-01-01", end="2024-12-01", trend=1.004, growth=1.003, base=350.0,
          mm0=10000, season=_TRUE):
    months = pd.date_range(start, end, freq="MS")
    t = np.arange(len(months))
    pmpm = base * (trend ** t) * season[months.month.values - 1]
    members = (mm0 * (growth ** t)).round()
    return pd.DataFrame(
        {"month": months, "claims": pmpm * members, "member_months": members, "pmpm": pmpm}
    )


# --- working days -----------------------------------------------------------

def test_business_days_match_known_2024_federal():
    bd = business_days_in_period(pd.date_range("2024-01-01", "2024-12-01", freq="MS"))
    assert list(bd.values) == [21, 20, 21, 22, 22, 19, 22, 22, 20, 22, 19, 21]


def test_weekmask_string_equivalence():
    a = business_days_in_period(pd.date_range("2024-01-01", "2024-06-01", freq="MS"), weekmask="Mon Tue Wed Thu Fri")
    b = business_days_in_period(pd.date_range("2024-01-01", "2024-06-01", freq="MS"), weekmask="1111100")
    assert list(a.values) == list(b.values)


def test_holidays_none_gives_plain_weekdays():
    # July 2024 has 23 weekdays; with the federal July 4 holiday it drops to 22.
    with_hol = business_days_in_period([pd.Timestamp("2024-07-15")])
    no_hol = business_days_in_period([pd.Timestamp("2024-07-15")], holidays=None)
    assert int(with_hol.iloc[0]) == 22
    assert int(no_hol.iloc[0]) == 23


def test_custom_holiday_list():
    base = business_days_in_period([pd.Timestamp("2024-03-15")], holidays=None)  # 21 weekdays
    one_off = business_days_in_period([pd.Timestamp("2024-03-15")], holidays=["2024-03-15"])
    assert int(base.iloc[0]) - int(one_off.iloc[0]) == 1


def test_add_business_days_column_and_per_day_rate():
    df = pd.DataFrame({"month": pd.date_range("2024-01-01", "2024-12-01", freq="MS"), "paid": 1000.0})
    out = add_business_days(df, "month")
    assert list(out["business_days"]) == [21, 20, 21, 22, 22, 19, 22, 22, 20, 22, 19, 21]
    # original is not mutated by default
    assert "business_days" not in df.columns


def test_quarterly_business_days():
    bd = business_days_in_period(pd.date_range("2024-01-01", "2024-10-01", freq="QS"), freq="Q")
    # Q1 2024 = Jan+Feb+Mar weekdays minus 3 federal holidays = 62.
    assert int(bd.iloc[0]) == 62


# --- seasonal factors -------------------------------------------------------

def test_recovers_known_pattern_under_trend_and_growth():
    df = _book()
    f = seasonality_factors(df, date_col="month", value_col="claims", exposure_col="member_months")
    assert np.max(np.abs(f.values - _TRUE)) < 0.005
    assert np.isclose(f.mean(), 1.0)
    assert list(f.index) == list(range(1, 13))


def test_factors_normalized_to_mean_one():
    f = seasonality_factors(_book(), date_col="month", value_col="claims", exposure_col="member_months")
    assert np.isclose(f.mean(), 1.0)


def test_deseasonalize_then_reseasonalize_roundtrip():
    df = _book()
    f = seasonality_factors(df, date_col="month", value_col="claims", exposure_col="member_months")
    ds = deseasonalize(df, f, date_col="month", value_col="pmpm")
    rs = apply_seasonality(ds, f, date_col="month", value_col="pmpm_deseasonalized", out_col="roundtrip")
    assert np.allclose(rs["roundtrip"].values, df["pmpm"].values)


def test_deseasonalize_flattens_to_smooth_trend():
    df = _book()
    f = seasonality_factors(df, date_col="month", value_col="claims", exposure_col="member_months")
    ds = deseasonalize(df, f, date_col="month", value_col="pmpm")
    t = np.arange(len(df))
    detrended = ds["pmpm_deseasonalized"].values / (350.0 * (1.004 ** t))
    assert np.std(detrended) < 0.01  # seasonality removed -> only smooth trend remains


def test_value_only_matches_exposure_weighted_when_membership_flat():
    df = _book(growth=1.0)  # constant membership -> totals and PMPM share the same shape
    f_rate = seasonality_factors(df, date_col="month", value_col="claims", exposure_col="member_months")
    f_val = seasonality_factors(df, date_col="month", value_col="claims")
    assert np.allclose(f_rate.values, f_val.values, atol=1e-9)


def test_exclude_distorted_year_changes_factors():
    df = _book()
    # A uniform full-year scale would cancel in ratio-to-MA, so distort the *shape*:
    # spike only summer 2021 (a COVID-deferred-care style surge in months 6-8).
    mask = (df["month"].dt.year == 2021) & (df["month"].dt.month.isin([6, 7, 8]))
    df.loc[mask, "claims"] = df.loc[mask, "claims"] * 1.6
    f_all = seasonality_factors(df, date_col="month", value_col="claims", exposure_col="member_months")
    f_excl = seasonality_factors(df, date_col="month", value_col="claims",
                                 exposure_col="member_months", exclude=[2021])
    # the distortion pushes the summer factors up; excluding 2021 restores the true shape
    assert np.max(np.abs(f_all.values - _TRUE)) > np.max(np.abs(f_excl.values - _TRUE))
    assert np.isclose(f_excl.mean(), 1.0)


def test_period_share_method_runs_and_normalizes():
    f = seasonality_factors(_book(), date_col="month", value_col="claims",
                            exposure_col="member_months", method="period_share")
    assert np.isclose(f.mean(), 1.0)
    assert len(f) == 12


def test_median_aggregate_robust_to_single_outlier_month():
    df = _book()
    # one freak January (single month) should barely move the median factor
    df.loc[df["month"] == pd.Timestamp("2022-01-01"), "claims"] *= 3
    f = seasonality_factors(df, date_col="month", value_col="claims",
                            exposure_col="member_months", aggregate="median")
    assert abs(f.loc[1] - _TRUE[0]) < 0.05


def test_quarterly_factors_have_four_seasons():
    df = _book()
    f = seasonality_factors(df, date_col="month", value_col="claims",
                            exposure_col="member_months", freq="Q")
    assert list(f.index) == [1, 2, 3, 4]
    assert np.isclose(f.mean(), 1.0)


def test_short_history_warns():
    df = _book(end="2019-12-01")  # one year only
    with pytest.warns(InsufficientDataWarning):
        seasonality_factors(df, date_col="month", value_col="claims", exposure_col="member_months")


def test_bad_freq_and_method_raise():
    df = _book()
    with pytest.raises(ValueError):
        seasonality_factors(df, date_col="month", value_col="claims", freq="W")
    with pytest.raises(ValueError):
        seasonality_factors(df, date_col="month", value_col="claims", method="nope")
    with pytest.raises(ValueError):
        seasonality_factors(df, date_col="month", value_col="claims", aggregate="nope")


def test_missing_columns_raise():
    df = _book()
    with pytest.raises((KeyError, ValueError)):
        seasonality_factors(df, date_col="month", value_col="not_there")


# --- Experience.deseasonalize lens ------------------------------------------

from actuarialpy import Experience  # noqa: E402


def _exp_book():
    df = _book().assign(premium=lambda d: (d["claims"] * 1.15).round(2))
    exp = Experience(df, expense="claims", revenue="premium", exposure="member_months", date="month")
    factors = seasonality_factors(df, date_col="month", value_col="claims", exposure_col="member_months")
    return df, exp, factors


def test_experience_deseasonalize_returns_new_experience():
    _, exp, f = _exp_book()
    clean = exp.deseasonalize(f)
    assert isinstance(clean, Experience)
    assert clean is not exp


def test_experience_deseasonalize_leaves_original_untouched():
    df, exp, f = _exp_book()
    before = exp.data["claims"].copy()
    exp.deseasonalize(f)
    assert np.allclose(exp.data["claims"], before)


def test_experience_deseasonalize_divides_expense_only():
    df, exp, f = _exp_book()
    clean = exp.deseasonalize(f)
    fac = df["month"].dt.month.map(f).values
    assert np.allclose(clean.data["claims"], df["claims"].values / fac)
    assert np.allclose(clean.data["member_months"], df["member_months"])  # exposure untouched
    assert np.allclose(clean.data["premium"], df["premium"])  # revenue untouched


def test_experience_deseasonalize_removes_monthly_swing():
    df, exp, f = _exp_book()
    clean = exp.deseasonalize(f)
    raw = (df["claims"] / df["member_months"]).values
    cl = (clean.data["claims"] / clean.data["member_months"]).values
    assert np.std(np.diff(cl) / cl[:-1]) < np.std(np.diff(raw) / raw[:-1]) / 5


def test_experience_deseasonalize_composes_with_views():
    _, exp, f = _exp_book()
    clean = exp.deseasonalize(f)
    assert clean.by().shape[0] >= 1
    assert clean.rolling(12).shape[0] >= 1


def test_experience_deseasonalize_columns_override_multi():
    df = _book().assign(rx=lambda d: (d["claims"] * 0.2).round(2), premium=lambda d: (d["claims"] * 1.15).round(2))
    exp = Experience(df, expense=["claims", "rx"], revenue="premium", exposure="member_months", date="month")
    f = seasonality_factors(df, date_col="month", value_col="claims", exposure_col="member_months")
    clean = exp.deseasonalize(f)
    fac = df["month"].dt.month.map(f).values
    assert np.allclose(clean.data["claims"], df["claims"].values / fac)
    assert np.allclose(clean.data["rx"], (df["claims"].values * 0.2) / fac)


def test_experience_deseasonalize_columns_subset():
    df, exp, f = _exp_book()
    clean = exp.deseasonalize(f, columns="claims")
    fac = df["month"].dt.month.map(f).values
    assert np.allclose(clean.data["claims"], df["claims"].values / fac)
    assert np.allclose(clean.data["premium"], df["premium"])  # not selected -> untouched


def test_experience_deseasonalize_requires_date():
    df, _, f = _exp_book()
    nodate = Experience(df, expense="claims", revenue="premium", exposure="member_months")
    with pytest.raises(ValueError):
        nodate.deseasonalize(f)


# --- grouped (per-segment) seasonality via by= -----------------------------

from actuarialpy import seasonality_factors_by  # noqa: E402

_SHAPES = {
    "A": np.array([1.25, 1.15, 1.05, 0.97, 0.93, 0.88, 0.86, 0.89, 0.96, 1.02, 1.06, 1.18]),
    "B": np.array([1.05, 1.04, 1.02, 1.00, 0.99, 0.98, 0.97, 0.98, 1.00, 1.01, 1.02, 1.04]),
}


def _two_lob_seasonal():
    rows = []
    for lob, s in _SHAPES.items():
        s = s / s.mean()
        m = pd.date_range("2021-01-01", "2024-12-01", freq="MS")
        t = np.arange(len(m))
        mm = (8000 * 1.003 ** t).round()
        pmpm = 300 * 1.004 ** t * s[m.month.values - 1]
        for i in range(len(m)):
            rows.append({"lob": lob, "month": m[i], "claims": round(pmpm[i] * mm[i], 2), "member_months": mm[i]})
    return pd.DataFrame(rows)


def test_seasonality_factors_by_tidy_shape():
    df = _two_lob_seasonal()
    sf = seasonality_factors_by(df, groupby="lob", date_col="month", value_col="claims", exposure_col="member_months")
    assert list(sf.columns) == ["lob", "season", "seasonal_factor"]
    assert len(sf) == 24  # 12 seasons x 2 LOBs
    # each LOB's factors average ~1.0
    for lob in _SHAPES:
        assert np.isclose(sf[sf["lob"] == lob]["seasonal_factor"].mean(), 1.0)


def test_grouped_deseasonalize_collapses_swing_per_group():
    df = _two_lob_seasonal()
    sf = seasonality_factors_by(df, groupby="lob", date_col="month", value_col="claims", exposure_col="member_months")
    ds = deseasonalize(df, sf, date_col="month", value_col="claims", by="lob")
    for lob in _SHAPES:
        g = ds[ds["lob"] == lob]
        raw = (g["claims"] / g["member_months"]).values
        cl = (g["claims_deseasonalized"] / g["member_months"]).values
        assert np.std(np.diff(cl) / cl[:-1]) < np.std(np.diff(raw) / raw[:-1]) / 5


def test_grouped_deseasonalize_reseasonalize_roundtrip():
    df = _two_lob_seasonal()
    sf = seasonality_factors_by(df, groupby="lob", date_col="month", value_col="claims", exposure_col="member_months")
    ds = deseasonalize(df, sf, date_col="month", value_col="claims", by="lob")
    rs = apply_seasonality(ds, sf, date_col="month", value_col="claims_deseasonalized", by="lob", out_col="rt")
    assert np.allclose(rs["rt"].values, df["claims"].values)


def test_grouped_seasonality_fanout_guard():
    df = _two_lob_seasonal()
    sf = seasonality_factors_by(df, groupby="lob", date_col="month", value_col="claims", exposure_col="member_months")
    dup = pd.concat([sf, sf.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        deseasonalize(df, dup, date_col="month", value_col="claims", by="lob")


def test_grouped_seasonality_absent_group_is_nan():
    df = _two_lob_seasonal()
    sf = seasonality_factors_by(df, groupby="lob", date_col="month", value_col="claims", exposure_col="member_months")
    df2 = df.copy()
    df2.loc[df2.index[0], "lob"] = "Z"
    ds = deseasonalize(df2, sf, date_col="month", value_col="claims", by="lob")
    assert pd.isna(ds.iloc[0]["claims_deseasonalized"])


def test_experience_deseasonalize_grouped():
    df = _two_lob_seasonal().assign(premium=lambda d: d["claims"] * 1.2)
    sf = seasonality_factors_by(df, groupby="lob", date_col="month", value_col="claims", exposure_col="member_months")
    exp = Experience(df, expense="claims", revenue="premium", exposure="member_months", date="month")
    clean = exp.deseasonalize(sf, by="lob")
    ref = deseasonalize(df, sf, date_col="month", value_col="claims", by="lob")
    assert np.allclose(clean.data["claims"].to_numpy(), ref["claims_deseasonalized"].to_numpy())
    assert clean.by().shape[0] >= 1


def test_multi_column_grouping():
    df = _two_lob_seasonal().assign(product="P1")
    sf = seasonality_factors_by(df, groupby=["lob", "product"], date_col="month", value_col="claims",
                                exposure_col="member_months")
    assert list(sf.columns) == ["lob", "product", "season", "seasonal_factor"]
    ds = deseasonalize(df, sf, date_col="month", value_col="claims", by=["lob", "product"])
    assert not ds["claims_deseasonalized"].isna().any()
