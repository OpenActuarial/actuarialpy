"""Tests for the general adjust() lens and its equivalence to the special cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from actuarialpy import (
    adjust,
    apply_completion,
    completion_factors,
    deseasonalize,
    development_months,
    make_completion_triangle,
    seasonality_factors,
)


def _df():
    return pd.DataFrame({"region": ["N", "S", "N", "W"], "claims": [100.0, 200.0, 300.0, 400.0]})


# --- basic application -------------------------------------------------------

def test_scalar_multiply_and_divide():
    df = _df()
    assert adjust(df, 1.05, value_col="claims")["claims"].tolist() == [105.0, 210.0, 315.0, 420.0]
    assert np.allclose(adjust(df, 2.0, value_col="claims", how="divide")["claims"], [50, 100, 150, 200])


def test_series_by_key():
    df = _df()
    area = pd.Series({"N": 1.10, "S": 0.95, "W": 1.20})
    out = adjust(df, area, value_col="claims", on="region")
    assert np.allclose(out["claims"], [110.0, 190.0, 330.0, 480.0])


def test_grouped_dataframe_factor():
    df = pd.DataFrame({"lob": ["A", "A", "B"], "plan": ["P1", "P2", "P1"], "claims": [100.0, 100.0, 100.0]})
    tbl = pd.DataFrame({"lob": ["A", "A", "B"], "plan": ["P1", "P2", "P1"], "factor": [1.1, 0.9, 1.5]})
    out = adjust(df, tbl, value_col="claims", on="plan", by="lob")
    assert np.allclose(out["claims"], [110.0, 90.0, 150.0])


def test_out_col_leaves_source_untouched():
    df = _df()
    out = adjust(df, 1.1, value_col="claims", out_col="trended")
    assert "trended" in out.columns and np.allclose(out["claims"], df["claims"])


# --- guards ------------------------------------------------------------------

def test_absent_key_is_nan_surfaced():
    df = pd.DataFrame({"region": ["N", "Z"], "claims": [100.0, 100.0]})
    area = pd.Series({"N": 1.10})
    assert pd.isna(adjust(df, area, value_col="claims", on="region")["claims"].iloc[1])


def test_absent_key_default_means_no_adjustment():
    df = pd.DataFrame({"region": ["N", "Z"], "claims": [100.0, 100.0]})
    area = pd.Series({"N": 1.10})
    out = adjust(df, area, value_col="claims", on="region", default=1.0)
    assert np.allclose(out["claims"], [110.0, 100.0])


def test_fanout_guard():
    df = _df()
    tbl = pd.DataFrame({"region": ["N", "N"], "factor": [1.1, 1.2]})
    with pytest.raises(ValueError):
        adjust(df, tbl, value_col="claims", on="region")


def test_bad_how_raises():
    with pytest.raises(ValueError):
        adjust(_df(), 1.1, value_col="claims", how="square")


def test_series_with_by_raises():
    df = pd.DataFrame({"region": ["N"], "lob": ["A"], "claims": [100.0]})
    with pytest.raises(ValueError):
        adjust(df, pd.Series({"N": 1.1}), value_col="claims", on="region", by="lob")


# --- audit trail -------------------------------------------------------------

def test_audit_col_accumulates_net_multiplier():
    df = _df()
    area = pd.Series({"N": 1.10, "S": 0.95, "W": 1.20})
    chained = adjust(
        adjust(df, 1.05, value_col="claims", audit_col="restate"),
        area, value_col="claims", on="region", audit_col="restate",
    )
    expected = 1.05 * area.reindex(df["region"]).to_numpy()
    assert np.allclose(chained["restate"], expected)
    assert np.allclose(chained["claims"], df["claims"] * chained["restate"])


def test_audit_divide_contributes_reciprocal():
    df = pd.DataFrame({"k": ["a"], "v": [100.0]})
    out = adjust(df, pd.Series({"a": 4.0}), value_col="v", on="k", how="divide", audit_col="m")
    assert np.allclose(out["v"], [25.0]) and np.allclose(out["m"], [0.25])


# --- equivalence to the special cases (the design claim, as a regression) ----

def test_deseasonalize_equals_adjust_on_derived_season():
    months = pd.date_range("2021-01-01", "2024-12-01", freq="MS")
    shape = np.array([1.25, 1.16, 1.06, 0.97, 0.92, 0.87, 0.85, 0.88, 0.96, 1.03, 1.07, 1.18])
    pool = pd.DataFrame({"month": months,
                         "claims": 1000.0 * shape[months.month - 1] * (1.003 ** np.arange(len(months))),
                         "mm": 1000})
    sf = seasonality_factors(pool, date_col="month", value_col="claims", exposure_col="mm")
    by_special = deseasonalize(pool, sf, date_col="month", value_col="claims")["claims_deseasonalized"].to_numpy()
    via_adjust = adjust(pool.assign(season=pool["month"].dt.month), sf,
                        value_col="claims", on="season", how="divide")["claims"].to_numpy()
    assert np.allclose(by_special, via_adjust)


def test_apply_completion_in_range_equals_adjust_on_development():
    patt = np.array([0.34, 0.22, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002])
    patt = patt / patt.sum()
    origins = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    valuation = origins[-1]
    rows = []
    for og in origins:
        for d in range((pd.Period(valuation, "M") - pd.Period(og, "M")).n + 1):
            rows.append({"origin": og, "val": (pd.Period(og, "M") + d).to_timestamp("M"), "paid": 1e6 * patt[d]})
    pay = pd.DataFrame(rows)
    cf = completion_factors(make_completion_triangle(pay, origin_col="origin", valuation_col="val", amount_col="paid"))
    latest = pay.groupby("origin")["paid"].sum().reset_index().rename(columns={"paid": "ptd"})
    by_special = apply_completion(latest, cf, value_col="ptd", date_col="origin", valuation_date=valuation)["ptd_completed"].to_numpy()
    latest_dev = latest.assign(dev=development_months(latest["origin"], pd.Series(valuation, index=latest.index)))
    via_adjust = adjust(latest_dev, cf, value_col="ptd", on="dev", how="divide")["ptd"].to_numpy()
    assert np.allclose(by_special, via_adjust)
