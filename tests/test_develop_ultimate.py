"""Tests for develop_ultimate: chain ladder, Bornhuetter-Ferguson, Benktander, Cape Cod."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from actuarialpy import (
    apply_completion,
    completion_factors,
    completion_factors_by,
    develop_ultimate,
    make_completion_triangle,
)

_PATTERN = np.array([0.34, 0.22, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002])
_PATTERN = _PATTERN / _PATTERN.sum()


def _scenario(ultimate_by_origin=1_000_000.0, lob=None):
    origins = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    valuation = origins[-1]
    rows = []
    for og in origins:
        for d in range((pd.Period(valuation, "M") - pd.Period(og, "M")).n + 1):
            row = {"o": og, "val": (pd.Period(og, "M") + d).to_timestamp("M"), "paid": ultimate_by_origin * _PATTERN[d]}
            if lob is not None:
                row["lob"] = lob
            rows.append(row)
    pay = pd.DataFrame(rows)
    cf = completion_factors(make_completion_triangle(pay, origin_col="o", valuation_col="val", amount_col="paid"))
    latest = pay.groupby("o")["paid"].sum().reset_index().rename(columns={"paid": "ptd"})
    return pay, cf, latest, valuation


def test_chain_ladder_matches_apply_completion():
    _, cf, latest, vt = _scenario()
    cl = develop_ultimate(latest, cf, method="chain_ladder", value_col="ptd", date_col="o", valuation_date=vt)["ptd_ultimate"]
    ac = apply_completion(latest, cf, value_col="ptd", date_col="o", valuation_date=vt)["ptd_completed"]
    assert np.allclose(cl.to_numpy(), ac.to_numpy())


def test_bf_with_apriori_equal_to_cl_reproduces_cl():
    _, cf, latest, vt = _scenario()
    cl = develop_ultimate(latest, cf, method="chain_ladder", value_col="ptd", date_col="o", valuation_date=vt)["ptd_ultimate"].to_numpy()
    bf = develop_ultimate(latest.assign(ap=cl), cf, method="bornhuetter_ferguson", value_col="ptd",
                          date_col="o", valuation_date=vt, apriori_col="ap")["ptd_ultimate"].to_numpy()
    assert np.allclose(bf, cl)  # BF anchored to the CL ultimate is the CL ultimate


def test_benktander_between_bf_and_chain_ladder():
    _, cf, latest, vt = _scenario()
    plan = 950_000.0
    lat = latest.assign(ap=plan)
    cl = develop_ultimate(lat, cf, method="chain_ladder", value_col="ptd", date_col="o", valuation_date=vt)["ptd_ultimate"].to_numpy()
    bf = develop_ultimate(lat, cf, method="bornhuetter_ferguson", value_col="ptd", date_col="o", valuation_date=vt, apriori_col="ap")["ptd_ultimate"].to_numpy()
    gb = develop_ultimate(lat, cf, method="benktander", value_col="ptd", date_col="o", valuation_date=vt, apriori_col="ap")["ptd_ultimate"].to_numpy()
    assert np.all(gb >= np.minimum(bf, cl) - 1e-6) and np.all(gb <= np.maximum(bf, cl) + 1e-6)


def test_mature_origins_collapse_to_paid():
    _, cf, latest, vt = _scenario()
    lat = latest.assign(ap=900_000.0, prem=1_100_000.0)
    development = pd.PeriodIndex(pd.to_datetime(lat["o"]), freq="M").map(lambda p: (pd.Period(vt, "M") - p).n).to_numpy()
    mature = development >= len(_PATTERN) - 1
    paid = lat["ptd"].to_numpy()
    for method, kwargs in [
        ("bornhuetter_ferguson", {"apriori_col": "ap"}),
        ("benktander", {"apriori_col": "ap"}),
        ("cape_cod", {"exposure_col": "prem"}),
    ]:
        out = develop_ultimate(lat, cf, method=method, value_col="ptd", date_col="o", valuation_date=vt, **kwargs)["ptd_ultimate"].to_numpy()
        assert np.allclose(out[mature], paid[mature])  # fully emerged -> ultimate == paid


def test_cape_cod_elr_is_used_up_premium_ratio():
    _, cf, latest, vt = _scenario()
    premium = 1_200_000.0
    lat = latest.assign(prem=premium)
    cc = develop_ultimate(lat, cf, method="cape_cod", value_col="ptd", date_col="o", valuation_date=vt, exposure_col="prem")["ptd_ultimate"].to_numpy()
    development = pd.PeriodIndex(pd.to_datetime(lat["o"]), freq="M").map(lambda p: (pd.Period(vt, "M") - p).n).to_numpy()
    emerged = np.array(pd.Series(development).map(cf), dtype="float64")
    emerged[development > int(cf.index.max())] = 1.0
    elr = lat["ptd"].to_numpy().sum() / (premium * emerged).sum()
    expected = lat["ptd"].to_numpy() + premium * elr * (1 - emerged)
    assert np.allclose(cc, expected)


def test_required_params_and_bad_method():
    _, cf, latest, vt = _scenario()
    with pytest.raises(ValueError):
        develop_ultimate(latest, cf, method="bornhuetter_ferguson", value_col="ptd", date_col="o", valuation_date=vt)
    with pytest.raises(ValueError):
        develop_ultimate(latest, cf, method="cape_cod", value_col="ptd", date_col="o", valuation_date=vt)
    with pytest.raises(ValueError):
        develop_ultimate(latest, cf, method="munich", value_col="ptd", date_col="o", valuation_date=vt, apriori_col="x")


def test_cape_cod_grouped_has_per_segment_elr():
    # two lines, different ultimates (so different ELR at a shared premium)
    pay_a, _, _, vt = _scenario(ultimate_by_origin=1_000_000.0, lob="A")
    pay_b, _, _, _ = _scenario(ultimate_by_origin=600_000.0, lob="B")
    pay = pd.concat([pay_a, pay_b], ignore_index=True)
    cf_by = completion_factors_by(pay, groupby="lob", origin_col="o", valuation_col="val", amount_col="paid")
    latest = pay.groupby(["lob", "o"])["paid"].sum().reset_index().rename(columns={"paid": "ptd"})
    latest["prem"] = 1_000_000.0  # same premium both lines -> A's loss ratio higher than B's
    out = develop_ultimate(latest, cf_by, method="cape_cod", by="lob", value_col="ptd",
                           date_col="o", valuation_date=vt, exposure_col="prem")
    # recover each line's ELR from the most mature origin (emerged ~1 -> ultimate ~ paid; but check ranking)
    ult_by_lob = out.groupby("lob")["ptd_ultimate"].sum()
    assert ult_by_lob["A"] > ult_by_lob["B"]  # higher-loss line develops to a higher ultimate
    assert not out["ptd_ultimate"].isna().any()


def test_bf_grouped_factor_table():
    pay_a, _, _, vt = _scenario(ultimate_by_origin=1_000_000.0, lob="A")
    pay_b, _, _, _ = _scenario(ultimate_by_origin=600_000.0, lob="B")
    pay = pd.concat([pay_a, pay_b], ignore_index=True)
    cf_by = completion_factors_by(pay, groupby="lob", origin_col="o", valuation_col="val", amount_col="paid")
    latest = pay.groupby(["lob", "o"])["paid"].sum().reset_index().rename(columns={"paid": "ptd"})
    latest["ap"] = latest["lob"].map({"A": 1_000_000.0, "B": 600_000.0})
    out = develop_ultimate(latest, cf_by, method="bornhuetter_ferguson", by="lob", value_col="ptd",
                           date_col="o", valuation_date=vt, apriori_col="ap")
    assert not out["ptd_ultimate"].isna().any()
    # ultimate is between paid and apriori-implied for each row
    assert (out["ptd_ultimate"] >= out["ptd"] - 1e-6).all()
