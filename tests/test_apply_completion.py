"""Tests for applying completion factors (apply_completion).

The decisive correctness property is that apply_completion on the latest diagonal
reproduces ChainLadder.project's per-origin ultimates exactly -- proving the lag scale,
the join, and off-by-one are right -- and that the join is by lag value, never index
alignment.
"""

import numpy as np
import pandas as pd
import pytest

from actuarialpy import (
    ChainLadder,
    apply_completion,
    completion_factors,
    make_completion_triangle,
)

_CUM = np.array([0.40, 0.70, 0.85, 0.93, 0.97, 0.99, 1.00])  # cumulative % complete by lag 0..6
_INC = np.diff(np.concatenate([[0.0], _CUM]))


def _scenario():
    """Synthetic development data with an exact, shared completion pattern."""
    origins = pd.period_range("2023-01", "2024-12", freq="M")
    val_cap = pd.Period("2024-12", freq="M")
    val_ts = val_cap.to_timestamp("M")
    t = np.arange(len(origins))
    ult_true = 1_000_000 * (1.004 ** t) * (1 + 0.05 * np.sin(2 * np.pi * origins.month.values / 12))
    rows = []
    for i, og in enumerate(origins):
        for lag in range((val_cap - og).n + 1):
            incr = ult_true[i] * _INC[lag] if lag < len(_INC) else 0.0
            rows.append({"origin": og.to_timestamp(), "valuation": (og + lag).to_timestamp("M"), "paid": incr})
    dev = pd.DataFrame(rows)
    tri = make_completion_triangle(dev, origin_col="origin", valuation_col="valuation", amount_col="paid")
    cf = completion_factors(tri)
    latest = dev.groupby("origin")["paid"].sum().reset_index().rename(columns={"paid": "claims"})
    latest["member_months"] = 10000
    key = pd.PeriodIndex(pd.to_datetime(latest["origin"]), freq="M")
    lag = np.array([(val_cap - p).n for p in key])
    return dev, tri, cf, latest, val_ts, key, lag, ult_true


def test_matches_chain_ladder_projection():
    dev, tri, cf, latest, val_ts, key, lag, ult_true = _scenario()
    comp = apply_completion(latest, cf, value_col="claims", date_col="origin", valuation_date=val_ts)
    ult_cl = ChainLadder.fit(tri).project(tri)["ultimate"].reindex(key).to_numpy()
    assert np.allclose(comp["claims_completed"].to_numpy(), ult_cl)


def test_recovers_true_ultimates_on_exact_pattern():
    dev, tri, cf, latest, val_ts, key, lag, ult_true = _scenario()
    comp = apply_completion(latest, cf, value_col="claims", date_col="origin", valuation_date=val_ts)
    assert np.allclose(comp["claims_completed"].to_numpy(), ult_true, rtol=1e-9)


def test_mature_unchanged_recent_grossed_up():
    dev, tri, cf, latest, val_ts, key, lag, ult_true = _scenario()
    comp = apply_completion(latest, cf, value_col="claims", date_col="origin", valuation_date=val_ts)
    mature = lag >= 6
    recent = lag == 0
    assert np.allclose(comp.loc[mature, "claims_completed"], comp.loc[mature, "claims"])
    ratio = (comp.loc[recent, "claims_completed"] / comp.loc[recent, "claims"]).iloc[0]
    assert abs(ratio - 1 / 0.40) < 1e-6


def test_ibnr_identity():
    dev, tri, cf, latest, val_ts, key, lag, ult_true = _scenario()
    comp = apply_completion(latest, cf, value_col="claims", date_col="origin", valuation_date=val_ts)
    proj = ChainLadder.fit(tri).project(tri)
    ibnr_col = [c for c in proj.columns if "ibnr" in c.lower()][0]
    assert np.allclose((comp["claims_completed"] - comp["claims"]).to_numpy(), proj[ibnr_col].reindex(key).to_numpy())


@pytest.mark.parametrize("index", ["reversed", "duplicated", "string"])
def test_join_is_index_safe(index):
    dev, tri, cf, latest, val_ts, key, lag, ult_true = _scenario()
    base = apply_completion(latest, cf, value_col="claims", date_col="origin", valuation_date=val_ts)
    idx_map = {
        "reversed": pd.Index(np.arange(len(latest))[::-1]),
        "duplicated": pd.Index(np.zeros(len(latest), dtype=int)),
        "string": pd.Index([f"r{i}" for i in range(len(latest))]),
    }
    adv = apply_completion(latest.set_index(idx_map[index]), cf, value_col="claims",
                           date_col="origin", valuation_date=val_ts)
    assert np.allclose(adv["claims_completed"].to_numpy(), base["claims_completed"].to_numpy())


def test_development_col_matches_date_path():
    dev, tri, cf, latest, val_ts, key, lag, ult_true = _scenario()
    by_date = apply_completion(latest, cf, value_col="claims", date_col="origin", valuation_date=val_ts)
    by_lag = apply_completion(latest.assign(lag=lag), cf, value_col="claims", development_col="lag")
    assert np.allclose(by_lag["claims_completed"].to_numpy(), by_date["claims_completed"].to_numpy())


def test_out_of_range_lag_is_complete():
    dev, tri, cf, latest, val_ts, *_ = _scenario()
    older = pd.DataFrame({"origin": [pd.Timestamp("2019-01-01")], "claims": [500.0]})
    out = apply_completion(older, cf, value_col="claims", date_col="origin", valuation_date=val_ts)
    assert np.isclose(out["claims_completed"].iloc[0], 500.0)  # factor 1.0


def test_negative_lag_raises():
    dev, tri, cf, latest, val_ts, *_ = _scenario()
    future = pd.DataFrame({"origin": [pd.Timestamp("2025-06-01")], "claims": [1.0]})
    with pytest.raises(ValueError):
        apply_completion(future, cf, value_col="claims", date_col="origin", valuation_date=val_ts)


def test_requires_lag_or_date_and_valuation():
    dev, tri, cf, latest, val_ts, *_ = _scenario()
    with pytest.raises(ValueError):
        apply_completion(latest, cf, value_col="claims")  # neither development_col nor (date_col, valuation_date)


def test_interior_missing_development_stays_nan():
    # a factor series with a hole at lag 2; a row at lag 2 must surface NaN, not be filled
    cf = pd.Series({0: 0.40, 1: 0.70, 3: 0.93}, name="completion")  # lag 2 missing, max lag 3
    df = pd.DataFrame({"lag": [0, 1, 2, 3], "claims": [100.0, 100.0, 100.0, 100.0]})
    out = apply_completion(df, cf, value_col="claims", development_col="lag")
    assert pd.isna(out["claims_completed"].iloc[2])
    assert np.isclose(out["claims_completed"].iloc[0], 250.0)  # 100 / 0.40


# --- grouped (per-segment) completion via by= ------------------------------

from actuarialpy import completion_factors_by, development_months, lag_months  # noqa: E402


def test_lag_months_alias():
    assert development_months is lag_months
    assert int(development_months(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))) == 11


_PATTERNS = {
    "A": np.array([0.40, 0.70, 0.85, 0.93, 0.97, 0.99, 1.00]),
    "B": np.array([0.25, 0.55, 0.78, 0.90, 0.96, 0.99, 1.00]),
}


def _two_lob_scenario():
    origins = pd.period_range("2023-01", "2024-12", freq="M")
    val_cap = pd.Period("2024-12", freq="M")
    val_ts = val_cap.to_timestamp("M")
    rows = []
    for lob, cum in _PATTERNS.items():
        inc = np.diff(np.concatenate([[0.0], cum]))
        t = np.arange(len(origins))
        ult = 800_000 * (1.003 ** t) * (1.2 if lob == "A" else 1.0)
        for i, og in enumerate(origins):
            for d in range((val_cap - og).n + 1):
                rows.append({"lob": lob, "o": og.to_timestamp(), "v": (og + d).to_timestamp("M"),
                             "p": ult[i] * inc[d] if d < len(inc) else 0.0})
    dev = pd.DataFrame(rows)
    cf_by = completion_factors_by(dev, groupby="lob", origin_col="o", valuation_col="v", amount_col="p")
    latest = dev.groupby(["lob", "o"])["p"].sum().reset_index().rename(columns={"p": "claims"})
    return dev, cf_by, latest, val_ts


def test_completion_factors_by_has_development_column():
    _, cf_by, _, _ = _two_lob_scenario()
    assert list(cf_by.columns) == ["lob", "development_month", "completion_factor"]


def test_grouped_completion_matches_per_group_chain_ladder():
    dev, cf_by, latest, val_ts = _two_lob_scenario()
    comp = apply_completion(latest, cf_by, value_col="claims", date_col="o", valuation_date=val_ts, by="lob")
    for lob in _PATTERNS:
        sub = dev[dev["lob"] == lob]
        tri = make_completion_triangle(sub, origin_col="o", valuation_col="v", amount_col="p")
        key = pd.PeriodIndex(pd.to_datetime(latest[latest["lob"] == lob]["o"]), freq="M")
        want = ChainLadder.fit(tri).project(tri)["ultimate"].reindex(key).to_numpy()
        got = comp[comp["lob"] == lob]["claims_completed"].to_numpy()
        assert np.allclose(got, want)


def test_grouped_completion_fanout_guard():
    dev, cf_by, latest, val_ts = _two_lob_scenario()
    dup = pd.concat([cf_by, cf_by.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        apply_completion(latest, dup, value_col="claims", date_col="o", valuation_date=val_ts, by="lob")


def test_grouped_completion_absent_group_is_nan():
    dev, cf_by, latest, val_ts = _two_lob_scenario()
    latest2 = latest.copy()
    latest2.loc[latest2.index[0], "lob"] = "Z"  # group not in factor table
    out = apply_completion(latest2, cf_by, value_col="claims", date_col="o", valuation_date=val_ts, by="lob")
    assert pd.isna(out.iloc[0]["claims_completed"])


def test_grouped_completion_beyond_group_max_is_complete():
    dev, cf_by, latest, val_ts = _two_lob_scenario()
    comp = apply_completion(latest, cf_by, value_col="claims", date_col="o", valuation_date=val_ts, by="lob")
    # the oldest origin in each LOB is fully developed -> completed == paid
    for lob in _PATTERNS:
        sub = comp[comp["lob"] == lob].sort_values("o")
        assert np.isclose(sub.iloc[0]["claims_completed"], sub.iloc[0]["claims"])

