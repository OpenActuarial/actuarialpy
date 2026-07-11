"""Property-based tests (hypothesis) pinning ActuarialPy's advertised guarantees.

These complement the example-based tests: instead of one worked case each, they assert
invariants over many randomly generated inputs --

* deseasonalize and reseasonalize are inverses;
* adjust(multiply) and adjust(divide) by the same factor are inverses;
* an audit column accumulates exactly the product of the factors applied;
* the factor join is by value and order-preserving regardless of index;
* chain-ladder completion factors are bounded in (0, 1], monotone, and reach 1.0;
* development_months counts the months added to an origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from actuarialpy import (
    adjust,
    apply_seasonality,
    completion_factors,
    deseasonalize,
    development_months,
    factor_lookup,
    make_completion_triangle,
)

SETTINGS = settings(max_examples=60, deadline=None)


def pos_floats(lo: float, hi: float):
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


@SETTINGS
@given(
    claims=st.lists(pos_floats(1.0, 1e6), min_size=12, max_size=36),
    factors=st.lists(pos_floats(0.5, 2.0), min_size=12, max_size=12),
)
def test_reseasonalize_inverts_deseasonalize(claims, factors):
    months = pd.date_range("2021-01-01", periods=len(claims), freq="MS")
    df = pd.DataFrame({"month": months, "claims": claims})
    f = pd.Series(factors, index=range(1, 13))  # season -> factor
    de = deseasonalize(df, f, date_col="month", value_col="claims")
    back = apply_seasonality(de, f, date_col="month", value_col="claims_deseasonalized", out_col="back")
    assert np.allclose(back["back"].to_numpy(), df["claims"].to_numpy(), rtol=1e-9, atol=1e-6)


@SETTINGS
@given(values=st.lists(pos_floats(1.0, 1e6), min_size=1, max_size=20), k=pos_floats(0.1, 10.0))
def test_adjust_multiply_then_divide_is_identity(values, k):
    df = pd.DataFrame({"v": values})
    up = adjust(df, k, value_col="v", how="multiply")
    back = adjust(up, k, value_col="v", how="divide")
    assert np.allclose(back["v"].to_numpy(), df["v"].to_numpy(), rtol=1e-9)


@SETTINGS
@given(
    values=st.lists(pos_floats(1.0, 1e6), min_size=1, max_size=15),
    chain=st.lists(pos_floats(0.2, 5.0), min_size=1, max_size=5),
)
def test_audit_equals_product_of_factors(values, chain):
    df = pd.DataFrame({"v": values})
    out = df
    for k in chain:
        out = adjust(out, k, value_col="v", audit_col="m")
    product = float(np.prod(chain))
    assert np.allclose(out["m"].to_numpy(), product, rtol=1e-9)
    assert np.allclose(out["v"].to_numpy(), df["v"].to_numpy() * out["m"].to_numpy(), rtol=1e-9)


@SETTINGS
@given(
    keys=st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=1, max_size=30),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_factor_lookup_is_index_independent_and_by_value(keys, seed):
    distinct = sorted(set(keys))
    factor_map = {key: 1.0 + i * 0.5 for i, key in enumerate(distinct)}
    table = pd.DataFrame({"k": list(factor_map), "factor": list(factor_map.values())})
    # an arbitrary, non-monotone index must not affect the row-order result
    rng = np.random.default_rng(seed)
    index = rng.permutation(np.arange(1000, 1000 + len(keys)))
    df = pd.DataFrame({"k": keys}, index=index)
    got = factor_lookup(df, table, "k", factor_col="factor")
    want = np.array([factor_map[key] for key in keys])
    assert np.allclose(got, want)


@SETTINGS
@given(
    increments=st.lists(pos_floats(0.01, 5.0), min_size=3, max_size=6),
    ultimates=st.lists(pos_floats(1000.0, 1e6), min_size=4, max_size=8),
)
def test_completion_factors_bounded_monotone_reach_one(increments, ultimates):
    k = len(increments)
    pattern = np.array(increments) / float(sum(increments))
    origins = pd.date_range("2020-01-01", periods=len(ultimates), freq="MS")
    rows = []
    for ultimate, origin in zip(ultimates, origins, strict=True):
        incremental = ultimate * pattern
        for d in range(k):
            rows.append({
                "o": origin,
                "val": (pd.Period(origin, "M") + d).to_timestamp("M"),
                "paid": float(incremental[d]),
            })
    triangle = make_completion_triangle(pd.DataFrame(rows), origin_col="o", valuation_col="val", amount_col="paid")
    cf = completion_factors(triangle).sort_index().to_numpy()
    assert np.all(cf > 0.0) and np.all(cf <= 1.0 + 1e-12)        # bounded in (0, 1]
    assert np.all(np.diff(cf) >= -1e-9)                          # non-decreasing in development period
    assert abs(cf[-1] - 1.0) < 1e-9                              # fully emerged by the last period (tail = 1)


@SETTINGS
@given(
    year=st.integers(min_value=2000, max_value=2030),
    month=st.integers(min_value=1, max_value=12),
    added=st.integers(min_value=0, max_value=60),
)
def test_development_months_counts_added_months(year, month, added):
    origin = pd.Timestamp(year=year, month=month, day=1)
    valuation = origin + pd.DateOffset(months=added)
    # scalar/scalar
    assert int(development_months(origin, valuation)) == added
    # Series origin against a scalar valuation (the formerly-broken mixed case)
    series_result = development_months(pd.Series([origin, origin]), valuation)
    assert series_result.tolist() == [added, added]


@SETTINGS
@given(
    g=st.floats(min_value=-0.10, max_value=0.30, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=12, max_value=48),
)
def test_fit_trend_recovers_planted_exponential_trend(g, n):
    from actuarialpy import fit_trend
    dates = pd.date_range("2021-01-01", periods=n, freq="MS")
    t = np.asarray((dates - dates[0]).days) / 365.25
    df = pd.DataFrame({"month": dates, "v": 300.0 * (1.0 + g) ** t, "e": 1.0})
    fit = fit_trend(df, value_col="v", date_col="month", exposure_col="e")
    assert abs(fit.annual_trend - g) < 1e-6          # exact recovery of a clean exponential
    assert fit.r_squared > 1 - 1e-9
    assert fit.ci_low <= fit.annual_trend <= fit.ci_high


# ----- Mack standard errors: invariants over generated triangles ----- #
from actuarialpy.reserving import ChainLadder  # noqa: E402


@st.composite
def _cumulative_triangles(draw):
    n = draw(st.integers(min_value=4, max_value=7))
    base = draw(st.lists(st.floats(min_value=50.0, max_value=5_000.0),
                         min_size=n, max_size=n))
    factors = draw(st.lists(st.floats(min_value=1.01, max_value=2.5),
                            min_size=n - 1, max_size=n - 1))
    noise = draw(st.lists(st.floats(min_value=0.85, max_value=1.15),
                          min_size=n * n, max_size=n * n))
    rows = {}
    it = iter(noise)
    for i in range(n):
        vals = [base[i]]
        for k in range(n - 1 - i):
            vals.append(vals[-1] * factors[k] * next(it))
        rows[i] = vals + [np.nan] * i
    tri = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=range(1, n + 1)).astype(float)
    tri.index.name = "origin"
    return tri


@given(_cumulative_triangles(), st.floats(min_value=0.01, max_value=1000.0))
@settings(max_examples=40, deadline=None)
def test_mack_scale_equivariance(tri, scale):
    cl = ChainLadder.fit(tri)
    a = cl.mack_standard_errors(tri)
    scaled = tri * scale
    b = ChainLadder.fit(scaled).mack_standard_errors(scaled)
    # atol anchored to the ultimates: degenerate zero-variance triangles
    # produce se = 0 exactly, and any rescaling by a non-dyadic factor turns
    # that into a one-ulp positive number -- zero for every purpose
    atol = 1e-12 * float(b["ultimate"].max())
    np.testing.assert_allclose(b["se"], a["se"] * scale, rtol=1e-9, atol=atol)
    np.testing.assert_allclose(b["ultimate"], a["ultimate"] * scale, rtol=1e-9)


@given(_cumulative_triangles())
@settings(max_examples=40, deadline=None)
def test_mack_origin_order_invariance(tri):
    cl = ChainLadder.fit(tri)
    a = cl.mack_standard_errors(tri)
    shuffled = tri.iloc[::-1]
    b = ChainLadder.fit(shuffled).mack_standard_errors(shuffled)
    atol = 1e-12 * float(a["ultimate"].max())
    np.testing.assert_allclose(b.loc[a.index.drop("Total"), "se"],
                               a.drop("Total")["se"], rtol=1e-9, atol=atol)
    np.testing.assert_allclose(b.loc["Total", "se"], a.loc["Total", "se"],
                               rtol=1e-9, atol=atol)


@given(_cumulative_triangles())
@settings(max_examples=40, deadline=None)
def test_mack_total_dominates_independent_sum(tri):
    # total mse = sum(per-origin mse) + a nonnegative estimation-covariance
    # term, so the total se can never fall below the independence value
    cl = ChainLadder.fit(tri)
    out = cl.mack_standard_errors(tri)
    per_origin = out.drop("Total")["se"].to_numpy()
    assert out.loc["Total", "se"] >= np.sqrt(np.sum(per_origin**2)) - 1e-9
