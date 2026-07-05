"""Mack (1993) standard errors: the published Taylor-Ashe results, an
independent recursion, and the estimator's structural identities."""
import numpy as np
import pandas as pd
import pytest

from actuarialpy.reserving import ChainLadder

# The Taylor & Ashe (1983) cumulative triangle, as printed in Mack (1993) --
# the canonical dataset every chain-ladder implementation is checked against.
_TA = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085, None],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315, None, None],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268, None, None, None],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311, None, None, None, None],
    [396132, 1333217, 2180715, 2985752, 3691712, None, None, None, None, None],
    [440832, 1288463, 2419861, 3483130, None, None, None, None, None, None],
    [359480, 1421128, 2864498, None, None, None, None, None, None, None],
    [376686, 1363294, None, None, None, None, None, None, None, None],
    [344014, None, None, None, None, None, None, None, None, None],
]


@pytest.fixture(scope="module")
def taylor_ashe():
    tri = pd.DataFrame(
        _TA,
        index=pd.Index(range(1, 11), name="origin"),
        columns=range(1, 11),
    ).astype(float)
    return tri, ChainLadder.fit(tri)


def test_published_taylor_ashe_results(taylor_ashe):
    tri, cl = taylor_ashe
    out = cl.mack_standard_errors(tri)
    # Mack (1993): total reserve 18,680,856 and total s.e. 2,447,095
    assert out.loc["Total", "ibnr"] == pytest.approx(18_680_856, rel=1e-6)
    assert out.loc["Total", "se"] == pytest.approx(2_447_095, rel=5e-3)
    # the paper's headline: total CV of roughly 13%
    assert out.loc["Total", "cv"] == pytest.approx(0.131, abs=0.003)


def test_matches_independent_recursion(taylor_ashe):
    """Transcribe Mack's Theorem 3/4 directly and demand exact agreement."""
    tri, cl = taylor_ashe
    out = cl.mack_standard_errors(tri)
    cols = list(tri.columns)
    f = cl.age_to_age
    sig2 = cl.mack_sigma_squared(tri)
    s_k = {
        c0: tri[[c0, c1]].dropna()[c0].sum()
        for c0, c1 in zip(cols[:-1], cols[1:])
    }
    # projected paths
    paths, ult = {}, {}
    for origin in tri.index:
        obs = tri.loc[origin].dropna()
        path = dict(obs)
        run, on = float(obs.iloc[-1]), False
        for c0, c1 in zip(cols[:-1], cols[1:]):
            if c0 == obs.index[-1]:
                on = True
            if on:
                run *= f[c0]
                path[c1] = run
        paths[origin], ult[origin] = path, run if on else float(obs.iloc[-1])
    mse = {}
    for origin in tri.index:
        obs = tri.loc[origin].dropna()
        ks = cols[cols.index(obs.index[-1]):-1]
        mse[origin] = ult[origin] ** 2 * sum(
            sig2[k] / f[k] ** 2 * (1 / paths[origin][k] + 1 / s_k[k]) for k in ks
        )
        assert out.loc[origin, "se"] == pytest.approx(np.sqrt(mse[origin]), rel=1e-10)
    total = sum(mse.values())
    origins = list(tri.index)
    for i, origin in enumerate(origins):
        obs = tri.loc[origin].dropna()
        ks = cols[cols.index(obs.index[-1]):-1]
        later = sum(ult[o] for o in origins[i + 1:])
        total += ult[origin] * later * sum(
            2 * sig2[k] / (f[k] ** 2 * s_k[k]) for k in ks
        )
    assert out.loc["Total", "se"] == pytest.approx(np.sqrt(total), rel=1e-10)


def test_scale_equivariance(taylor_ashe):
    tri, cl = taylor_ashe
    big = tri * 1000.0
    cl_big = ChainLadder.fit(big)
    a = cl.mack_standard_errors(tri)
    b = cl_big.mack_standard_errors(big)
    np.testing.assert_allclose(b["se"], a["se"] * 1000.0, rtol=1e-10)
    np.testing.assert_allclose(
        cl_big.mack_sigma_squared(big), cl.mack_sigma_squared(tri) * 1000.0,
        rtol=1e-10,
    )


def test_fully_developed_origin_has_zero_se(taylor_ashe):
    tri, cl = taylor_ashe
    out = cl.mack_standard_errors(tri)
    assert out.loc[1, "ibnr"] == 0.0
    assert out.loc[1, "se"] == 0.0
    assert np.isnan(out.loc[1, "cv"])


def test_tail_is_deterministic_by_construction(taylor_ashe):
    tri, _ = taylor_ashe
    a = ChainLadder.fit(tri, tail=1.0).mack_standard_errors(tri)
    b = ChainLadder.fit(tri, tail=1.05).mack_standard_errors(tri)
    # a deterministic tail scales every ultimate (hence every se) by exactly
    # the tail factor and contributes no variance of its own
    np.testing.assert_allclose(
        b["se"].drop("Total"), a["se"].drop("Total") * 1.05, rtol=1e-10
    )
    assert b.loc["Total", "se"] == pytest.approx(
        a.loc["Total", "se"] * 1.05, rel=1e-10
    )


def test_last_sigma_squared_uses_mack_extrapolation(taylor_ashe):
    tri, cl = taylor_ashe
    sig2 = cl.mack_sigma_squared(tri)
    cols = list(tri.columns)
    a, b = sig2[cols[-4]], sig2[cols[-3]]
    assert sig2[cols[-2]] == pytest.approx(min(b**2 / a, min(a, b)), rel=1e-12)


def test_simple_method_refuses(taylor_ashe):
    tri, _ = taylor_ashe
    cl = ChainLadder.fit(tri, method="simple")
    with pytest.raises(ValueError, match="volume"):
        cl.mack_standard_errors(tri)


def test_mismatched_triangle_refuses(taylor_ashe):
    tri, cl = taylor_ashe
    with pytest.raises(ValueError, match="same triangle"):
        cl.mack_sigma_squared(tri[[1, 2, 3]])


def test_interior_thin_pair_refuses():
    tri = pd.DataFrame(
        {1: [100.0, 110.0, 120.0], 2: [150.0, np.nan, np.nan],
         3: [165.0, np.nan, np.nan], 4: [170.0, np.nan, np.nan]},
        index=pd.Index([1, 2, 3], name="origin"),
    )
    cl = ChainLadder.fit(tri)
    with pytest.raises(ValueError, match="interior"):
        cl.mack_sigma_squared(tri)


def test_three_column_triangle_uses_fallback_sigma():
    tri = pd.DataFrame(
        {1: [100.0, 110.0, 120.0], 2: [150.0, 168.0, np.nan],
         3: [165.0, np.nan, np.nan]},
        index=pd.Index([1, 2, 3], name="origin"),
    )
    cl = ChainLadder.fit(tri)
    sig2 = cl.mack_sigma_squared(tri)
    # too small for Mack's extrapolation: the final period falls back to
    # the previous period's estimate (documented conservative choice)
    assert sig2[2] == pytest.approx(sig2[1], rel=1e-12)
    out = cl.mack_standard_errors(tri)
    assert np.isfinite(out.loc["Total", "se"]) and out.loc["Total", "se"] > 0


def test_three_by_three_hand_derivation():
    """Every Mack quantity derived by hand for a 3x3 triangle.

    f1 = (150+320)/(100+200) = 47/30;  f2 = 180/150 = 1.2
    sigma^2_1 = 100*(1.5 - 47/30)^2 + 200*(1.6 - 47/30)^2
              = 100/225 + 200/900 = 2/3        (n_k - 1 = 1)
    sigma^2_2: single ratio, 3 columns -> fallback = sigma^2_1 = 2/3
    Ultimates: o2 = 320*1.2 = 384;  o3 = 150*(47/30)*1.2 = 282
    mse(o2) = 384^2 * (2/3)/1.2^2 * (1/320 + 1/150) = 6016/9
    """
    tri = pd.DataFrame(
        {1: [100.0, 200.0, 150.0], 2: [150.0, 320.0, np.nan],
         3: [180.0, np.nan, np.nan]},
        index=pd.Index(["o1", "o2", "o3"], name="origin"),
    )
    cl = ChainLadder.fit(tri)
    assert cl.age_to_age[1] == pytest.approx(47.0 / 30.0, rel=1e-14)
    assert cl.age_to_age[2] == pytest.approx(1.2, rel=1e-14)

    sig2 = cl.mack_sigma_squared(tri)
    assert sig2[1] == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert sig2[2] == pytest.approx(2.0 / 3.0, rel=1e-12)

    out = cl.mack_standard_errors(tri)
    assert out.loc["o2", "ultimate"] == pytest.approx(384.0, rel=1e-12)
    assert out.loc["o3", "ultimate"] == pytest.approx(282.0, rel=1e-12)
    assert out.loc["o2", "se"] == pytest.approx(np.sqrt(6016.0 / 9.0), rel=1e-12)

    # o3 by the same closed formulas, both development periods
    term1 = (2 / 3) / (47 / 30) ** 2 * (1 / 150 + 1 / 300)
    term2 = (2 / 3) / 1.2**2 * (1 / 235 + 1 / 150)
    assert out.loc["o3", "se"] == pytest.approx(
        282.0 * np.sqrt(term1 + term2), rel=1e-12)

    # total = per-origin mses + the single cross term (o2, o3)
    cross = 384.0 * 282.0 * (2 * (2 / 3) / (1.2**2 * 150.0))
    total_mse = 6016.0 / 9.0 + 282.0**2 * (term1 + term2) + cross
    assert out.loc["Total", "se"] == pytest.approx(np.sqrt(total_mse), rel=1e-12)


def test_mack_calibration_under_the_models_own_generator():
    """Simulate full rectangles from the chain-ladder model itself
    (gamma increments matching Mack's mean f_k*C and variance
    sigma^2_k*C), mask, fit, and check the standardized prediction errors
    z = (actual_ult - predicted_ult) / se: mean ~ 0, sd ~ 1, ~95% within
    1.96. This validates the whole chain -- factors, sigma^2, and the mse
    recursion -- as a calibrated prediction system.

    Shape matters: a square 5x5 triangle leaves the late sigma^2 with ONE
    degree of freedom, and z built on a chi^2_1 scale estimate is wildly
    heavy-tailed (observed |z| of 77 in development). Mack's mse treats
    sigma^2 as known, so the normal reference is only owed where sigma^2
    has real df -- hence a 10x5 trapezoid, whose last development pair
    has six."""
    rng = np.random.default_rng(12)
    f_true = np.array([1.8, 1.3, 1.1, 1.05])
    sig2_true = np.array([3.0, 2.0, 1.0, 0.5])
    n_origins, n_dev = 10, 5
    zs = []
    for _ in range(250):
        full = np.empty((n_origins, n_dev))
        full[:, 0] = rng.uniform(400.0, 600.0, n_origins)
        for k in range(n_dev - 1):
            c = full[:, k]
            mean, var = f_true[k] * c, sig2_true[k] * c
            shape = mean**2 / var
            full[:, k + 1] = rng.gamma(shape, var / mean)
        tri = pd.DataFrame(full, index=pd.Index(range(n_origins),
                                                name="origin"),
                           columns=range(1, n_dev + 1))
        for i in range(n_origins - n_dev + 1, n_origins):
            tri.iloc[i, n_origins - i:] = np.nan
        cl = ChainLadder.fit(tri)
        out = cl.mack_standard_errors(tri)
        for i in range(n_origins - n_dev + 1, n_origins):
            if out.loc[i, "se"] > 0:
                zs.append((full[i, -1] - out.loc[i, "ultimate"])
                          / out.loc[i, "se"])
    zs = np.asarray(zs)
    assert abs(zs.mean()) < 0.15, zs.mean()
    assert 0.80 <= zs.std() <= 1.20, zs.std()
    within = np.mean(np.abs(zs) < 1.959964)
    assert 0.88 <= within <= 0.99, within


def test_mack_per_group_via_chain_ladder_by():
    """The grouped path: two lines with very different volatility fit by
    chain_ladder_by, Mack applied per line -- the noisier line must show
    the larger CV, and each line's numbers must equal a standalone fit."""
    from actuarialpy.reserving import chain_ladder_by, make_completion_triangle

    rng = np.random.default_rng(21)
    rows = []
    for line, sig in (("stable", 0.01), ("volatile", 0.12)):
        for origin_year in range(2020, 2025):
            origin = pd.Timestamp(f"{origin_year}-01-01")
            c = 1_000.0
            rows.append((line, origin, origin, c))  # first payment
            for lag in range(1, 2025 - origin_year):
                new_cum = c * (1.5 if lag == 1 else 1.1) * (
                    1 + rng.normal(0.0, sig))
                rows.append((line, origin,
                             origin + pd.DateOffset(months=12 * lag),
                             new_cum - c))  # incremental paid
                c = new_cum
    df = pd.DataFrame(rows, columns=["line", "origin", "valuation", "paid"])
    patterns = chain_ladder_by(df, groupby="line", origin_col="origin",
                               valuation_col="valuation", amount_col="paid")
    cvs = {}
    for line, cl in patterns.items():
        tri = make_completion_triangle(
            df[df["line"] == line], origin_col="origin",
            valuation_col="valuation", amount_col="paid")
        out = cl.mack_standard_errors(tri)
        cvs[line] = float(out.loc["Total", "cv"])
        # per-line result identical to a standalone fit on that line
        alone = ChainLadder.fit(tri).mack_standard_errors(tri)
        pd.testing.assert_frame_equal(out, alone)
    assert cvs["volatile"] > 3 * cvs["stable"]
