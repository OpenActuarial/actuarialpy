"""Credibility tests.

The Buhlmann and BuhlmannStraub tests are ported verbatim from lossmodels (with
the import path updated) so that behavior parity across the move is verified.
Additional tests cover the credibility_weighted_estimate primitive added in core.
"""

import numpy as np
import pandas as pd
import pytest

from actuarialpy import Buhlmann, BuhlmannStraub, credibility_weighted_estimate
from actuarialpy.credibility import Buhlmann as Buhlmann_submodule


# --------------------------------------------------------------------------- #
# credibility_weighted_estimate primitive
# --------------------------------------------------------------------------- #
def test_blend_scalar_returns_float():
    out = credibility_weighted_estimate(observed=14.0, complement=10.0, z=0.5)
    assert isinstance(out, float)
    assert out == 12.0


def test_blend_series_preserves_index():
    observed = pd.Series([14.0, 8.0], index=[3, 7])
    complement = pd.Series([10.0, 10.0], index=[3, 7])
    out = credibility_weighted_estimate(observed, complement, z=0.5)
    assert isinstance(out, pd.Series)
    assert out.index.tolist() == [3, 7]
    assert out.tolist() == [12.0, 9.0]


def test_blend_array_and_vector_z():
    out = credibility_weighted_estimate(
        observed=[14.0, 8.0], complement=[10.0, 10.0], z=[0.5, 0.25]
    )
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [12.0, 9.5])


def test_blend_matches_buhlmann_premium():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)
    risk_mean = 14.0
    assert np.isclose(
        credibility_weighted_estimate(risk_mean, model.overall_mean, model.z),
        model.premium(risk_mean),
    )


def test_top_level_and_submodule_are_same_class():
    assert Buhlmann is Buhlmann_submodule


# --------------------------------------------------------------------------- #
# Buhlmann (ported)
# --------------------------------------------------------------------------- #
def test_buhlmann_init_valid():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)
    assert model.overall_mean == 10.0
    assert model.epv == 4.0
    assert model.vhm == 2.0
    assert model.n_obs == 3


def test_buhlmann_init_invalid():
    with pytest.raises(ValueError):
        Buhlmann(overall_mean=10.0, epv=-1.0, vhm=2.0, n_obs=3)
    with pytest.raises(ValueError):
        Buhlmann(overall_mean=10.0, epv=1.0, vhm=-2.0, n_obs=3)
    with pytest.raises(ValueError):
        Buhlmann(overall_mean=10.0, epv=1.0, vhm=2.0, n_obs=0)


def test_buhlmann_k_and_z():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)
    assert np.isclose(model.k, 2.0)
    assert np.isclose(model.z, 3.0 / (3.0 + 2.0))


def test_buhlmann_k_and_z_when_vhm_zero():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=0.0, n_obs=3)
    assert model.k == float("inf")
    assert model.z == 0.0


def test_buhlmann_premium_scalar_and_vector():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)
    risk_mean = 14.0
    expected = model.z * risk_mean + (1.0 - model.z) * model.overall_mean
    assert np.isclose(model.premium(risk_mean), expected)

    risk_means = np.array([8.0, 10.0, 14.0])
    premiums = model.premium(risk_means)
    expected_vec = model.z * risk_means + (1.0 - model.z) * model.overall_mean
    assert isinstance(premiums, np.ndarray)
    assert np.allclose(premiums, expected_vec)


def test_buhlmann_fit_basic_quantities():
    data = np.array([[10.0, 12.0, 14.0], [8.0, 9.0, 10.0], [15.0, 16.0, 17.0]])
    model = Buhlmann.fit(data)
    risk_means = np.mean(data, axis=1)
    overall_mean = np.mean(data)
    epv = np.mean(np.var(data, axis=1, ddof=1))
    vhm = max(np.var(risk_means, ddof=1) - epv / data.shape[1], 0.0)
    assert np.isclose(model.overall_mean, overall_mean)
    assert np.isclose(model.epv, epv)
    assert np.isclose(model.vhm, vhm)
    assert model.n_obs == data.shape[1]


def test_buhlmann_fit_invalid():
    with pytest.raises(ValueError):
        Buhlmann.fit([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        Buhlmann.fit(np.array([[10.0, 12.0, 14.0]]))
    with pytest.raises(ValueError):
        Buhlmann.fit(np.array([[10.0], [12.0]]))


def test_buhlmann_fit_vhm_floored_at_zero():
    data = np.array([[10.0, 11.0, 9.0], [10.2, 10.8, 9.5], [9.8, 10.1, 10.3]])
    model = Buhlmann.fit(data)
    assert model.vhm >= 0.0
    assert 0.0 <= model.z <= 1.0


def test_buhlmann_repr():
    text = repr(Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3))
    assert "Buhlmann" in text
    assert "overall_mean=10.0" in text


# --------------------------------------------------------------------------- #
# BuhlmannStraub (ported)
# --------------------------------------------------------------------------- #
def test_bs_init_valid_and_invalid():
    model = BuhlmannStraub(overall_mean=10.0, epv=4.0, vhm=2.0, weights=[3.0, 5.0, 7.0])
    assert np.allclose(model.weights, [3.0, 5.0, 7.0])
    for bad in (
        dict(overall_mean=10.0, epv=-1.0, vhm=2.0, weights=[1.0, 2.0]),
        dict(overall_mean=10.0, epv=1.0, vhm=-2.0, weights=[1.0, 2.0]),
        dict(overall_mean=10.0, epv=1.0, vhm=2.0, weights=[]),
        dict(overall_mean=10.0, epv=1.0, vhm=2.0, weights=[1.0, 0.0]),
        dict(overall_mean=10.0, epv=1.0, vhm=2.0, weights=[[1.0, 2.0]]),
    ):
        with pytest.raises(ValueError):
            BuhlmannStraub(**bad)


def test_bs_k_and_z():
    model = BuhlmannStraub(overall_mean=10.0, epv=4.0, vhm=2.0, weights=[3.0, 5.0])
    assert np.isclose(model.k, 2.0)
    assert np.isclose(model.z(3.0), 3.0 / (3.0 + 2.0))
    weights = np.array([3.0, 5.0, 7.0])
    assert np.allclose(model.z(weights), weights / (weights + 2.0))


def test_bs_z_zero_when_vhm_zero():
    model = BuhlmannStraub(overall_mean=10.0, epv=4.0, vhm=0.0, weights=[3.0, 5.0])
    assert model.z(3.0) == 0.0
    assert np.allclose(model.z([3.0, 5.0]), [0.0, 0.0])


def test_bs_z_invalid_weight():
    model = BuhlmannStraub(overall_mean=10.0, epv=4.0, vhm=2.0, weights=[3.0, 5.0])
    with pytest.raises(ValueError):
        model.z(0.0)
    with pytest.raises(ValueError):
        model.z([-1.0, 2.0])


def test_bs_premium_scalar_and_vector():
    model = BuhlmannStraub(overall_mean=10.0, epv=4.0, vhm=2.0, weights=[3.0, 5.0, 7.0])
    risk_means = np.array([8.0, 10.0, 14.0])
    weights = np.array([3.0, 5.0, 7.0])
    z = weights / (weights + 2.0)
    expected = z * risk_means + (1.0 - z) * 10.0
    premiums = model.premium(risk_means, weights)
    assert isinstance(premiums, np.ndarray)
    assert np.allclose(premiums, expected)


def test_bs_fit_basic_quantities():
    data = np.array([[10.0, 12.0, 14.0], [8.0, 9.0, 10.0], [15.0, 16.0, 17.0]])
    weights = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 1.0]])
    model = BuhlmannStraub.fit(data, weights)
    risk_weights = np.sum(weights, axis=1)
    weighted_risk_means = np.sum(weights * data, axis=1) / risk_weights
    overall_mean = np.sum(weights * data) / np.sum(weights)
    assert np.isclose(model.overall_mean, overall_mean)
    assert np.allclose(model.weights, risk_weights)
    assert model.epv >= 0.0
    assert model.vhm >= 0.0
    premiums = model.premium(weighted_risk_means, risk_weights)
    for prem, rm in zip(premiums, weighted_risk_means, strict=True):
        assert min(rm, overall_mean) <= prem <= max(rm, overall_mean)


def test_bs_fit_invalid_inputs():
    data = np.array([[10.0, 12.0, 14.0], [8.0, 9.0, 10.0]])
    weights = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    with pytest.raises(ValueError):
        BuhlmannStraub.fit([1.0, 2.0, 3.0], weights)
    with pytest.raises(ValueError):
        BuhlmannStraub.fit(data, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        BuhlmannStraub.fit(data, np.array([[1.0, 2.0], [1.0, 2.0]]))
    with pytest.raises(ValueError):
        BuhlmannStraub.fit(np.array([[10.0, 12.0, 14.0]]), np.array([[1.0, 1.0, 1.0]]))
    with pytest.raises(ValueError):
        BuhlmannStraub.fit(np.array([[10.0], [12.0]]), np.array([[1.0], [1.0]]))
    with pytest.raises(ValueError):
        BuhlmannStraub.fit(data, np.array([[1.0, 2.0, 1.0], [1.0, 0.0, 2.0]]))


def test_bs_repr():
    text = repr(BuhlmannStraub(overall_mean=10.0, epv=4.0, vhm=2.0, weights=[3.0, 5.0]))
    assert "BuhlmannStraub" in text
    assert "overall_mean=10.0" in text


# --------------------------------------------------------------------------- #
# BuhlmannStraub — general estimator (unequal period counts) and from_frame
# --------------------------------------------------------------------------- #
def test_bs_fit_general_estimator_exact_unequal_periods():
    # Risk A: 2 obs, Risk B: 3 obs (ragged) -- the old equal-only fit could not
    # take this shape. Values hand-computed from the general estimators.
    risk_a = np.array([10.0, 12.0])
    risk_b = np.array([20.0, 22.0, 24.0])
    w_a = np.array([1.0, 1.0])
    w_b = np.array([1.0, 1.0, 1.0])
    model = BuhlmannStraub.fit([risk_a, risk_b], [w_a, w_b])

    # m_A=2, X̄_A=11; m_B=3, X̄_B=22; m=5; X̄=88/5=17.6
    assert model.overall_mean == pytest.approx(17.6)
    assert np.allclose(model.weights, [2.0, 3.0])
    # EPV = (2 + 8) / ((2-1)+(3-1)) = 10/3
    assert model.epv == pytest.approx(10.0 / 3.0)
    # between = 2*6.6^2 + 3*4.4^2 = 145.2 ; denom = 5 - 13/5 = 2.4
    # VHM = (145.2 - 1*10/3) / 2.4
    assert model.vhm == pytest.approx((145.2 - 10.0 / 3.0) / 2.4)
    # more exposure -> more credibility
    z = model.z(model.weights)
    assert z[1] > z[0]


def test_bs_fit_ragged_requires_two_obs_per_risk():
    with pytest.raises(ValueError):
        BuhlmannStraub.fit([np.array([10.0]), np.array([20.0, 22.0])],
                           [np.array([1.0]), np.array([1.0, 1.0])])


def test_bs_from_frame_matches_ragged_fit():
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B", "B"],
        "period": [1, 2, 1, 2, 3],
        "loss": [10.0, 12.0, 20.0, 22.0, 24.0],
        "exposure": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    model = BuhlmannStraub.from_frame(df, group="group", value="loss",
                                      weight="exposure", period="period")
    ref = BuhlmannStraub.fit([np.array([10.0, 12.0]), np.array([20.0, 22.0, 24.0])],
                             [np.array([1.0, 1.0]), np.array([1.0, 1.0, 1.0])])
    assert model.overall_mean == pytest.approx(ref.overall_mean)
    assert model.epv == pytest.approx(ref.epv)
    assert model.vhm == pytest.approx(ref.vhm)
    assert list(model.groups_) == ["A", "B"]
    assert np.allclose(model.risk_means_, [11.0, 22.0])
    assert np.allclose(model.weights, [2.0, 3.0])


def test_bs_from_frame_exposure_weighted_means():
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "loss": [100.0, 200.0, 300.0, 300.0],
        "exposure": [3.0, 1.0, 1.0, 1.0],
    })
    model = BuhlmannStraub.from_frame(df, group="group", value="loss", weight="exposure")
    # X̄_A = (3*100 + 1*200)/4 = 125 ; X̄_B = 300
    assert np.allclose(model.risk_means_, [125.0, 300.0])


def test_bs_from_frame_no_signal_floors_vhm():
    # identical group means -> VHM floored to 0 -> zero credibility
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B", "C", "C"],
        "loss": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "exposure": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
    })
    model = BuhlmannStraub.from_frame(df, group="group", value="loss", weight="exposure")
    assert model.vhm == 0.0
    assert np.allclose(model.z(model.weights), 0.0)


def test_bs_from_frame_missing_column_raises():
    df = pd.DataFrame({"group": ["A", "B"], "loss": [1.0, 2.0]})
    with pytest.raises(KeyError):
        BuhlmannStraub.from_frame(df, group="group", value="loss", weight="exposure")
