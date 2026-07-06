"""Vectorization tests for the financial (time value of money) module.

These verify the type-mirroring contract added to the element-wise functions:
a scalar rate returns a float, a NumPy array returns an array, and a pandas
Series returns a Series with the index and name preserved -- while every
element matches the corresponding scalar call. Append to tests/test_financial.py.
"""
import numpy as np
import pandas as pd
import pytest

import actuarialpy as ap


# the element-wise functions and a scalar-argument closure for each
ELEMENTWISE = {
    "discount_factor": lambda r: ap.discount_factor(r, 10),
    "accumulation_factor": lambda r: ap.accumulation_factor(r, 10),
    "effective_discount": lambda r: ap.effective_discount(r),
    "force_of_interest": lambda r: ap.force_of_interest(r),
    "rate_from_force": lambda r: ap.rate_from_force(r),
    "nominal_interest": lambda r: ap.nominal_interest(r, 12),
    "nominal_discount": lambda r: ap.nominal_discount(r, 4),
    "rate_from_nominal_interest": lambda r: ap.rate_from_nominal_interest(r, 12),
    "annuity_immediate": lambda r: ap.annuity_immediate(r, 20),
    "annuity_due": lambda r: ap.annuity_due(r, 20),
    "accumulated_immediate": lambda r: ap.accumulated_immediate(r, 20),
    "accumulated_due": lambda r: ap.accumulated_due(r, 20),
    "annuity_continuous": lambda r: ap.annuity_continuous(r, 20),
    "annuity_immediate_mthly": lambda r: ap.annuity_immediate_mthly(r, 20, 12),
    "increasing_annuity_immediate": lambda r: ap.increasing_annuity_immediate(r, 20),
    "decreasing_annuity_immediate": lambda r: ap.decreasing_annuity_immediate(r, 20),
    "geometric_annuity_immediate": lambda r: ap.geometric_annuity_immediate(r, 20, 0.02),
    "deferred_annuity_immediate": lambda r: ap.deferred_annuity_immediate(r, 20, 5),
}

# rates spanning ordinary values plus the i == 0 edge handled by np.where limits
RATES = [0.03, 0.05, 0.07]


@pytest.mark.parametrize("name", list(ELEMENTWISE))
def test_scalar_returns_float(name):
    fn = ELEMENTWISE[name]
    out = fn(0.05)
    assert isinstance(out, float)


@pytest.mark.parametrize("name", list(ELEMENTWISE))
def test_ndarray_in_ndarray_out_matches_scalar(name):
    fn = ELEMENTWISE[name]
    arr = np.array(RATES + [0.0])  # include the zero-rate edge
    out = fn(arr)
    assert isinstance(out, np.ndarray)
    per_element = np.array([fn(float(r)) for r in arr])
    assert np.allclose(out, per_element, equal_nan=True)


@pytest.mark.parametrize("name", list(ELEMENTWISE))
def test_series_preserves_index_name_and_values(name):
    fn = ELEMENTWISE[name]
    s = pd.Series(RATES, index=["a", "b", "c"], name="rate")
    out = fn(s)
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["a", "b", "c"]
    assert out.name == "rate"
    per_element = np.array([fn(float(r)) for r in s.values])
    assert np.allclose(out.values, per_element)


def test_zero_rate_limits_vectorized():
    arr = np.array([0.0, 0.05])
    assert ap.annuity_immediate(arr, 20)[0] == pytest.approx(20.0)
    assert ap.annuity_due(arr, 20)[0] == pytest.approx(20.0)
    assert ap.accumulated_immediate(arr, 20)[0] == pytest.approx(20.0)
    assert ap.increasing_annuity_immediate(arr, 10)[0] == pytest.approx(55.0)
    assert ap.decreasing_annuity_immediate(arr, 10)[0] == pytest.approx(55.0)
    assert ap.annuity_continuous(arr, 20)[0] == pytest.approx(20.0)


def test_geometric_rate_equals_growth_limit_vectorized():
    # i == g branch: (1+g)/(1+i) == 1, limit is n/(1+i)
    arr = np.array([0.05, 0.06])
    out = ap.geometric_annuity_immediate(arr, 10, 0.05)
    assert out[0] == pytest.approx(10 / 1.05)


def test_present_and_future_value_broadcast():
    amounts = pd.Series([1000.0, 2000.0], index=["x", "y"])
    rates = pd.Series([0.03, 0.05], index=["x", "y"])
    pv = ap.present_value(amounts, rates, 10)
    assert isinstance(pv, pd.Series)
    assert list(pv.index) == ["x", "y"]
    assert pv["x"] == pytest.approx(1000.0 * 1.03**-10)


def test_dataframe_column_assignment_roundtrip():
    df = pd.DataFrame({"rate": RATES})
    df["disc"] = ap.discount_factor(df["rate"], 10)
    df["ann"] = ap.annuity_immediate(df["rate"], 20)
    assert df["disc"].iloc[1] == pytest.approx(1.05**-10)
    assert df["ann"].iloc[1] == pytest.approx((1 - 1.05**-20) / 0.05)


def test_vectorized_validation_matches_scalar():
    # an array with any entry <= -1 raises, like the scalar guard
    with pytest.raises(ValueError):
        ap.discount_factor(np.array([0.05, -1.2, 0.03]))
    with pytest.raises(ValueError):
        ap.annuity_immediate(pd.Series([0.05, -1.0]), 20)
    # perpetuities require i > 0 element-wise
    with pytest.raises(ValueError):
        ap.perpetuity_immediate(np.array([0.05, 0.0]))
    # nominal discount too large for m, element-wise
    with pytest.raises(ValueError):
        ap.rate_from_nominal_discount(np.array([0.05, 2.0]), 1)


def test_no_numpy_warnings_at_edges():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ap.annuity_immediate(np.array([0.0, 0.05]), 20)
        ap.increasing_annuity_immediate(np.array([0.0, 0.05]), 20)
        ap.geometric_annuity_immediate(np.array([0.05, 0.05]), 20, 0.05)
        ap.annuity_continuous(np.array([0.0, 0.05]), 20)


def test_reductions_still_scalar():
    # NPV / IRR / present_value_curve take a sequence and return a scalar; unchanged
    assert isinstance(ap.net_present_value(0.05, [-100, 60, 60]), float)
    assert isinstance(ap.internal_rate_of_return([-100, 60, 60]), float)
    assert isinstance(
        ap.present_value_curve([100, 100], [0.03, 0.04], [1, 2]), float
    )
