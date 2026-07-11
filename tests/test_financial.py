"""Tests for the financial mathematics (time value of money) module."""
import numpy as np
import pytest

import actuarialpy as ap


# ----- rate fundamentals and conversions ----- #
def test_discount_and_accumulation():
    assert ap.discount_factor(0.05, 1) == pytest.approx(1 / 1.05)
    assert ap.accumulation_factor(0.05, 2) == pytest.approx(1.1025)
    assert ap.effective_discount(0.05) == pytest.approx(0.05 / 1.05)
    assert ap.force_of_interest(0.05) == pytest.approx(np.log(1.05))
    assert ap.rate_from_force(np.log(1.05)) == pytest.approx(0.05)


def test_nominal_round_trips():
    assert ap.rate_from_nominal_interest(ap.nominal_interest(0.05, 12), 12) == pytest.approx(0.05)
    assert ap.rate_from_nominal_discount(ap.nominal_discount(0.05, 4), 4) == pytest.approx(0.05)


def test_rate_validation():
    with pytest.raises(ValueError):
        ap.discount_factor(-1.0, 1)
    with pytest.raises(ValueError):
        ap.nominal_interest(0.05, 0)


# ----- annuities-certain (textbook values at 5%) ----- #
def test_annuity_immediate_and_due():
    assert ap.annuity_immediate(0.05, 10) == pytest.approx(7.721735, rel=1e-5)
    assert ap.annuity_due(0.05, 10) == pytest.approx(8.107822, rel=1e-5)
    # due = immediate * (1 + i)
    assert ap.annuity_due(0.05, 10) == pytest.approx(ap.annuity_immediate(0.05, 10) * 1.05)


def test_accumulated_values():
    assert ap.accumulated_immediate(0.05, 10) == pytest.approx(12.577893, rel=1e-5)
    assert ap.accumulated_due(0.05, 10) == pytest.approx(12.577893 * 1.05, rel=1e-5)


def test_zero_interest_limits():
    assert ap.annuity_immediate(0.0, 10) == 10
    assert ap.annuity_due(0.0, 10) == 10
    assert ap.accumulated_immediate(0.0, 7) == 7


def test_perpetuities():
    assert ap.perpetuity_immediate(0.05) == pytest.approx(20.0)
    assert ap.perpetuity_due(0.05) == pytest.approx(21.0)


def test_deferred_annuity():
    expected = ap.discount_factor(0.05, 3) * ap.annuity_immediate(0.05, 5)
    assert ap.deferred_annuity_immediate(0.05, 5, 3) == pytest.approx(expected)


def test_continuous_annuity():
    expected = (1 - 1.05**-10) / np.log(1.05)
    assert ap.annuity_continuous(0.05, 10) == pytest.approx(expected)


def test_increasing_decreasing_identity():
    # (Ia)_n + (Da)_n = (n+1) a_n
    n, i = 4, 0.05
    ia = ap.increasing_annuity_immediate(i, n)
    da = ap.decreasing_annuity_immediate(i, n)
    assert ia == pytest.approx(8.64876, rel=1e-4)
    assert ia + da == pytest.approx((n + 1) * ap.annuity_immediate(i, n))


def test_geometric_annuity_matches_direct_sum():
    i, g, n = 0.05, 0.03, 3
    v = 1 / (1 + i)
    direct = sum((1 + g) ** (t - 1) * v**t for t in range(1, n + 1))
    assert ap.geometric_annuity_immediate(i, n, g) == pytest.approx(direct)
    # i == g limit
    assert ap.geometric_annuity_immediate(0.05, 3, 0.05) == pytest.approx(3 / 1.05)


# ----- cash-flow analysis ----- #
def test_npv():
    assert ap.net_present_value(0.1, [-100, 110]) == pytest.approx(0.0, abs=1e-9)
    assert ap.net_present_value(0.0, [-100, 50, 60]) == pytest.approx(10.0)


def test_irr_simple_and_roundtrip():
    assert ap.internal_rate_of_return([-100, 110]) == pytest.approx(0.10, rel=1e-6)
    cf = [-1000, 500, 500, 500]
    r = ap.internal_rate_of_return(cf)
    assert 0 < r < 1
    assert ap.net_present_value(r, cf) == pytest.approx(0.0, abs=1e-6)


def test_irr_no_sign_change_raises():
    with pytest.raises(ValueError):
        ap.internal_rate_of_return([100, 110, 120])


# ----- loans and amortization ----- #
def test_level_payment_and_schedule():
    principal, i, n = 1000.0, 0.05, 10
    pay = ap.level_payment(principal, i, n)
    assert pay == pytest.approx(129.504575, rel=1e-5)
    sched = ap.amortization_schedule(principal, i, n)
    assert len(sched) == n
    assert sched["balance"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert sched["principal"].sum() == pytest.approx(principal)
    assert sched["interest"].iloc[0] == pytest.approx(principal * i)
    assert np.allclose(sched["payment"], pay)


def test_outstanding_balance():
    assert ap.outstanding_balance(1000, 0.05, 10, 0) == pytest.approx(1000.0, rel=1e-6)
    assert ap.outstanding_balance(1000, 0.05, 10, 10) == pytest.approx(0.0, abs=1e-9)


# ----- curve discounting ----- #
def test_discount_curve():
    df = ap.discount_factors([0.02, 0.03], [1, 2])
    assert np.allclose(df, [1 / 1.02, 1 / 1.03**2])
    pv = ap.present_value_curve([100, 100], [0.02, 0.03], [1, 2])
    assert pv == pytest.approx(100 / 1.02 + 100 / 1.03**2)


# ----- day-count ----- #
def test_year_fraction_conventions():
    assert ap.year_fraction("2024-01-01", "2025-01-01", "actual/365") == pytest.approx(366 / 365)
    assert ap.year_fraction("2023-01-01", "2024-01-01", "actual/365") == pytest.approx(1.0)
    assert ap.year_fraction("2024-01-15", "2024-07-15", "30/360") == pytest.approx(0.5)
    assert ap.year_fraction("2023-01-01", "2023-07-01", "actual/actual") == pytest.approx(181 / 365)
    # leap-spanning actual/actual computed independently
    assert ap.year_fraction("2023-07-01", "2024-07-01", "actual/actual") == pytest.approx(
        184 / 365 + 182 / 366, rel=1e-9
    )


def test_year_fraction_unknown_raises():
    with pytest.raises(ValueError):
        ap.year_fraction("2024-01-01", "2024-02-01", "30/365")
