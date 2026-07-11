r"""Financial mathematics: the time-value-of-money primitives.

Every reserve, premium, and valuation discounts cash flows, so this module is
the foundation the rest of the toolkit stands on. It covers interest-rate
fundamentals and their conversions, present/accumulated values, annuities-
certain, cash-flow analysis (NPV/IRR), loan amortization, discounting against a
spot curve, and day-count year fractions.

Notation: ``i`` is the effective annual rate, ``v = 1/(1+i)`` the discount
factor, ``d = i/(1+i)`` the effective rate of discount, and ``delta = ln(1+i)``
the force of interest. Nominal rates convertible ``m`` times per year are
``i^(m)`` and ``d^(m)``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

DateLike = object


# --------------------------------------------------------------------------- #
# scalar / array-aware helpers
#
# The element-wise time-value functions below (rate conversions, discount and
# accumulation factors, present/future value, and the closed-form annuities)
# follow the same type-mirroring contract as the metric primitives: a scalar
# rate returns a Python ``float``, while a NumPy array or pandas Series returns
# a NumPy array or pandas Series with the index (and name) preserved, so a
# per-row or per-scenario rate column produces a result you can assign straight
# back onto the source frame. The genuine reductions (NPV, IRR, and the
# spot-curve present value) still take a sequence of cash flows and return a
# scalar, and the amortization schedule and day-count helpers remain scalar-
# structured, since none of those is an element-wise map over a rate.
# --------------------------------------------------------------------------- #
def _is_pandas(obj: Any) -> bool:
    return isinstance(obj, (pd.Series, pd.DataFrame))


def _as_float_scalar_or_array(x: Any):
    """Coerce to a float scalar or a float array, preserving a pandas index.

    Returns a triple ``(values, is_scalar, wrap)``. ``values`` is a Python
    ``float`` when ``x`` is scalar, otherwise a float NumPy array of the
    underlying values. ``wrap`` re-attaches a pandas Series wrapper (index and
    name) when ``x`` was pandas, and is the identity otherwise -- callers apply
    it to the computed result so scalar-in stays scalar-out and Series-in stays
    Series-out.
    """
    if _is_pandas(x):
        series = x if isinstance(x, pd.Series) else x.squeeze()
        values = np.asarray(series, dtype=float)
        index, name = series.index, series.name

        def wrap(result):
            return pd.Series(np.asarray(result, dtype=float), index=index, name=name)

        return values, False, wrap
    if isinstance(x, (int, float, np.number)):
        return float(x), True, (lambda result: float(result))
    values = np.asarray(x, dtype=float)

    def wrap(result):
        return np.asarray(result, dtype=float)

    return values, False, wrap


def _check_rate(i: Any):
    """Validate an effective rate, scalar or array-like, and return floats.

    Mirrors the scalar contract (``i`` must exceed -1) element-wise: an array
    or Series with any entry at or below -1 raises, matching the scalar error.
    """
    values, is_scalar, _ = _as_float_scalar_or_array(i)
    if is_scalar:
        if values <= -1.0:
            raise ValueError("the effective rate i must exceed -1.")
    elif np.any(values <= -1.0):
        raise ValueError("the effective rate i must exceed -1.")
    return values


def discount_factor(i: Any, t: float = 1.0) -> Any:
    r"""Discount factor :math:`v^t = (1+i)^{-t}`.

    Accepts a scalar rate, a NumPy array, or a pandas Series and returns the
    same kind (a Series keeps its index and name), so a column of rates maps to
    a column of factors.
    """
    values, _, wrap = _as_float_scalar_or_array(i)
    _check_rate(i)
    return wrap((1.0 + values) ** (-float(t)))


def accumulation_factor(i: Any, t: float = 1.0) -> Any:
    r"""Accumulation factor :math:`(1+i)^t`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    values, _, wrap = _as_float_scalar_or_array(i)
    _check_rate(i)
    return wrap((1.0 + values) ** float(t))


def effective_discount(i: Any) -> Any:
    r"""Effective rate of discount :math:`d = i/(1+i) = 1 - v`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    values = _check_rate(i)
    _, _, wrap = _as_float_scalar_or_array(i)
    return wrap(values / (1.0 + values))


def force_of_interest(i: Any) -> Any:
    r"""Force of interest :math:`\delta = \ln(1+i)`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    values = _check_rate(i)
    _, _, wrap = _as_float_scalar_or_array(i)
    return wrap(np.log1p(values))


def rate_from_force(delta: Any) -> Any:
    r"""Effective rate from the force of interest: :math:`i = e^\delta - 1`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    values, _, wrap = _as_float_scalar_or_array(delta)
    return wrap(np.expm1(values))


def nominal_interest(i: Any, m: int) -> Any:
    r"""Nominal interest convertible ``m`` times: :math:`i^{(m)} = m[(1+i)^{1/m}-1]`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    m = _check_periods(m)
    values = _check_rate(i)
    _, _, wrap = _as_float_scalar_or_array(i)
    return wrap(m * ((1.0 + values) ** (1.0 / m) - 1.0))


def nominal_discount(i: Any, m: int) -> Any:
    r"""Nominal discount convertible ``m`` times: :math:`d^{(m)} = m[1-v^{1/m}]`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    m = _check_periods(m)
    values = _check_rate(i)
    _, _, wrap = _as_float_scalar_or_array(i)
    v = 1.0 / (1.0 + values)
    return wrap(m * (1.0 - v ** (1.0 / m)))


def rate_from_nominal_interest(nominal: Any, m: int) -> Any:
    r"""Effective rate from a nominal interest rate: :math:`(1+i^{(m)}/m)^m - 1`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    m = _check_periods(m)
    values, _, wrap = _as_float_scalar_or_array(nominal)
    return wrap((1.0 + values / m) ** m - 1.0)


def rate_from_nominal_discount(nominal: Any, m: int) -> Any:
    r"""Effective rate from a nominal discount rate: :math:`(1-d^{(m)}/m)^{-m} - 1`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    m = _check_periods(m)
    values, is_scalar, wrap = _as_float_scalar_or_array(nominal)
    base = 1.0 - values / m
    if is_scalar:
        if base <= 0:
            raise ValueError("nominal discount too large for the given m.")
    elif np.any(base <= 0):
        raise ValueError("nominal discount too large for the given m.")
    return wrap(base ** (-m) - 1.0)


def _check_periods(m: int) -> int:
    m = int(m)
    if m <= 0:
        raise ValueError("the number of periods/conversions must be positive.")
    return m


# --------------------------------------------------------------------------- #
# present and future value
# --------------------------------------------------------------------------- #
def present_value(amount: Any, i: Any, t: float) -> Any:
    """Present value of a single ``amount`` due in ``t`` years.

    ``amount`` and ``i`` may be scalars, arrays, or pandas Series and broadcast
    together; a pandas operand carries its index (and name) to the result.
    """
    return amount * discount_factor(i, t)


def future_value(amount: Any, i: Any, t: float) -> Any:
    """Accumulated value of a single ``amount`` after ``t`` years.

    ``amount`` and ``i`` may be scalars, arrays, or pandas Series and broadcast
    together; a pandas operand carries its index (and name) to the result.
    """
    return amount * accumulation_factor(i, t)


# --------------------------------------------------------------------------- #
# annuities-certain
# --------------------------------------------------------------------------- #
def annuity_immediate(i: Any, n: int) -> Any:
    r"""Present value of an annuity-immediate :math:`a_{\overline{n}|}=(1-v^n)/i`.

    Accepts a scalar rate, array, or pandas Series and returns the same kind
    (a Series keeps its index and name). The :math:`i = 0` limit evaluates to
    ``n`` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    if is_scalar:
        return wrap(float(n) if values == 0 else (1.0 - (1.0 / (1.0 + values)) ** n) / values)
    safe = np.where(values == 0, 1.0, values)  # avoid divide-by-zero warning at i==0
    v = 1.0 / (1.0 + values)
    result = np.where(values == 0, float(n), (1.0 - v**n) / safe)
    return wrap(result)


def annuity_due(i: Any, n: int) -> Any:
    r"""Present value of an annuity-due :math:`\ddot a_{\overline{n}|}=(1-v^n)/d`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    The :math:`i = 0` limit evaluates to ``n`` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    imm = np.asarray(annuity_immediate(np.asarray(values), n)) if not is_scalar else annuity_immediate(values, n)
    if is_scalar:
        return wrap(float(n) if values == 0 else imm * (1.0 + values))
    return wrap(np.where(values == 0, float(n), imm * (1.0 + values)))


def accumulated_immediate(i: Any, n: int) -> Any:
    r"""Accumulated value of an annuity-immediate :math:`s_{\overline{n}|}`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    The :math:`i = 0` limit evaluates to ``n`` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    if is_scalar:
        return wrap(float(n) if values == 0 else ((1.0 + values) ** n - 1.0) / values)
    safe = np.where(values == 0, 1.0, values)
    result = np.where(values == 0, float(n), ((1.0 + values) ** n - 1.0) / safe)
    return wrap(result)


def accumulated_due(i: Any, n: int) -> Any:
    r"""Accumulated value of an annuity-due :math:`\ddot s_{\overline{n}|}`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    The :math:`i = 0` limit evaluates to ``n`` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    acc = np.asarray(accumulated_immediate(np.asarray(values), n)) if not is_scalar else accumulated_immediate(values, n)
    if is_scalar:
        return wrap(float(n) if values == 0 else acc * (1.0 + values))
    return wrap(np.where(values == 0, float(n), acc * (1.0 + values)))


def perpetuity_immediate(i: Any) -> Any:
    r"""Present value of a perpetuity-immediate :math:`1/i`.

    Requires ``i > 0`` (element-wise for array or Series input). Scalar in,
    scalar out; array or Series in, same out (index preserved).
    """
    values = _check_rate(i)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    if is_scalar:
        if values <= 0:
            raise ValueError("a perpetuity requires i > 0.")
    elif np.any(values <= 0):
        raise ValueError("a perpetuity requires i > 0.")
    return wrap(1.0 / values)


def perpetuity_due(i: Any) -> Any:
    r"""Present value of a perpetuity-due :math:`1/d`.

    Requires ``i > 0`` (element-wise for array or Series input). Scalar in,
    scalar out; array or Series in, same out (index preserved).
    """
    values = _check_rate(i)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    if is_scalar:
        if values <= 0:
            raise ValueError("a perpetuity requires i > 0.")
    elif np.any(values <= 0):
        raise ValueError("a perpetuity requires i > 0.")
    return wrap((1.0 + values) / values)


def deferred_annuity_immediate(i: Any, n: int, defer: int) -> Any:
    r"""Present value of an ``n``-year annuity-immediate deferred ``defer`` years.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    """
    if defer < 0:
        raise ValueError("defer must be non-negative.")
    return discount_factor(i, defer) * annuity_immediate(i, n)


def annuity_continuous(i: Any, n: int) -> Any:
    r"""Present value of a continuous annuity :math:`\bar a_{\overline{n}|}=(1-v^n)/\delta`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    The :math:`i = 0` limit evaluates to ``n`` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    if is_scalar:
        if values == 0:
            return wrap(float(n))
        v = 1.0 / (1.0 + values)
        return wrap((1.0 - v**n) / np.log1p(values))
    safe = np.where(values == 0, 1.0, values)  # placeholder to keep log1p finite/nonzero
    v = 1.0 / (1.0 + values)
    delta = np.log1p(safe)
    result = np.where(values == 0, float(n), (1.0 - v**n) / delta)
    return wrap(result)


def annuity_immediate_mthly(i: Any, n: int, m: int) -> Any:
    r"""Present value of an ``m``-thly annuity-immediate :math:`a^{(m)}_{\overline{n}|}`.

    Scalar in, scalar out; array or Series in, same out (index preserved).
    The :math:`i = 0` limit evaluates to ``n`` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    if is_scalar:
        if values == 0:
            return wrap(float(n))
        v = 1.0 / (1.0 + values)
        return wrap((1.0 - v**n) / nominal_interest(values, m))
    safe = np.where(values == 0, 1.0, values)
    v = 1.0 / (1.0 + values)
    im = np.asarray(nominal_interest(np.asarray(safe), m))
    result = np.where(values == 0, float(n), (1.0 - v**n) / im)
    return wrap(result)


def increasing_annuity_immediate(i: Any, n: int) -> Any:
    r"""Present value of an increasing annuity :math:`(Ia)_{\overline{n}|}`.

    Payments of 1, 2, ..., n at times 1, ..., n. Scalar in, scalar out; array
    or Series in, same out (index preserved). The :math:`i = 0` limit evaluates
    to :math:`n(n+1)/2` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    zero_limit = float(n * (n + 1) / 2)
    if is_scalar:
        if values == 0:
            return wrap(zero_limit)
        v = 1.0 / (1.0 + values)
        return wrap((annuity_due(values, n) - n * v**n) / values)
    safe = np.where(values == 0, 1.0, values)
    v = 1.0 / (1.0 + values)
    due = np.asarray(annuity_due(np.asarray(values), n))
    result = np.where(values == 0, zero_limit, (due - n * v**n) / safe)
    return wrap(result)


def decreasing_annuity_immediate(i: Any, n: int) -> Any:
    r"""Present value of a decreasing annuity :math:`(Da)_{\overline{n}|}`.

    Payments of n, n-1, ..., 1 at times 1, ..., n. Scalar in, scalar out; array
    or Series in, same out (index preserved). The :math:`i = 0` limit evaluates
    to :math:`n(n+1)/2` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    zero_limit = float(n * (n + 1) / 2)
    if is_scalar:
        if values == 0:
            return wrap(zero_limit)
        return wrap((n - annuity_immediate(values, n)) / values)
    safe = np.where(values == 0, 1.0, values)
    imm = np.asarray(annuity_immediate(np.asarray(values), n))
    result = np.where(values == 0, zero_limit, (n - imm) / safe)
    return wrap(result)


def geometric_annuity_immediate(i: Any, n: int, growth: float) -> Any:
    r"""Present value of a geometrically increasing annuity-immediate.

    Payments :math:`1, (1+g), (1+g)^2, \ldots` at times :math:`1, \ldots, n`:

    .. math::
        \frac{1 - \left(\frac{1+g}{1+i}\right)^n}{i - g}, \qquad i \neq g.

    Scalar in, scalar out; array or Series in, same out (index preserved). The
    :math:`i = g` limit evaluates to :math:`n/(1+i)` element-wise.
    """
    values = _check_rate(i)
    n = _check_term(n)
    g = float(growth)
    _, is_scalar, wrap = _as_float_scalar_or_array(i)
    if is_scalar:
        if abs(values - g) < 1e-15:
            return wrap(float(n / (1.0 + values)))
        ratio = (1.0 + g) / (1.0 + values)
        return wrap((1.0 - ratio**n) / (values - g))
    at_limit = np.abs(values - g) < 1e-15
    safe = np.where(at_limit, 1.0, values - g)  # avoid divide-by-zero at i==g
    ratio = (1.0 + g) / (1.0 + values)
    result = np.where(at_limit, n / (1.0 + values), (1.0 - ratio**n) / safe)
    return wrap(result)


def _check_term(n: int) -> int:
    n = int(n)
    if n < 0:
        raise ValueError("the term n must be non-negative.")
    return n


# --------------------------------------------------------------------------- #
# cash-flow analysis
# --------------------------------------------------------------------------- #
def net_present_value(
    rate: float,
    cashflows: Sequence[float],
    times: Sequence[float] | None = None,
) -> float:
    """Net present value of ``cashflows`` discounted at ``rate``.

    If ``times`` is omitted the cash flows are assumed to occur at times
    ``0, 1, 2, ...``.
    """
    rate = _check_rate(rate)
    cf = np.asarray(cashflows, dtype=float)
    t = np.arange(len(cf)) if times is None else np.asarray(times, dtype=float)
    if t.shape != cf.shape:
        raise ValueError("times and cashflows must have the same length.")
    return float(np.sum(cf * (1.0 + rate) ** (-t)))


def internal_rate_of_return(
    cashflows: Sequence[float],
    times: Sequence[float] | None = None,
    *,
    low: float = -0.9999,
    high: float = 1e6,
    tol: float = 1e-10,
) -> float:
    """Internal rate of return: the ``rate`` solving ``net_present_value == 0``.

    Uses a bracketed bisection over ``(low, high)``, which is robust for the
    usual single-sign-change cash-flow streams. Raises if no sign change is
    found in the search range (e.g. all-positive or all-negative flows).
    """
    cf = np.asarray(cashflows, dtype=float)
    t = np.arange(len(cf)) if times is None else np.asarray(times, dtype=float)
    if t.shape != cf.shape:
        raise ValueError("times and cashflows must have the same length.")

    def npv(r: float) -> float:
        return float(np.sum(cf * (1.0 + r) ** (-t)))

    # scan for a sign change on a log-spaced grid above -1
    grid = np.concatenate(
        [np.linspace(low, 1.0, 200), np.linspace(1.0, high, 200)[1:]]
    )
    vals = np.array([npv(r) for r in grid])
    sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if sign_change.size == 0:
        raise ValueError("no sign change in NPV over the search range; IRR not found.")

    a, b = grid[sign_change[0]], grid[sign_change[0] + 1]
    fa = npv(a)
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = npv(mid)
        if abs(fm) < tol or (b - a) < 1e-15:
            return float(mid)
        if np.sign(fm) == np.sign(fa):
            a, fa = mid, fm
        else:
            b = mid
    return float(0.5 * (a + b))


# --------------------------------------------------------------------------- #
# loans and amortization
# --------------------------------------------------------------------------- #
def level_payment(principal: Any, i: Any, n: int) -> Any:
    r"""Level payment amortizing ``principal`` over ``n`` periods at rate ``i``.

    :math:`P = L / a_{\overline{n}|}`. ``principal`` and ``i`` may be scalars,
    arrays, or pandas Series and broadcast together; a pandas operand carries
    its index (and name) to the result.
    """
    _, is_scalar, _ = _as_float_scalar_or_array(i)
    a = annuity_immediate(i, n)
    if is_scalar and a == 0:
        raise ValueError("cannot amortize over zero periods.")
    if not is_scalar and np.any(np.asarray(a) == 0):
        raise ValueError("cannot amortize over zero periods.")
    return principal / a


def outstanding_balance(principal: Any, i: Any, n: int, t: int) -> Any:
    """Prospective outstanding loan balance just after the ``t``-th payment.

    ``principal`` and ``i`` may be scalars, arrays, or pandas Series and
    broadcast together; a pandas operand carries its index (and name) through.
    """
    if not 0 <= t <= n:
        raise ValueError("t must be between 0 and n.")
    payment = level_payment(principal, i, n)
    return payment * annuity_immediate(i, n - t)


def amortization_schedule(
    principal: float, i: float, n: int, payment: float | None = None
) -> pd.DataFrame:
    """Amortization schedule with the interest/principal split and balance.

    Returns one row per period with columns ``period``, ``payment``,
    ``interest``, ``principal``, and ``balance``.
    """
    i = _check_rate(i)
    n = _check_term(n)
    pay = level_payment(principal, i, n) if payment is None else float(payment)
    rows = []
    balance = float(principal)
    for period in range(1, n + 1):
        interest = balance * i
        principal_paid = pay - interest
        balance = balance - principal_paid
        rows.append(
            {
                "period": period,
                "payment": pay,
                "interest": interest,
                "principal": principal_paid,
                "balance": balance,
            }
        )
    return pd.DataFrame(rows, columns=["period", "payment", "interest", "principal", "balance"])


# --------------------------------------------------------------------------- #
# discounting against a spot curve
# --------------------------------------------------------------------------- #
def discount_factors(spot_rates: Sequence[float], times: Sequence[float]) -> np.ndarray:
    r"""Discount factors :math:`(1+s_t)^{-t}` from spot rates at ``times``."""
    s = np.asarray(spot_rates, dtype=float)
    t = np.asarray(times, dtype=float)
    if s.shape != t.shape:
        raise ValueError("spot_rates and times must have the same length.")
    if np.any(s <= -1.0):
        raise ValueError("spot rates must exceed -1.")
    return (1.0 + s) ** (-t)


def present_value_curve(
    cashflows: Sequence[float],
    spot_rates: Sequence[float],
    times: Sequence[float],
) -> float:
    """Present value of ``cashflows`` discounted on a spot-rate curve."""
    cf = np.asarray(cashflows, dtype=float)
    df = discount_factors(spot_rates, times)
    if cf.shape != df.shape:
        raise ValueError("cashflows, spot_rates, and times must have the same length.")
    return float(np.sum(cf * df))


# --------------------------------------------------------------------------- #
# day-count year fractions
# --------------------------------------------------------------------------- #
def year_fraction(start: DateLike, end: DateLike, convention: str = "actual/365") -> float:
    """Year fraction between two dates under a day-count convention.

    Supported conventions: ``"actual/365"``, ``"actual/360"``, ``"30/360"``
    (US/NASD), and ``"actual/actual"`` (ISDA).
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    conv = convention.lower().replace(" ", "")

    if conv in ("actual/365", "act/365", "actual/365fixed"):
        return (e - s).days / 365.0
    if conv in ("actual/360", "act/360"):
        return (e - s).days / 360.0
    if conv in ("30/360", "30u/360", "bond"):
        d1, d2 = min(s.day, 30), min(e.day, 30) if s.day >= 30 else e.day
        days = 360 * (e.year - s.year) + 30 * (e.month - s.month) + (d2 - d1)
        return days / 360.0
    if conv in ("actual/actual", "act/act", "actual/actualisda"):
        if e < s:
            return -year_fraction(end, start, convention)
        if s.year == e.year:
            denom = 366.0 if s.is_leap_year else 365.0
            return (e - s).days / denom
        total = 0.0
        # leading stub to year-end
        year_end = pd.Timestamp(year=s.year, month=12, day=31)
        denom = 366.0 if s.is_leap_year else 365.0
        total += ((year_end - s).days + 1) / denom
        # whole years between
        total += e.year - s.year - 1
        # trailing stub from year-start
        year_start = pd.Timestamp(year=e.year, month=1, day=1)
        denom = 366.0 if e.is_leap_year else 365.0
        total += (e - year_start).days / denom
        return float(total)
    raise ValueError(f"unknown day-count convention: {convention!r}.")
