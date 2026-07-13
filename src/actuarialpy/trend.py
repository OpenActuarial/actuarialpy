"""Trend and projection primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.frame import Experience, resolve_amount, resolve_date, single_role_or_none
from actuarialpy.metrics import safe_divide
from actuarialpy.validation import validate_positive


def period_change(current: Any, prior: Any) -> Any:
    """Calculate period-over-period change: current / prior - 1."""
    return safe_divide(current, prior) - 1


def annualized_trend(current: Any, prior: Any, months_between: float) -> Any:
    """Annualize change between two values separated by a number of months."""
    if months_between <= 0:
        raise ValueError("months_between must be positive")
    return safe_divide(current, prior) ** (12 / months_between) - 1


def trend_factor(annual_trend: Any, months: float) -> Any:
    """Convert an annual trend rate into a trend factor over a number of months."""
    return (1 + annual_trend) ** (months / 12)


def project_forward(value: Any, annual_trend: Any, months: float) -> Any:
    """Project a value forward using an annual trend rate."""
    return value * trend_factor(annual_trend, months)


def midpoint_trend_factor(base_midpoint, projection_midpoint, annual_trend: Any) -> Any:
    """Trend factor between base and projection midpoints."""
    base = pd.to_datetime(base_midpoint)
    projection = pd.to_datetime(projection_midpoint)
    months = (projection.year - base.year) * 12 + (projection.month - base.month)
    return trend_factor(annual_trend, months)



def _date_range_mask(df: pd.DataFrame, date_col: str, start, end) -> pd.Series:
    dates = pd.to_datetime(df[date_col])
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    if end_date < start_date:
        raise ValueError("range end must be greater than or equal to range start")
    return (dates >= start_date) & (dates <= end_date)


def _comparison_masks(
    df: pd.DataFrame,
    *,
    period_col: str | None = None,
    prior_period=None,
    current_period=None,
    date_col: str | None = None,
    prior_start=None,
    prior_end=None,
    current_start=None,
    current_end=None,
    prior_filter=None,
    current_filter=None,
) -> tuple[pd.Series, pd.Series, str]:
    period_args_supplied = period_col is not None or prior_period is not None or current_period is not None
    date_args_supplied = (
        date_col is not None
        or prior_start is not None
        or prior_end is not None
        or current_start is not None
        or current_end is not None
    )
    filter_args_supplied = prior_filter is not None or current_filter is not None
    modes = sum([period_args_supplied, date_args_supplied, filter_args_supplied])
    if modes != 1:
        raise ValueError(
            "Use exactly one comparison mode: period_col/prior_period/current_period, "
            "date_col with prior/current ranges, or prior_filter/current_filter."
        )

    if period_args_supplied:
        if period_col is None or prior_period is None or current_period is None:
            raise ValueError("period_col, prior_period, and current_period must all be supplied together.")
        return df[period_col] == prior_period, df[period_col] == current_period, "period"

    if date_args_supplied:
        if None in (date_col, prior_start, prior_end, current_start, current_end):
            raise ValueError(
                "date_col, prior_start, prior_end, current_start, and current_end must all be supplied together."
            )
        assert date_col is not None  # narrowed by the guard above
        return (
            _date_range_mask(df, date_col, prior_start, prior_end),
            _date_range_mask(df, date_col, current_start, current_end),
            "date",
        )

    if prior_filter is None or current_filter is None:
        raise ValueError("prior_filter and current_filter must be supplied together.")
    return prior_filter, current_filter, "filter"

def trend_summary(
    df: pd.DataFrame | Experience,
    *,
    period_col: str | None = None,
    prior_period=None,
    current_period=None,
    date_col: str | None = None,
    prior_start=None,
    prior_end=None,
    current_start=None,
    current_end=None,
    groupby=None,
    amount_col: str | None = None,
    exposure_col: str | None = None,
    prior_filter=None,
    current_filter=None,
    prior_label: str = "prior",
    current_label: str = "current",
) -> pd.DataFrame:
    """Summarize current vs prior trend by optional grouping.

    Accepts an :class:`Experience` (bound expense / exposure / date roles fill
    ``amount_col``, ``exposure_col``, and -- for date-range comparisons --
    ``date_col``) or a plain DataFrame with ``amount_col`` named explicitly.

    Supported comparison modes:
    - ``period_col='year', prior_period=2025, current_period=2026``
    - ``date_col='incurred_date'`` with prior/current start and end dates
    - explicit boolean ``prior_filter`` and ``current_filter`` masks
    """
    if isinstance(df, Experience):
        exp = df
        df, amount_col = resolve_amount(exp, amount_col)
        if exposure_col is None:
            exposure_col = single_role_or_none(exp.exposure)
        # Bound date only fills the date-range mode; injecting it alongside
        # period_col would create two comparison modes and raise.
        if date_col is None and period_col is None:
            date_col = exp.date
    elif amount_col is None:
        raise TypeError(
            "amount_col is required when passing a DataFrame; "
            "pass an Experience to use its bound roles."
        )
    groups = as_list(groupby)
    required = groups + [amount_col] + ([exposure_col] if exposure_col else [])
    if period_col is not None:
        required.append(period_col)
    if date_col is not None:
        required.append(date_col)
    validate_columns(df, required)

    prior_filter, current_filter, mode = _comparison_masks(
        df,
        period_col=period_col,
        prior_period=prior_period,
        current_period=current_period,
        date_col=date_col,
        prior_start=prior_start,
        prior_end=prior_end,
        current_start=current_start,
        current_end=current_end,
        prior_filter=prior_filter,
        current_filter=current_filter,
    )

    def summarize(mask, label):
        # Aggregate only grouping, amount, and exposure columns. The comparison
        # column (for example, ``year``) is used only to select records and must
        # not leak into the final output as a summed numeric column such as
        # ``year_x`` / ``year_y``.
        summary_cols = groups + [amount_col] + ([exposure_col] if exposure_col else [])
        temp = df.loc[mask, summary_cols].copy()
        if groups:
            out = temp.groupby(groups, dropna=False, as_index=False).sum(numeric_only=True)
        else:
            out = pd.DataFrame({amount_col: [temp[amount_col].sum()]})
            if exposure_col:
                out[exposure_col] = temp[exposure_col].sum()
        out = out.rename(columns={amount_col: f"{label}_{amount_col}"})
        if exposure_col:
            out = out.rename(columns={exposure_col: f"{label}_{exposure_col}"})
            out[f"{label}_{amount_col}_per_{exposure_col}"] = safe_divide(
                out[f"{label}_{amount_col}"], out[f"{label}_{exposure_col}"]
            )
        return out

    prior = summarize(prior_filter, prior_label)
    current = summarize(current_filter, current_label)
    out = prior.merge(current, on=groups, how="outer") if groups else pd.concat([prior, current], axis=1)
    prior_metric = f"{prior_label}_{amount_col}_per_{exposure_col}" if exposure_col else f"{prior_label}_{amount_col}"
    current_metric = f"{current_label}_{amount_col}_per_{exposure_col}" if exposure_col else f"{current_label}_{amount_col}"
    out["trend"] = period_change(out[current_metric], out[prior_metric])
    if mode == "period":
        out.insert(len(groups), "prior_period", prior_period)
        out.insert(len(groups) + 1, "current_period", current_period)
    elif mode == "date":
        out.insert(len(groups), "prior_start", pd.to_datetime(prior_start))
        out.insert(len(groups) + 1, "prior_end", pd.to_datetime(prior_end))
        out.insert(len(groups) + 2, "current_start", pd.to_datetime(current_start))
        out.insert(len(groups) + 3, "current_end", pd.to_datetime(current_end))
    return out


def _inverse_normal_cdf(p: float) -> float:
    """Standard-normal quantile via Acklam's rational approximation (no SciPy)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - p_low:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
    r"""Regularized incomplete beta :math:`I_x(a, b)` via a Lentz continued fraction.

    The Numerical-Recipes ``betai``: exact (to machine tolerance) with no SciPy,
    used to evaluate the Student-t CDF for the quantile inversion below.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    # Continued fraction converges fast for x < (a+1)/(a+b+2); else use the symmetry.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(x, a, b) / a
    return 1.0 - front * _beta_cf(1.0 - x, b, a) / b


def _beta_cf(x: float, a: float, b: float) -> float:
    """Lentz evaluation of the beta continued fraction (helper for ``_reg_incomplete_beta``)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _student_t_cdf(t: float, df: float) -> float:
    """Student-t CDF via the regularized incomplete beta (exact, no SciPy)."""
    x = df / (df + t * t)
    ib = _reg_incomplete_beta(x, df / 2.0, 0.5)
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def _student_t_ppf(p: float, df: float) -> float:
    """Exact Student-t quantile (percent-point function).

    Uses SciPy's ``scipy.stats.t.ppf`` when SciPy is importable; otherwise falls
    back to an exact, dependency-free evaluation -- closed forms at ``df == 1``
    (Cauchy) and ``df == 2``, and bisection on the incomplete-beta CDF for
    larger ``df``. Correct to ~1e-10 across all degrees of freedom.

    Replaces an earlier Cornish-Fisher approximation that was ~11% too narrow at
    ``df == 1`` (it returned 11.30 versus the exact 12.71 at the 97.5th
    percentile) despite claiming to be conservative for small ``df`` -- which
    understated confidence intervals on the smallest trend fits (``n == 3`` gives
    ``df == 1``).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")

    try:  # opportunistic: SciPy is not a dependency of actuarialpy
        from scipy.stats import t as _scipy_t

        return float(_scipy_t.ppf(p, df))
    except ImportError:
        pass

    if p == 0.5:
        return 0.0
    if p < 0.5:  # symmetry
        return -_student_t_ppf(1.0 - p, df)

    q = p - 0.5
    if df == 1.0:  # Cauchy
        return float(math.tan(math.pi * q))
    if df == 2.0:  # exact closed form
        return float(2.0 * q * math.sqrt(2.0 / (1.0 - 4.0 * q * q)))

    # General df: bisection on the exact CDF. The normal quantile is a lower
    # bound and the (heavier-tailed) Cauchy quantile an upper bound for p > 0.5.
    lo = _inverse_normal_cdf(p)
    hi = math.tan(math.pi * q)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class TrendFit:
    """Result of :func:`fit_trend`: an exponential trend fitted to a rate series.

    ``annual_trend`` is the fitted multiplicative annual trend (``exp(slope) - 1`` on the
    log scale). ``r_squared`` is the goodness of fit, ``std_error`` the delta-method
    standard error of ``annual_trend``, and ``(ci_low, ci_high)`` its confidence interval
    (asymmetric -- the endpoints are transformed from the log-scale slope interval).
    ``slope`` and ``intercept`` describe the underlying ``log(value) = intercept + slope * t``
    fit with ``t`` measured in years from the first period.
    """

    annual_trend: float
    r_squared: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence: float
    n_periods: int
    slope: float
    intercept: float

    @property
    def ci(self) -> tuple[float, float]:
        """The confidence interval as a ``(low, high)`` tuple."""
        return (self.ci_low, self.ci_high)

    def factor(self, months: float) -> float:
        """Trend factor over ``months`` at the fitted rate: ``(1 + annual_trend) ** (months / 12)``."""
        return (1.0 + self.annual_trend) ** (months / 12.0)

    def __repr__(self) -> str:
        return (
            f"TrendFit(annual_trend={self.annual_trend:.2%}, R2={self.r_squared:.3f}, "
            f"{self.confidence:.0%} CI [{self.ci_low:.2%}, {self.ci_high:.2%}], n={self.n_periods})"
        )


def fit_trend(
    df: pd.DataFrame | Experience,
    *,
    value_col: str | None = None,
    date_col: str | None = None,
    exposure_col: str | None = None,
    freq: str = "M",
    min_periods: int = 3,
    confidence: float = 0.95,
) -> TrendFit:
    """Fit an exponential trend to a rate series by log-linear regression.

    Accepts an :class:`Experience` -- the bound expense, date, and exposure
    roles fill ``value_col``, ``date_col``, and ``exposure_col`` -- or a plain
    DataFrame with those columns named explicitly.

    Aggregates ``df`` to the ``freq`` grain (summing ``value_col`` and, if given,
    ``exposure_col``), forms the rate -- ``value / exposure`` (the per-exposure rate) when
    ``exposure_col`` is supplied, otherwise ``value`` itself -- and fits
    ``log(rate) = intercept + slope * t`` by ordinary least squares, with ``t`` in years
    from the first period. The fitted annual trend is ``exp(slope) - 1``.

    Unlike :func:`annualized_trend` (a two-point CAGR between a single current and prior
    value), this uses every period, so one noisy month does not swing the estimate, and it
    returns goodness of fit and a confidence interval -- what a *developed* (rather than
    received) trend is judged on. It does not select the trend: the window, the rate basis
    (allowed vs paid), any benefit leveraging, and the blend with external trends remain
    judgment. Run it on completed, deseasonalized history (``complete -> deseasonalize ->
    fit_trend``) so runout and seasonality do not contaminate the slope; apply the result
    with :func:`trend_factor`/:meth:`TrendFit.factor` or :func:`adjust`.

    Time is measured from actual period dates, so an occasional missing period is handled
    correctly. Requires at least ``min_periods`` distinct periods with strictly positive
    rates (non-positive values, which cannot be logged, raise). Returns a :class:`TrendFit`.
    """
    if isinstance(df, Experience):
        exp = df
        df, value_col = resolve_amount(exp, value_col)
        date_col = resolve_date(exp, date_col)
        if exposure_col is None:
            exposure_col = single_role_or_none(exp.exposure)
    elif value_col is None or date_col is None:
        raise TypeError(
            "value_col and date_col are required when passing a DataFrame; "
            "pass an Experience to use its bound roles."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    cols = [value_col, date_col] + ([exposure_col] if exposure_col else [])
    validate_columns(df, cols)

    period = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq=freq)
    work = pd.DataFrame({"_value": pd.to_numeric(df[value_col]).to_numpy()}, index=period)
    if exposure_col:
        work["_exposure"] = pd.to_numeric(df[exposure_col]).to_numpy()
    grouped = work.groupby(level=0).sum().sort_index()

    rate = grouped["_value"] / grouped["_exposure"] if exposure_col else grouped["_value"]
    rate = rate.to_numpy(dtype="float64")
    if len(rate) < max(min_periods, 3):
        raise ValueError(f"fit_trend needs at least {max(min_periods, 3)} periods; got {len(rate)}.")
    # A bare ``rate <= 0`` test would let NaN through (NaN <= 0 is False) and it would
    # then propagate silently through log(); validate_positive rejects non-finite too.
    validate_positive(rate, "fit_trend rate")

    timestamps = grouped.index.to_timestamp()
    t = (timestamps - timestamps[0]).days.to_numpy(dtype="float64") / 365.25
    if np.ptp(t) == 0:
        raise ValueError("fit_trend needs at least two distinct periods.")
    y = np.log(rate)
    n = len(y)

    t_mean, y_mean = t.mean(), y.mean()
    sxx = float(np.sum((t - t_mean) ** 2))
    sxy = float(np.sum((t - t_mean) * (y - y_mean)))
    slope = sxy / sxx
    intercept = y_mean - slope * t_mean

    residuals = y - (intercept + slope * t)
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - y_mean) ** 2))
    # a flat series has no variance to explain (sst ~ 0 up to rounding); a constant fits it
    # perfectly, so R^2 is 1.0 there rather than the unstable 0/0 of 1 - sse/sst.
    r_squared = 1.0 if sst <= 1e-12 * max(1.0, abs(y_mean)) else 1.0 - sse / sst
    resid_var = sse / (n - 2)
    slope_se = float(np.sqrt(resid_var / sxx))

    annual_trend = float(np.exp(slope) - 1.0)
    std_error = float(np.exp(slope) * slope_se)  # delta method
    t_crit = _student_t_ppf((1.0 + confidence) / 2.0, n - 2)
    ci_low = float(np.exp(slope - t_crit * slope_se) - 1.0)
    ci_high = float(np.exp(slope + t_crit * slope_se) - 1.0)

    return TrendFit(
        annual_trend=annual_trend, r_squared=r_squared, std_error=std_error,
        ci_low=ci_low, ci_high=ci_high, confidence=confidence, n_periods=n,
        slope=float(slope), intercept=float(intercept),
    )
