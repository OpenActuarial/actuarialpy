"""Seasonality and working-day adjustment for periodic (usually monthly) series.

Two complementary, dependency-free tools:

- **Working days** -- :func:`business_days_in_period` / :func:`add_business_days`
  count weekdays minus holidays in each period (US federal holidays via pandas by
  default). Dividing a paid amount by working days removes the "fewer offices open
  this month" effect, which otherwise looks like real movement.
- **Seasonal factors** -- :func:`seasonality_factors` learns one multiplier per
  calendar period (12 monthly or 4 quarterly) by the classical ratio-to-moving-
  average method, normalized to average 1.0. :func:`deseasonalize` divides the
  pattern out; :func:`apply_seasonality` multiplies it back when projecting forward.

Seasonal factors are the workhorse and stand on their own -- they already absorb
each period's *typical* working-day count. Working-day normalization is an optional
refinement for the residual year-to-year variation (an unusually holiday-heavy month,
say). When you do use both, normalize by working days first, then fit factors on the
normalized series, so the working-day effect is not mistaken for seasonality.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from actuarialpy.columns import as_list, grouped_factor_lookup, validate_columns
from actuarialpy.reserving import InsufficientDataWarning

_FREQS = {"M": 12, "Q": 4}


def _periods_per_year(freq: str) -> int:
    if freq not in _FREQS:
        raise ValueError(f"freq must be one of {sorted(_FREQS)}; got {freq!r}.")
    return _FREQS[freq]


def _season_values(dates: Any, freq: str) -> np.ndarray:
    dt = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.asarray(dt.month if freq == "M" else dt.quarter)


def _resolve_holidays(holidays: Any, start: Any, end: Any) -> np.ndarray:
    if holidays is None:
        return np.array([], dtype="datetime64[D]")
    if isinstance(holidays, str):
        if holidays == "us_federal":
            from pandas.tseries.holiday import USFederalHolidayCalendar

            hol = USFederalHolidayCalendar().holidays(start, end)
            return hol.values.astype("datetime64[D]")
        raise ValueError("holidays string must be 'us_federal' (or pass a list of dates, or None).")
    return pd.DatetimeIndex(pd.to_datetime(list(holidays))).values.astype("datetime64[D]")


def business_days_in_period(
    periods: Any,
    *,
    freq: str = "M",
    holidays: Any = "us_federal",
    weekmask: str = "Mon Tue Wed Thu Fri",
) -> pd.Series:
    """Count business days (weekdays minus holidays) in each distinct period.

    ``periods`` is any set of dates; they are mapped to their period (month or
    quarter) and de-duplicated. ``holidays`` is ``"us_federal"`` (pandas' built-in
    US federal calendar), ``None`` (weekdays only), or a list of holiday dates.
    ``weekmask`` controls which weekdays count. Returns a Series indexed by period
    start timestamp.
    """
    _periods_per_year(freq)
    pidx = pd.PeriodIndex(pd.to_datetime(periods), freq=freq).unique().sort_values()
    starts = pidx.start_time.normalize()
    ends = pidx.end_time.normalize()
    hol = _resolve_holidays(holidays, starts.min(), ends.max())
    counts = np.busday_count(
        starts.values.astype("datetime64[D]"),
        (ends + pd.Timedelta(days=1)).values.astype("datetime64[D]"),
        weekmask=weekmask,
        holidays=hol,
    )
    return pd.Series(counts, index=pidx.start_time, name="business_days")


def add_business_days(
    df: pd.DataFrame,
    date_col: str,
    *,
    freq: str = "M",
    out_col: str = "business_days",
    holidays: Any = "us_federal",
    weekmask: str = "Mon Tue Wed Thu Fri",
    copy: bool = True,
) -> pd.DataFrame:
    """Add a column with the number of business days in each row's period.

    Divide a paid-amount column by this to get an amount-per-business-day series that
    is comparable across short and long months.
    """
    validate_columns(df, [date_col])
    result = df.copy() if copy else df
    bdays = business_days_in_period(result[date_col], freq=freq, holidays=holidays, weekmask=weekmask)
    period_start = pd.PeriodIndex(pd.to_datetime(result[date_col]), freq=freq).start_time
    result[out_col] = bdays.reindex(period_start).to_numpy()
    return result


def _centered_moving_average(series: pd.Series, periods_per_year: int) -> pd.Series:
    k = periods_per_year
    if k % 2 == 0:
        # 2xk moving average: a (k+1)-window with half weight on the two endpoints
        weights = np.concatenate(([0.5], np.ones(k - 1), [0.5])) / k
        return series.rolling(window=k + 1, center=True).apply(lambda x: float(np.dot(x, weights)), raw=True)
    return series.rolling(window=k, center=True).mean()


def seasonality_factors(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    exposure_col: str | None = None,
    freq: str = "M",
    method: str = "ratio_to_moving_average",
    aggregate: str = "mean",
    exclude: Iterable[int] | None = None,
    min_years: int = 2,
) -> pd.Series:
    """Estimate seasonal factors -- one multiplier per calendar period, mean 1.0.

    The series is first aggregated to the period grain (summing ``value_col`` and, if
    given, ``exposure_col``). With ``exposure_col`` the factors are computed on the
    rate ``value / exposure`` (e.g. PMPM), which is the right basis for health
    seasonality; without it they are computed on the value directly.

    Methods:

    - ``"ratio_to_moving_average"`` (default): classical multiplicative decomposition.
      Each period is divided by a centered moving average (which removes trend and
      level), and the seasonal factor for a calendar period is the average of those
      ratios across years. Robust to trend and membership growth.
    - ``"period_share"``: each period expressed as a share of its own year's average,
      then averaged by calendar period. Simpler, but assumes little within-year trend.

    ``aggregate`` is ``"mean"`` or ``"median"`` (median is more robust to outlier
    months). ``exclude`` drops whole years from the estimate -- e.g.
    ``exclude=[2020, 2021]`` to keep COVID-distorted years out of the factors. A
    warning is raised when fewer than ``min_years`` years inform any period. Factors
    are normalized to average exactly 1.0.
    """
    cols = [date_col, value_col] + ([exposure_col] if exposure_col else [])
    validate_columns(df, cols)
    pp = _periods_per_year(freq)

    work = df[cols].copy()
    work["_period"] = pd.PeriodIndex(pd.to_datetime(work[date_col]), freq=freq)
    agg = {value_col: "sum"}
    if exposure_col:
        agg[exposure_col] = "sum"
    grouped = work.groupby("_period").agg(agg).sort_index()

    rate = grouped[value_col] / grouped[exposure_col] if exposure_col else grouped[value_col].astype(float)
    full = pd.period_range(grouped.index.min(), grouped.index.max(), freq=freq)
    rate = rate.reindex(full)
    series = pd.Series(rate.to_numpy(), index=full.to_timestamp())

    if method == "ratio_to_moving_average":
        ratio = series / _centered_moving_average(series, pp)
    elif method == "period_share":
        year_mean = series.groupby(series.index.year).transform("mean")
        ratio = series / year_mean
    else:
        raise ValueError("method must be 'ratio_to_moving_average' or 'period_share'.")

    rdf = pd.DataFrame(
        {"season": _season_values(ratio.index, freq), "year": ratio.index.year, "ratio": ratio.to_numpy()}
    ).dropna(subset=["ratio"])
    if exclude is not None:
        rdf = rdf[~rdf["year"].isin(set(as_list(exclude)))]

    per_season_years = rdf.groupby("season")["year"].nunique()
    if rdf.empty or per_season_years.min() < min_years:
        warnings.warn(
            f"Seasonal factors rest on fewer than {min_years} years for some periods; "
            "factors may be unstable. Supply more history or raise min_years.",
            InsufficientDataWarning,
            stacklevel=2,
        )

    if aggregate not in ("mean", "median"):
        raise ValueError("aggregate must be 'mean' or 'median'.")
    factors = rdf.groupby("season")["ratio"].agg(aggregate).reindex(range(1, pp + 1))
    factors = factors / factors.mean()
    factors.index.name = "season"
    factors.name = "seasonal_factor"
    return factors


def _factor_for_rows(
    df: pd.DataFrame,
    factors: pd.Series | pd.DataFrame,
    date_col: str,
    freq: str,
    *,
    by: str | list[str] | None = None,
    factor_col: str = "seasonal_factor",
    season_name: str = "season",
) -> np.ndarray:
    season = _season_values(df[date_col], freq)
    if isinstance(factors, pd.DataFrame):
        return grouped_factor_lookup(df, factors, by, season, key_col=season_name, factor_col=factor_col)
    return pd.Series(season).map(factors).to_numpy()


def deseasonalize(
    df: pd.DataFrame,
    factors: pd.Series | pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    freq: str = "M",
    by: str | list[str] | None = None,
    factor_col: str = "seasonal_factor",
    season_name: str = "season",
    out_col: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Divide ``value_col`` by each row's seasonal factor, removing the pattern.

    ``factors`` is either a flat Series indexed by season (one pattern for the frame) or
    a tidy per-segment DataFrame -- grouping column(s), a season column (``season_name``)
    and a factor column (``factor_col``), the shape :func:`seasonality_factors_by`
    returns -- joined on ``by`` plus season. The grouped join is by value (index
    irrelevant), the factor table must be unique on ``by + [season]``, and a row whose
    ``(group, season)`` is absent yields ``NaN``.
    """
    validate_columns(df, [date_col, value_col] + as_list(by))
    result = df.copy() if copy else df
    factor = _factor_for_rows(result, factors, date_col, freq, by=by, factor_col=factor_col, season_name=season_name)
    result[out_col or f"{value_col}_deseasonalized"] = result[value_col] / factor
    return result


def apply_seasonality(
    df: pd.DataFrame,
    factors: pd.Series | pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    freq: str = "M",
    by: str | list[str] | None = None,
    factor_col: str = "seasonal_factor",
    season_name: str = "season",
    out_col: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Multiply ``value_col`` by each row's seasonal factor, adding the pattern back.

    ``factors`` may be flat (Series indexed by season) or a tidy per-segment table joined
    on ``by`` plus season; see :func:`deseasonalize` for the grouped-table contract.
    """
    validate_columns(df, [date_col, value_col] + as_list(by))
    result = df.copy() if copy else df
    factor = _factor_for_rows(result, factors, date_col, freq, by=by, factor_col=factor_col, season_name=season_name)
    result[out_col or f"{value_col}_seasonalized"] = result[value_col] * factor
    return result


def seasonality_factors_by(
    df: pd.DataFrame,
    *,
    groupby: str | list[str],
    date_col: str,
    value_col: str,
    exposure_col: str | None = None,
    freq: str = "M",
    method: str = "ratio_to_moving_average",
    aggregate: str = "mean",
    exclude: Iterable[int] | None = None,
    min_years: int = 2,
    season_name: str = "season",
    warn: bool = True,
) -> pd.DataFrame:
    """Seasonal factors per segment as a tidy table.

    Fits :func:`seasonality_factors` within each segment of ``groupby`` and stacks the
    results into one row per ``(segment, season)`` -- columns are the grouping column(s),
    ``season_name``, and ``seasonal_factor`` -- the shape :func:`deseasonalize` and
    :func:`apply_seasonality` consume via ``by=``. Seasons absent from a segment's history
    are omitted for that segment (they surface as ``NaN`` on join). Set ``warn=False`` to
    silence the thin-history :class:`InsufficientDataWarning` per segment.
    """
    group_cols = as_list(groupby)
    if not group_cols:
        raise ValueError("groupby must name at least one column")
    cols = group_cols + [date_col, value_col] + ([exposure_col] if exposure_col else [])
    validate_columns(df, cols)
    by_key = group_cols[0] if len(group_cols) == 1 else group_cols

    records: list[dict[str, Any]] = []
    for key, part in df.groupby(by_key, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_cols, key_tuple, strict=True))
        with warnings.catch_warnings():
            if not warn:
                warnings.simplefilter("ignore", InsufficientDataWarning)
            factors = seasonality_factors(
                part,
                date_col=date_col,
                value_col=value_col,
                exposure_col=exposure_col,
                freq=freq,
                method=method,
                aggregate=aggregate,
                exclude=exclude,
                min_years=min_years,
            )
        for season, factor in factors.items():
            if pd.notna(factor):
                records.append({**key_map, season_name: int(season), "seasonal_factor": float(factor)})
    if not records:
        return pd.DataFrame(columns=group_cols + [season_name, "seasonal_factor"])
    return pd.DataFrame(records)
