"""Reserving and claims-development tools.

Claims-development primitives that sit upstream of experience analysis: development
period measurement, development (completion) triangles, the IBNR identity, and
completion-factor validation.

ActuarialPy keeps factor *estimation* and factor *application* separate. The work of
turning transactional or development data into a triangle, measuring the development
period, and the completed/paid identity lives here, alongside :func:`completion_factors`.
Applying a factor is a single multiplication, but it hinges on a join -- each row's
development period matched to the right factor -- and a factor arriving in an arbitrary
external table can be joined many ways. :func:`apply_completion` therefore commits to one
well-defined contract: factors keyed by development period, each row's development period
taken as ``development_months(incurred, valuation)`` (or an explicit ``development_col``),
joined by value so the frame's index is irrelevant and a convention mismatch surfaces as
``NaN`` rather than silent corruption. Factors from this module's own pipeline satisfy
that contract by
construction; estimate them here, then complete in your pipeline or via
``Experience.complete``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from actuarialpy.columns import as_list, grouped_factor_lookup, validate_columns


def development_months(incurred_date, valuation_date):
    """Whole months of development between incurred (origin) and valuation.

    Either argument may be a scalar, a Series, or array-like, in any combination
    (e.g. a column of incurred dates against a single valuation date). The result is
    a Series when either argument is a Series, otherwise a scalar.
    """
    incurred = pd.to_datetime(incurred_date)
    valuation = pd.to_datetime(valuation_date)

    def year_month(value):
        accessor = value.dt if hasattr(value, "dt") else value  # Series use .dt; Timestamp/Index don't
        return accessor.year, accessor.month

    incurred_year, incurred_month = year_month(incurred)
    valuation_year, valuation_month = year_month(valuation)
    return (valuation_year - incurred_year) * 12 + (valuation_month - incurred_month)


# Backwards-compatible alias: "development" is the preferred cross-domain term.
lag_months = development_months


def ibnr(completed, paid):
    """IBNR as completed minus paid (the completed/paid identity).

    Works element-wise on scalars or Series. ``completed`` and ``paid`` must be on
    the same basis; the result is the amount bridging paid-to-date to ultimate.
    """
    return completed - paid


def validate_completion_factors(
    factors: pd.DataFrame, factor_col: str = "completion_factor", *, method: str = "divide"
) -> None:
    """Validate completion-factor values for a selected convention.

    ``divide`` factors (completed = paid / factor) should satisfy
    ``0 < factor <= 1``; ``multiply`` factors (completed = paid * factor) should
    satisfy ``factor >= 1``. Useful as a sanity check on estimated factors before
    they are applied upstream.
    """
    validate_columns(factors, [factor_col])
    values = factors[factor_col]
    bad_missing = values.isna()
    if bad_missing.any():
        raise ValueError(f"{bad_missing.sum()} completion factors are missing")
    if method == "divide":
        bad = (values <= 0) | (values > 1)
        if bad.any():
            raise ValueError("divide-method completion factors should generally satisfy 0 < factor <= 1")
    elif method == "multiply":
        bad = values < 1
        if bad.any():
            raise ValueError("multiply-method completion factors should generally satisfy factor >= 1")
    else:
        raise ValueError("method must be either 'divide' or 'multiply'")


def make_completion_triangle(
    df: pd.DataFrame,
    *,
    origin_col: str,
    valuation_col: str,
    amount_col: str,
    cumulative: bool = True,
    index_name: str = "origin_period",
    development_name: str = "development_month",
) -> pd.DataFrame:
    """Build a development (completion) triangle by origin period and development period.

    Each cell aggregates ``amount_col`` for an origin month at a given valuation
    development period (whole months between origin and valuation, via :func:`development_months`).
    ``amount_col`` is treated as the *incremental* amount in each (origin, development period)
    cell; with ``cumulative=True`` -- the default, and the usual basis for
    estimating development/completion factors -- the cells are accumulated across
    development period. Set ``cumulative=False`` to return the incremental triangle, or if your
    input amounts are already cumulative-to-date snapshots.

    This consumes a compact development aggregate (one row per origin x valuation,
    i.e. months x months); it does not require transaction/line-level data.
    """
    validate_columns(df, [origin_col, valuation_col, amount_col])
    temp = df.copy()
    temp[index_name] = pd.to_datetime(temp[origin_col]).dt.to_period("M")
    temp[development_name] = development_months(temp[origin_col], temp[valuation_col])
    grouped = temp.groupby([index_name, development_name], dropna=False)[amount_col].sum().reset_index()
    triangle = grouped.pivot(index=index_name, columns=development_name, values=amount_col).sort_index(axis=1)
    if cumulative:
        triangle = triangle.cumsum(axis=1)
    return triangle


@dataclass(frozen=True)
class ChainLadder:
    """Chain-ladder development pattern fitted from a cumulative triangle.

    Fit with :meth:`fit` from a cumulative development triangle (for example the
    output of :func:`make_completion_triangle` with ``cumulative=True``):

    - ``age_to_age`` -- link (age-to-age) factors, indexed by their starting development period.
    - ``cdf`` -- cumulative development factor to ultimate by development period, including the
      tail.
    - ``completion_factors`` -- ``1 / cdf`` by development period: the proportion of ultimate
      emerged by each development period. These are divide-convention factors in ``(0, 1]``
      (``completed = paid / factor``), so they line up with
      :func:`validate_completion_factors` and downstream completion.

    Use :meth:`project` to apply the pattern to a triangle and get per-origin
    ultimate and IBNR.
    """

    age_to_age: pd.Series
    cdf: pd.Series
    completion_factors: pd.Series
    tail: float
    method: str

    @classmethod
    def fit(cls, triangle: pd.DataFrame, *, method: str = "volume", tail: float = 1.0) -> ChainLadder:
        """Estimate the development pattern from a cumulative triangle.

        ``method`` is ``"volume"`` (volume-weighted age-to-age factors, the
        default) or ``"simple"`` (straight average of individual link ratios).
        ``tail`` (>= 1) extends development beyond the latest observed development period.
        """
        if method not in ("volume", "simple"):
            raise ValueError("method must be 'volume' or 'simple'")
        if tail < 1.0:
            raise ValueError("tail must be >= 1.0")
        if not isinstance(triangle, pd.DataFrame):
            raise TypeError("triangle must be a pandas DataFrame")

        tri = triangle.sort_index(axis=1)
        cols = list(tri.columns)
        if len(cols) < 2:
            raise ValueError("triangle must have at least two development periods")
        if tri.shape[0] < 2:
            raise ValueError("triangle must have at least two origin periods")

        # age-to-age (link) factors between each pair of adjacent development periods
        ratios: dict[object, float] = {}
        for start, end in zip(cols[:-1], cols[1:], strict=True):
            pair = tri[[start, end]].dropna()
            if pair.empty:
                raise ValueError(f"no overlapping origins to estimate the {start}->{end} development factor")
            if method == "volume":
                start_sum = float(pair[start].sum())
                if start_sum == 0:
                    raise ValueError(f"zero cumulative at development period {start}; cannot estimate {start}->{end} factor")
                ratios[start] = float(pair[end].sum()) / start_sum
            else:
                ratios[start] = float((pair[end] / pair[start]).mean())
        age_to_age = pd.Series(ratios, name="age_to_age")

        # cumulative development factors to ultimate (with tail), accumulating back
        cdf_vals: dict[object, float] = {cols[-1]: float(tail)}
        running = float(tail)
        for start in reversed(cols[:-1]):
            running *= age_to_age[start]
            cdf_vals[start] = running
        cdf = pd.Series(cdf_vals, name="cdf").reindex(cols)

        completion = (1.0 / cdf).rename("completion_factor")
        return cls(
            age_to_age=age_to_age,
            cdf=cdf,
            completion_factors=completion,
            tail=float(tail),
            method=method,
        )

    def project(self, triangle: pd.DataFrame) -> pd.DataFrame:
        """Project ultimate and IBNR per origin by applying the fitted pattern.

        For each origin, takes its latest observed cumulative amount and multiplies
        by the cumulative development factor at that development period. Returns one row per origin
        with the latest development period, latest cumulative, development factor applied,
        ultimate, and IBNR (ultimate minus latest).
        """
        tri = triangle.sort_index(axis=1)
        records: list[dict[str, float]] = []
        origins: list[object] = []
        for origin, row in tri.iterrows():
            observed = row.dropna()
            if observed.empty:
                continue
            latest_development = max(observed.index)
            if latest_development not in self.cdf.index:
                raise ValueError(f"no development factor for development period {latest_development}; fit on a matching triangle")
            latest = float(observed.loc[latest_development])
            factor = float(self.cdf.loc[latest_development])
            ultimate = latest * factor
            origins.append(origin)
            records.append({
                "latest_development": latest_development,
                "latest": latest,
                "development_factor": factor,
                "ultimate": ultimate,
                "ibnr": ultimate - latest,
            })
        return pd.DataFrame.from_records(records, index=pd.Index(origins, name=tri.index.name))


    def mack_sigma_squared(self, triangle: pd.DataFrame) -> pd.Series:
        r"""Mack's variance parameters :math:`\sigma_k^2` per development period.

        The chain-ladder variance assumption is
        :math:`\mathrm{Var}(C_{i,k+1} \mid C_{i,k}) = \sigma_k^2\, C_{i,k}`;
        the unbiased estimator (Mack, 1993) is

        .. math::
            \hat\sigma_k^2 = \frac{1}{n_k - 1} \sum_i C_{i,k}
                \left( \frac{C_{i,k+1}}{C_{i,k}} - \hat f_k \right)^2 .

        The final development period has only one observed link ratio, so
        its :math:`\sigma^2` cannot be estimated; Mack's log-linear
        extrapolation is used:
        :math:`\hat\sigma_{K-1}^2 = \min(\hat\sigma_{K-2}^4 / \hat\sigma_{K-3}^2,
        \min(\hat\sigma_{K-3}^2, \hat\sigma_{K-2}^2))`.

        Only defined for ``method="volume"`` -- Mack's model *is* the
        volume-weighted estimator; the assumptions do not describe the
        simple-average factors.
        """
        if self.method != "volume":
            raise ValueError(
                "Mack standard errors are defined for the volume-weighted "
                "chain ladder only (method='volume'); the variance model "
                "does not describe simple-average factors"
            )
        tri = triangle.sort_index(axis=1)
        cols = list(tri.columns)
        if list(self.age_to_age.index) != cols[:-1]:
            raise ValueError(
                "triangle development periods do not match the fitted pattern; "
                "pass the same triangle the pattern was fit on"
            )
        sig2: dict[object, float] = {}
        for start, end in zip(cols[:-1], cols[1:], strict=True):
            pair = tri[[start, end]].dropna()
            n_k = len(pair)
            f_k = float(self.age_to_age[start])
            if n_k >= 2:
                ratios = pair[end] / pair[start]
                sig2[start] = float(
                    (pair[start] * (ratios - f_k) ** 2).sum() / (n_k - 1)
                )
            elif start == cols[-2] and len(cols) >= 4:
                s_prev = sig2[cols[-3]]
                s_prev2 = sig2[cols[-4]]
                if s_prev2 > 0:
                    sig2[start] = float(
                        min(s_prev**2 / s_prev2, min(s_prev2, s_prev))
                    )
                else:
                    sig2[start] = 0.0
            elif start == cols[-2]:
                # triangle too small to extrapolate; conservative fallback
                sig2[start] = sig2.get(cols[-3], 0.0)
            else:
                raise ValueError(
                    f"only one observed link ratio for interior development "
                    f"period {start!r}; Mack variance cannot be estimated "
                    "(triangle too thin)"
                )
        return pd.Series(sig2, name="sigma_squared").reindex(cols[:-1])

    def mack_standard_errors(self, triangle: pd.DataFrame) -> pd.DataFrame:
        r"""Per-origin and total reserve standard errors (Mack, 1993).

        The distribution-free chain-ladder mean squared error: for origin
        ``i`` with ultimate :math:`\hat C_{iK}`,

        .. math::
            \widehat{\mathrm{mse}}_i = \hat C_{iK}^2 \sum_k
                \frac{\hat\sigma_k^2}{\hat f_k^2}
                \left( \frac{1}{\hat C_{ik}} + \frac{1}{S_k} \right),

        summing over the unobserved development periods, with :math:`S_k`
        the column sum entering :math:`\hat f_k`; the total adds Mack's
        cross-origin covariance term (estimation error is shared, process
        error is not), computed pairwise over the development periods
        unobserved by both origins of each pair, so the result does not
        depend on the triangle's row order.

        Returns one row per origin plus ``"Total"``: ``latest``,
        ``ultimate``, ``ibnr``, ``se``, ``cv`` (``se / ibnr``). ``se`` is
        conditional on the fitted tail -- a ``tail`` factor beyond the
        triangle carries no estimated variance and is treated as
        deterministic (stated here rather than hidden).
        """
        tri = triangle.sort_index(axis=1)
        cols = list(tri.columns)
        sig2 = self.mack_sigma_squared(tri)
        f = self.age_to_age
        # S_k: the denominator of the volume-weighted factor at k
        s_k: dict[object, float] = {}
        for start, end in zip(cols[:-1], cols[1:], strict=True):
            pair = tri[[start, end]].dropna()
            s_k[start] = float(pair[start].sum())

        proj = self.project(tri)
        origins = list(proj.index)
        # forward-fill each origin's projected path C_hat at every period
        paths: dict[object, dict[object, float]] = {}
        for origin in origins:
            observed = tri.loc[origin].dropna()
            latest_dev = max(observed.index)
            path = {dev: float(observed[dev]) for dev in observed.index}
            running = float(observed[latest_dev])
            started = False
            for start in cols[:-1]:
                if start == latest_dev:
                    started = True
                if started:
                    running = running * float(f[start])
                    end = cols[cols.index(start) + 1]
                    path[end] = running
            paths[origin] = path

        mse: dict[object, float] = {}
        for origin in origins:
            observed = tri.loc[origin].dropna()
            latest_dev = max(observed.index)
            start_idx = cols.index(latest_dev)
            ult = float(proj.loc[origin, "ultimate"])
            total = 0.0
            for start in cols[start_idx:-1]:
                total += (sig2[start] / f[start] ** 2) * (
                    1.0 / paths[origin][start] + 1.0 / s_k[start]
                )
            mse[origin] = ult**2 * total

        # total mse: sum of per-origin terms plus the pairwise estimation
        # covariance. For each origin pair the shared-f_k covariance runs
        # over the development periods unobserved by *both* origins, i.e.
        # from max(latest_i, latest_j) -- computed pairwise so the result
        # is independent of the triangle's row order (with rows sorted
        # most-developed first this reduces to the familiar textbook loop)
        total_mse = float(sum(mse.values()))
        h = [2.0 * sig2[c] / (f[c] ** 2 * s_k[c]) for c in cols[:-1]]
        suffix = np.concatenate([np.cumsum(h[::-1])[::-1], [0.0]])
        start_idx = {
            origin: cols.index(max(tri.loc[origin].dropna().index))
            for origin in origins
        }
        ults = {origin: float(proj.loc[origin, "ultimate"]) for origin in origins}
        for a_pos, oi in enumerate(origins):
            for oj in origins[a_pos + 1 :]:
                k0 = max(start_idx[oi], start_idx[oj])
                total_mse += ults[oi] * ults[oj] * float(suffix[k0])

        out = proj[["latest", "ultimate", "ibnr"]].copy()
        out["se"] = np.sqrt(pd.Series(mse).reindex(out.index).to_numpy())
        totals = pd.DataFrame(
            {
                "latest": [out["latest"].sum()],
                "ultimate": [out["ultimate"].sum()],
                "ibnr": [out["ibnr"].sum()],
                "se": [np.sqrt(total_mse)],
            },
            index=pd.Index(["Total"], name=out.index.name),
        )
        out = pd.concat([out, totals])
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cv"] = np.where(out["ibnr"] > 0, out["se"] / out["ibnr"], np.nan)
        return out


def completion_factors(triangle: pd.DataFrame, *, method: str = "volume", tail: float = 1.0) -> pd.Series:
    """Completion factors by development period, via chain-ladder.

    Convenience wrapper around :class:`ChainLadder`: returns the proportion of
    ultimate emerged by each development period (``1 / cdf``) estimated from a cumulative
    triangle. Divide-convention factors in ``(0, 1]`` (``completed = paid /
    factor``). See :class:`ChainLadder` for the full pattern and per-origin
    ultimate/IBNR.
    """
    return ChainLadder.fit(triangle, method=method, tail=tail).completion_factors


def _emerged_factor(
    df: pd.DataFrame,
    factors: pd.Series | pd.DataFrame,
    *,
    date_col: str | None,
    valuation_date: Any,
    development_col: str | None,
    by_cols: list[str],
    factor_col: str,
    development_name: str,
) -> np.ndarray:
    """Per-row completion factor (proportion emerged), with the join and tail rule.

    Each row's development period is taken from ``development_col`` or computed as
    ``development_months(date, valuation_date)``; the factor is then joined by value
    (flat Series or per-segment table) exactly as in :func:`apply_completion`. A row past
    its (group's) last development period is fully emerged (``1.0``); an interior gap or
    absent group stays ``NaN``; a negative development period raises.
    """
    if development_col is not None:
        development = pd.to_numeric(df[development_col]).to_numpy()
    else:
        valuation = pd.Series(pd.to_datetime(valuation_date), index=df.index)
        development = development_months(df[date_col], valuation).to_numpy()
    if (development < 0).any():
        raise ValueError("Negative development period: some rows have an incurred date after valuation_date.")

    if isinstance(factors, pd.DataFrame):
        factor = grouped_factor_lookup(
            df, factors, by_cols, development, key_col=development_name, factor_col=factor_col
        )
        by_key = by_cols[0] if len(by_cols) == 1 else by_cols
        group_max = factors.groupby(by_key)[development_name].max()
        if len(by_cols) == 1:
            row_max = group_max.reindex(df[by_cols[0]].to_numpy()).to_numpy()
        else:
            row_max = group_max.reindex(pd.MultiIndex.from_frame(df[by_cols].reset_index(drop=True))).to_numpy()
        beyond = np.isnan(factor) & (development > row_max)  # absent group -> row_max NaN -> stays NaN
        factor[beyond] = 1.0
    else:
        max_development = int(pd.Index(factors.index).max())
        factor = np.array(pd.Series(development).map(factors), dtype="float64")  # NaN where absent
        factor[development > max_development] = 1.0  # beyond the fitted triangle -> complete
    return factor


def _cape_cod_elr(
    paid: np.ndarray, exposure: np.ndarray, emerged: np.ndarray, df: pd.DataFrame, by_cols: list[str]
) -> np.ndarray:
    """Cape Cod expected loss ratio = sum(paid) / sum(exposure * emerged), per segment.

    The Stanard-Buhlmann "used-up premium" ELR: a single loss ratio per ``by`` segment
    (or one overall) derived from the data, broadcast back to each row. Rows whose factor
    is ``NaN`` are excluded from the ratio (and stay ``NaN`` in the result).
    """
    used = exposure * emerged
    valid = ~np.isnan(used)
    frame = pd.DataFrame({
        "_paid": np.where(valid, paid, np.nan),
        "_used": np.where(valid, used, np.nan),
    })
    if not by_cols:
        return np.full(len(paid), np.nansum(frame["_paid"].to_numpy()) / np.nansum(frame["_used"].to_numpy()))
    for col in by_cols:
        frame[col] = df[col].to_numpy()
    grouped = frame.groupby(by_cols, dropna=False)
    elr_by_group = grouped["_paid"].sum(min_count=1) / grouped["_used"].sum(min_count=1)
    if len(by_cols) == 1:
        return df[by_cols[0]].map(elr_by_group).to_numpy(dtype="float64")
    keys = pd.MultiIndex.from_frame(df[by_cols].reset_index(drop=True))
    return np.array(elr_by_group.reindex(keys), dtype="float64")


def apply_completion(
    df: pd.DataFrame,
    factors: pd.Series | pd.DataFrame,
    *,
    value_col: str,
    date_col: str | None = None,
    valuation_date: Any = None,
    development_col: str | None = None,
    by: str | list[str] | None = None,
    factor_col: str = "completion_factor",
    development_name: str = "development_month",
    out_col: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Develop a paid amount to estimated ultimate with completion factors.

    For each row the development period is taken from ``development_col`` if supplied,
    otherwise computed as ``development_months(df[date_col], valuation_date)`` -- the
    convention :func:`make_completion_triangle` uses, so factors from
    :func:`completion_factors` or :func:`completion_factors_by` join by construction.
    The completed amount is ``paid / factor`` (the divide convention, factors in
    ``(0, 1]``).

    ``factors`` may be either of:

    - a flat Series indexed by development period (one pattern for the whole frame), or
    - a tidy DataFrame of per-segment factors -- grouping column(s), a development-period
      column (``development_name``) and a factor column (``factor_col``), the shape
      :func:`completion_factors_by` returns -- joined on ``by`` plus development period.
      The table must be unique on ``by + [development]`` (a duplicate would fan out the
      data); this is checked.

    The join is by value, never index alignment, so the frame's own index is irrelevant.
    A row past its (group's) largest development period is taken as fully complete
    (factor ``1.0``); a development period inside the fitted range but absent stays
    ``NaN`` -- a surfaced gap; a row whose group is absent from the factor table stays
    ``NaN``; a negative development period (incurred after ``valuation_date``) raises.
    Supply either ``development_col``, or both ``date_col`` and ``valuation_date``.
    """
    if development_col is None and (date_col is None or valuation_date is None):
        raise ValueError(
            "Provide development_col, or both date_col and valuation_date, to determine each row's development period."
        )
    by_cols = as_list(by)
    needed = [value_col] + ([development_col] if development_col is not None else [date_col]) + by_cols
    validate_columns(df, needed)
    result = df.copy() if copy else df

    factor = _emerged_factor(
        result, factors, date_col=date_col, valuation_date=valuation_date, development_col=development_col,
        by_cols=by_cols, factor_col=factor_col, development_name=development_name,
    )
    result[out_col or f"{value_col}_completed"] = result[value_col].to_numpy() / factor
    return result


def develop_ultimate(
    df: pd.DataFrame,
    factors: pd.Series | pd.DataFrame,
    *,
    method: str = "bornhuetter_ferguson",
    value_col: str,
    date_col: str | None = None,
    valuation_date: Any = None,
    development_col: str | None = None,
    apriori_col: str | None = None,
    exposure_col: str | None = None,
    by: str | list[str] | None = None,
    factor_col: str = "completion_factor",
    development_name: str = "development_month",
    out_col: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Develop a paid amount to estimated ultimate by a chosen reserving method.

    All methods share one input -- the proportion emerged at each row's development
    period, joined exactly as :func:`apply_completion` does (flat Series or per-segment
    table, beyond-the-triangle rows fully emerged). They differ only in how they combine
    that with the paid-to-date and an *a priori* expectation:

    - ``"chain_ladder"`` -- ``paid / emerged``. Ignores the a priori; equivalent to
      :func:`apply_completion`. Volatile for immature periods (a thin latest diagonal
      drives the whole tail).
    - ``"bornhuetter_ferguson"`` -- ``paid + apriori * (1 - emerged)``. Takes the
      unemerged portion from the a priori rather than from the data, so it is stable for
      green periods. Requires ``apriori_col`` (an expected ultimate per row -- an input,
      e.g. a plan, budget, or manual times exposure).
    - ``"benktander"`` -- one Bornhuetter-Ferguson iteration using the BF ultimate as the
      a priori: ``paid + bf * (1 - emerged)``. A credibility blend sitting between BF and
      chain ladder (weight ``emerged`` on chain ladder). Requires ``apriori_col``.
    - ``"cape_cod"`` -- Bornhuetter-Ferguson with the a priori derived from the data: a
      single expected loss ratio per segment, ``sum(paid) / sum(exposure * emerged)``,
      times each row's exposure. Requires ``exposure_col`` (an on-level premium /
      exposure per row). The loss ratio is mechanical; the exposure base is an input.

    The library applies a method; it does not pick the a priori or the exposure base.
    Supply either ``development_col`` or both ``date_col`` and ``valuation_date``; pass
    ``by`` with a per-segment factor table (and Cape Cod then derives one loss ratio per
    segment). Returns ``df`` with an ``out_col`` (default ``f"{value_col}_ultimate"``).
    """
    methods = {"chain_ladder", "bornhuetter_ferguson", "benktander", "cape_cod"}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}; got {method!r}.")
    if development_col is None and (date_col is None or valuation_date is None):
        raise ValueError(
            "Provide development_col, or both date_col and valuation_date, to determine each row's development period."
        )
    by_cols = as_list(by)
    needed = [value_col] + ([development_col] if development_col is not None else [date_col]) + by_cols
    if method in ("bornhuetter_ferguson", "benktander"):
        if apriori_col is None:
            raise ValueError(f"method={method!r} requires apriori_col (an expected ultimate per row).")
        needed.append(apriori_col)
    if method == "cape_cod":
        if exposure_col is None:
            raise ValueError("method='cape_cod' requires exposure_col (an on-level premium / exposure per row).")
        needed.append(exposure_col)
    validate_columns(df, needed)
    result = df.copy() if copy else df

    emerged = _emerged_factor(
        result, factors, date_col=date_col, valuation_date=valuation_date, development_col=development_col,
        by_cols=by_cols, factor_col=factor_col, development_name=development_name,
    )
    paid = result[value_col].to_numpy(dtype="float64")

    if method == "chain_ladder":
        ultimate = paid / emerged
    elif method == "bornhuetter_ferguson":
        apriori = result[apriori_col].to_numpy(dtype="float64")
        ultimate = paid + apriori * (1.0 - emerged)
    elif method == "benktander":
        apriori = result[apriori_col].to_numpy(dtype="float64")
        bf = paid + apriori * (1.0 - emerged)
        ultimate = paid + bf * (1.0 - emerged)
    else:  # cape_cod
        exposure = result[exposure_col].to_numpy(dtype="float64")
        elr = _cape_cod_elr(paid, exposure, emerged, result, by_cols)
        ultimate = paid + exposure * elr * (1.0 - emerged)

    result[out_col or f"{value_col}_ultimate"] = ultimate
    return result


class InsufficientDataWarning(UserWarning):
    """Emitted when a segment has too little data to fit and is skipped or aggregated.

    Filter it with the standard :mod:`warnings` machinery, e.g.
    ``warnings.filterwarnings("ignore", category=InsufficientDataWarning)``.
    """


def chain_ladder_by(
    df: pd.DataFrame,
    *,
    groupby: str | list[str],
    origin_col: str,
    valuation_col: str,
    amount_col: str,
    cumulative: bool = True,
    method: str = "volume",
    tail: float = 1.0,
    on_insufficient: str = "raise",
    warn: bool = True,
) -> dict[Any, ChainLadder]:
    """Fit a chain-ladder development pattern per segment of ``df``.

    Groups ``df`` by ``groupby``, builds a development triangle for each segment
    (see :func:`make_completion_triangle`), and fits a :class:`ChainLadder` to
    each. Returns ``{segment_key: ChainLadder}`` -- the key is a scalar for a
    single grouping column, or a tuple for several.

    Segments too small to fit (fewer than two origins or development periods, a zero cumulative,
    and so on) are handled by ``on_insufficient``:

    - ``"raise"`` (default): raise a ``ValueError`` naming the failing segment.
    - ``"skip"``: omit those segments from the result.
    - ``"aggregate"``: use the pooled pattern fit on the whole frame for them.

    When ``on_insufficient`` is ``"skip"`` or ``"aggregate"`` and ``warn`` is true,
    an :class:`InsufficientDataWarning` naming the affected segments is emitted;
    ``warn=False`` suppresses it (the standard :mod:`warnings` filters also apply).
    To ignore thin segments entirely, use ``on_insufficient="skip", warn=False``.
    """
    if on_insufficient not in ("raise", "skip", "aggregate"):
        raise ValueError("on_insufficient must be 'raise', 'skip', or 'aggregate'")
    group_cols = as_list(groupby)
    if not group_cols:
        raise ValueError("groupby must name at least one column")
    validate_columns(df, group_cols + [origin_col, valuation_col, amount_col])

    def _fit(frame: pd.DataFrame) -> ChainLadder:
        triangle = make_completion_triangle(
            frame,
            origin_col=origin_col,
            valuation_col=valuation_col,
            amount_col=amount_col,
            cumulative=cumulative,
        )
        return ChainLadder.fit(triangle, method=method, tail=tail)

    aggregate_pattern: ChainLadder | None = None
    if on_insufficient == "aggregate":
        try:
            aggregate_pattern = _fit(df)
        except ValueError as exc:
            raise ValueError(f"cannot fit the aggregate fallback pattern: {exc}") from exc

    by_key = group_cols if len(group_cols) > 1 else group_cols[0]
    patterns: dict[Any, ChainLadder] = {}
    insufficient: list[Any] = []
    for key, part in df.groupby(by_key, sort=True):
        try:
            patterns[key] = _fit(part)
        except ValueError as exc:
            if on_insufficient == "raise":
                raise ValueError(f"segment {key!r} cannot be fit: {exc}") from exc
            insufficient.append(key)
            if on_insufficient == "aggregate" and aggregate_pattern is not None:
                patterns[key] = aggregate_pattern

    if insufficient and warn:
        action = "using the aggregate pattern for" if on_insufficient == "aggregate" else "skipping"
        warnings.warn(
            f"{action} {len(insufficient)} segment(s) with insufficient data: {insufficient}",
            InsufficientDataWarning,
            stacklevel=2,
        )
    return patterns


def completion_factors_by(
    df: pd.DataFrame,
    *,
    groupby: str | list[str],
    origin_col: str,
    valuation_col: str,
    amount_col: str,
    cumulative: bool = True,
    method: str = "volume",
    tail: float = 1.0,
    on_insufficient: str = "raise",
    warn: bool = True,
    development_name: str = "development_month",
) -> pd.DataFrame:
    """Completion factors per segment as a tidy table.

    Convenience over :func:`chain_ladder_by`: one row per (segment, development period) with the
    completion factor, ready to review, pivot, or join. Columns are the grouping
    column(s), ``development_name``, and ``completion_factor``. ``on_insufficient`` and
    ``warn`` behave as in :func:`chain_ladder_by`.
    """
    group_cols = as_list(groupby)
    patterns = chain_ladder_by(
        df,
        groupby=groupby,
        origin_col=origin_col,
        valuation_col=valuation_col,
        amount_col=amount_col,
        cumulative=cumulative,
        method=method,
        tail=tail,
        on_insufficient=on_insufficient,
        warn=warn,
    )
    records: list[dict[str, Any]] = []
    for key, fitted in patterns.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_cols, key_tuple, strict=True))
        for development, factor in fitted.completion_factors.items():
            records.append({**key_map, development_name: development, "completion_factor": float(factor)})
    if not records:
        return pd.DataFrame(columns=group_cols + [development_name, "completion_factor"])
    return pd.DataFrame.from_records(records)
