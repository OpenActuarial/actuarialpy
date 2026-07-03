"""Stateful facade for experience-analysis workflows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, cast

import pandas as pd
from pandas.api.types import is_numeric_dtype

from actuarialpy.banding import summarize_by_band
from actuarialpy.claimants import claim_concentration, summarize_claimants, top_claimants
from actuarialpy.cohorts import cohort_summary, duration_summary
from actuarialpy.columns import as_list, sum_columns, validate_columns
from actuarialpy.components import component_driver_analysis, summarize_components
from actuarialpy.credibility import credibility_weighted_estimate
from actuarialpy.decomposition import decompose_per_exposure_trend, frequency_severity_summary
from actuarialpy.expected import summarize_actual_vs_expected
from actuarialpy.experience import status_summary, summarize_experience, summarize_views
from actuarialpy.lifecycle import derive_status
from actuarialpy.metrics import per_exposure, safe_divide
from actuarialpy.pooling import pool_losses
from actuarialpy.adjustments import adjust as _adjust
from actuarialpy.reserving import apply_completion as _apply_completion
from actuarialpy.rolling import rolling_summary
from actuarialpy.seasonality import deseasonalize as _deseasonalize
from actuarialpy.trend import TrendFit, _comparison_masks, fit_trend as _fit_trend, trend_summary

_ID_LIKE_EXPOSURE_NAMES = {"member_id", "subscriber_id", "group_id", "employee_id", "policy_id", "claim_id"}


def _validate_exposure_names(exposures: list[str]) -> None:
    bad = [col for col in exposures if col.lower() in _ID_LIKE_EXPOSURE_NAMES or col.lower().endswith("_id")]
    if bad:
        raise ValueError(
            "Exposure columns must be numeric exposure measures, not identifiers. "
            f"Invalid exposure column(s): {bad}."
        )


def _validate_numeric_columns(df: pd.DataFrame, cols: list[str], *, role: str) -> None:
    bad = [col for col in cols if not is_numeric_dtype(df[col])]
    if bad:
        raise ValueError(f"{role} columns must be numeric. Non-numeric column(s): {bad}.")


@dataclass(frozen=True)
class Experience:
    """Bind an experience dataset to its actuarial column roles.

    ``Experience`` is the recommended entry point for repeated experience-analysis
    workflows. It stores common column roles once and delegates calculations to
    the package's free functions. The object is immutable: methods return
    DataFrames or new ``Experience`` objects rather than changing stored data in
    place.

    Bind ``count`` (a claim or service count) to unlock the frequency-severity views:
    :meth:`frequency_severity` and :meth:`decompose_trend` (frequency x severity,
    optionally x mix). :meth:`fit_trend` regresses a developed trend on the bound history.

    **Grain matters.** ``Experience`` aggregates by *summing* the bound columns, so it
    expects rows at the grain of the exposure unit -- one row per unit (a member-month in
    a health book, a policy-month in life), with the exposure column = 1 (or the eligible
    fraction). If your data is *long* (one row per service line or transaction, so the same
    exposure unit repeats across several rows), summing the exposure column overcounts it,
    and every per-exposure figure -- the per-exposure rate, frequency, the loss-ratio
    denominator -- is wrong by the number of rows per unit. ``Experience`` does not detect
    this: it has no record key, so it cannot tell a long frame from a wide one. For long or
    multi-table warehouse data, either aggregate to the exposure grain first, or use
    :meth:`bind`, which sources exposure from a correctly-grained table (e.g. a health
    book's eligibility) via :class:`~actuarialpy.Count` and never sums a repeated column.
    """

    data: pd.DataFrame
    expense: str | list[str]
    revenue: str | list[str]
    exposure: str | list[str] | None = None
    date: str | None = None
    profile: str | None = None
    count: str | None = None
    copy: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "expense", as_list(self.expense))
        object.__setattr__(self, "revenue", as_list(self.revenue))
        object.__setattr__(self, "exposure", as_list(self.exposure))
        if self.copy:
            object.__setattr__(self, "data", self.data.copy())

        required = as_list(self.expense) + as_list(self.revenue) + as_list(self.exposure)
        if self.date is not None:
            required.append(self.date)
        if self.count is not None:
            required.append(self.count)
        validate_columns(self.data, required)
        _validate_exposure_names(as_list(self.exposure))
        _validate_numeric_columns(self.data, as_list(self.expense), role="Expense")
        _validate_numeric_columns(self.data, as_list(self.revenue), role="Revenue")
        _validate_numeric_columns(self.data, as_list(self.exposure), role="Exposure")
        if self.count is not None:
            _validate_numeric_columns(self.data, [self.count], role="Count")

    def with_roles(
        self,
        *,
        data: pd.DataFrame | None = None,
        expense: str | list[str] | None = None,
        revenue: str | list[str] | None = None,
        exposure: str | list[str] | None = None,
        date: str | None = None,
        profile: str | None = None,
        count: str | None = None,
        copy: bool | None = None,
    ) -> "Experience":
        """Return a new ``Experience`` object with updated data or roles."""
        return replace(
            self,
            data=self.data if data is None else data,
            expense=self.expense if expense is None else expense,
            revenue=self.revenue if revenue is None else revenue,
            exposure=self.exposure if exposure is None else exposure,
            date=self.date if date is None else date,
            profile=self.profile if profile is None else profile,
            count=self.count if count is None else count,
            copy=self.copy if copy is None else copy,
        )

    def filter(
        self,
        mask: Any | None = None,
        *,
        query: str | None = None,
        copy: bool = True,
    ) -> "Experience":
        """Return a new ``Experience`` object over a filtered dataset.

        Use either a boolean mask or a pandas query string.
        """
        if (mask is None) == (query is None):
            raise ValueError("Pass exactly one of mask or query.")
        if query is not None:
            data = self.data.query(query)
        else:
            data = cast("pd.DataFrame", self.data.loc[mask])
        if copy:
            data = data.copy()
        return self.with_roles(data=data, copy=False)

    def deseasonalize(
        self,
        factors: pd.Series,
        *,
        columns: str | list[str] | None = None,
        freq: str = "M",
        by: str | list[str] | None = None,
        date_col: str | None = None,
    ) -> "Experience":
        """Return a new ``Experience`` with the seasonal pattern divided out.

        Each selected column is divided by its row's seasonal factor (as produced by
        :func:`seasonality_factors`), in place under the same name, so every
        downstream view -- :meth:`trend`, :meth:`rolling`, :meth:`by`, and the rest --
        then operates on the deseasonalized series. By default the expense
        (loss / claims) columns are adjusted; pass ``columns`` to choose others. Only
        the numerator is touched: exposure is left alone, so a deseasonalized
        per-exposure rate is simply deseasonalized claims over unchanged exposure.

        ``factors`` may be a flat Series (one pattern) or a tidy per-segment table from
        :func:`seasonality_factors_by`; with the latter, pass ``by`` naming the grouping
        column(s) to join on group plus season. Estimate factors on the broader pool,
        not on this object's own (often thin) data. To put the pattern back, apply
        :func:`apply_seasonality` to ``.data``.
        """
        resolved_date = self._resolve_date_col(date_col)
        cols = as_list(columns) if columns is not None else as_list(self.expense)
        if not cols:
            raise ValueError("No columns to deseasonalize; pass columns=... or bind an expense role.")
        validate_columns(self.data, cols + [resolved_date] + as_list(by))
        data = self.data.copy()
        for col in cols:
            data = _deseasonalize(
                data, factors, date_col=resolved_date, value_col=col, freq=freq, by=by, out_col=col, copy=False
            )
        return self.with_roles(data=data, copy=False)

    def complete(
        self,
        factors: pd.Series,
        *,
        valuation_date: Any = None,
        columns: str | list[str] | None = None,
        development_col: str | None = None,
        by: str | list[str] | None = None,
        date_col: str | None = None,
    ) -> "Experience":
        """Return a new ``Experience`` with paid amounts developed to ultimate.

        Grosses the expense (loss / claims) columns up to estimated ultimate in place
        under the same names -- ``completed = paid / completion_factor`` -- so downstream
        views (:meth:`trend`, :meth:`rolling`, :meth:`by`, ...) then run on the completed
        series. Each row's development period is
        ``development_months(date, valuation_date)`` (the convention
        :func:`make_completion_triangle` uses), or an explicit ``development_col``. The
        join is by value, so the frame's index is irrelevant; rows past the triangle's
        last development period are taken as fully complete, and only recent, immature
        months actually move.

        ``factors`` may be a flat Series (one pattern, from :func:`completion_factors`)
        or a tidy per-segment table from :func:`completion_factors_by`; with the latter,
        pass ``by`` naming the grouping column(s) to join on group plus development
        period. Only the numerator is developed -- exposure is left untouched. This
        applies to the latest-diagonal shape (one row per incurred month, ``claims``
        paid-to-date as of ``valuation_date``); a frame already on an ultimate basis must
        not be completed again.
        """
        cols = as_list(columns) if columns is not None else as_list(self.expense)
        if not cols:
            raise ValueError("No columns to complete; pass columns=... or bind an expense role.")
        if development_col is None:
            resolved_date = self._resolve_date_col(date_col)
            validate_columns(self.data, cols + [resolved_date] + as_list(by))
        else:
            resolved_date = None
            validate_columns(self.data, cols + [development_col] + as_list(by))
        data = self.data.copy()
        for col in cols:
            data = _apply_completion(
                data,
                factors,
                value_col=col,
                date_col=resolved_date,
                valuation_date=valuation_date,
                development_col=development_col,
                by=by,
                out_col=col,
                copy=False,
            )
        return self.with_roles(data=data, copy=False)

    def adjust(
        self,
        factors: float | int | pd.Series | pd.DataFrame,
        *,
        on: str | list[str] | None = None,
        columns: str | list[str] | None = None,
        by: str | list[str] | None = None,
        how: str = "multiply",
        factor_col: str = "factor",
        audit_col: str | None = None,
        default: float | None = None,
    ) -> "Experience":
        """Return a new ``Experience`` with an expense column restated by a factor.

        The general counterpart to :meth:`complete` and :meth:`deseasonalize`: joins a
        factor by the key ``on`` (a column already in the frame, optionally within ``by``
        segments) and multiplies -- or, with ``how="divide"``, divides -- the selected
        column(s) in place under the same name, so every downstream view composes on the
        restated series. ``factors`` is a scalar (one factor for all rows), a Series
        indexed by ``on``, or a tidy DataFrame keyed by ``by + on``.

        This is the spine of experience-period restatement -- trend, benefit / area /
        demographic relativities, network discounts -- where the methodology is supplied
        as the factors rather than encoded here. Chain freely
        (``exp.complete(...).adjust(trend).adjust(area, on="region")``); with ``audit_col``
        the cumulative restatement multiplier is carried across the chain, one value per
        row, for a reviewable audit trail. An absent key surfaces as ``NaN`` unless
        ``default`` is given (``default=1.0`` to mean "no adjustment for this key").
        """
        cols = as_list(columns) if columns is not None else as_list(self.expense)
        if not cols:
            raise ValueError("No columns to adjust; pass columns=... or bind an expense role.")
        validate_columns(self.data, cols + as_list(on) + as_list(by))
        data = self.data.copy()
        for col in cols:
            data = _adjust(
                data,
                factors,
                value_col=col,
                on=on,
                by=by,
                how=how,
                factor_col=factor_col,
                out_col=col,
                audit_col=audit_col,
                default=default,
                copy=False,
            )
        return self.with_roles(data=data, copy=False)

    def by(self, groupby: str | list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        """Summarize experience by optional grouping columns."""
        return summarize_experience(
            self.data,
            groupby=groupby,
            expense_cols=kwargs.pop("expense_cols", kwargs.pop("expense", self.expense)),
            revenue_cols=kwargs.pop("revenue_cols", kwargs.pop("revenue", self.revenue)),
            exposure_cols=kwargs.pop("exposure_cols", kwargs.pop("exposure", self.exposure)),
            profile=kwargs.pop("profile", self.profile),
            **kwargs,
        )

    def views(self, views: dict[str, str | Iterable[str] | None], **kwargs: Any) -> dict[str, pd.DataFrame]:
        """Create several named grouped experience views."""
        return summarize_views(
            self.data,
            views=views,
            expense_cols=kwargs.pop("expense_cols", kwargs.pop("expense", self.expense)),
            revenue_cols=kwargs.pop("revenue_cols", kwargs.pop("revenue", self.revenue)),
            exposure_cols=kwargs.pop("exposure_cols", kwargs.pop("exposure", self.exposure)),
            profile=kwargs.pop("profile", self.profile),
            **kwargs,
        )

    def rolling(
        self,
        window: int = 12,
        *,
        groupby: str | list[str] | None = None,
        date_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Create a rolling-period experience summary."""
        resolved_date = self._resolve_date_col(date_col)
        return rolling_summary(
            self.data,
            date_col=resolved_date,
            window=window,
            groupby=groupby,
            expense_cols=kwargs.pop("expense_cols", kwargs.pop("expense", self.expense)),
            revenue_cols=kwargs.pop("revenue_cols", kwargs.pop("revenue", self.revenue)),
            exposure_cols=kwargs.pop("exposure_cols", kwargs.pop("exposure", self.exposure)),
            **kwargs,
        )

    def trend(
        self,
        *,
        amount_col: str | None = None,
        exposure_col: str | None = None,
        groupby: str | list[str] | None = None,
        date_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compare amount or per-exposure experience between two periods."""
        data, resolved_amount = self._data_with_amount(amount_col)
        # Use the bound date column only for date-range comparisons. If the
        # caller supplies period_col/prior_period/current_period, passing the
        # bound date column would create two comparison modes and incorrectly
        # raise an error.
        resolved_date = date_col if date_col is not None else self.date
        if "period_col" in kwargs and date_col is None:
            resolved_date = None
        return trend_summary(
            data,
            amount_col=resolved_amount,
            exposure_col=exposure_col or self._single_exposure_or_none(),
            groupby=groupby,
            date_col=resolved_date,
            **kwargs,
        )

    def frequency_severity(
        self,
        *,
        count_col: str | None = None,
        loss_col: str | None = None,
        exposure_col: str | None = None,
        groupby: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """Per-group claim frequency, severity, and per-exposure loss (see ``frequency_severity_summary``).

        Uses the bound ``count``, ``expense`` (as the loss), and ``exposure`` roles, so the
        columns are specified once on the object. The identity ``loss_per_exposure == frequency *
        severity`` holds for every row.
        """
        data, resolved_loss = self._data_with_amount(loss_col)
        return frequency_severity_summary(
            data,
            count_col=self._resolve_count(count_col),
            loss_col=resolved_loss,
            exposure_col=self._resolve_exposure(exposure_col),
            groupby=groupby,
        )

    def decompose_trend(
        self,
        *,
        count_col: str | None = None,
        loss_col: str | None = None,
        exposure_col: str | None = None,
        mix_by: str | Iterable[str] | None = None,
        groupby: str | list[str] | None = None,
        period_col: str | None = None,
        prior_period: Any = None,
        current_period: Any = None,
        date_col: str | None = None,
        prior_start: Any = None,
        prior_end: Any = None,
        current_start: Any = None,
        current_end: Any = None,
        prior_filter: Any = None,
        current_filter: Any = None,
    ) -> pd.DataFrame:
        """Decompose the per-exposure loss trend between two periods of the bound data.

        Splits the bound frame into prior and current with the same comparison modes as
        :meth:`trend` -- ``period_col`` with ``prior_period`` / ``current_period``, a
        ``date_col`` with prior/current ranges (the bound ``date`` is used when no
        ``date_col`` is passed), or explicit ``prior_filter`` / ``current_filter`` masks --
        then decomposes the change via :func:`decompose_per_exposure_trend`, using the bound
        ``count``, ``expense`` (as the loss), and ``exposure`` roles. Pass ``mix_by`` to add
        the third LMDI mix term; ``groupby`` reports one decomposition per group.
        """
        resolved_count = self._resolve_count(count_col)
        resolved_exposure = self._resolve_exposure(exposure_col)
        data, resolved_loss = self._data_with_amount(loss_col)
        date_mode = any(v is not None for v in (date_col, prior_start, prior_end, current_start, current_end))
        resolved_date = (date_col if date_col is not None else self.date) if date_mode else None
        prior_mask, current_mask, _ = _comparison_masks(
            data,
            period_col=period_col,
            prior_period=prior_period,
            current_period=current_period,
            date_col=resolved_date,
            prior_start=prior_start,
            prior_end=prior_end,
            current_start=current_start,
            current_end=current_end,
            prior_filter=prior_filter,
            current_filter=current_filter,
        )
        return decompose_per_exposure_trend(
            data.loc[prior_mask],
            data.loc[current_mask],
            count_col=resolved_count,
            loss_col=resolved_loss,
            exposure_col=resolved_exposure,
            on=groupby,
            mix_by=mix_by,
        )

    def fit_trend(
        self,
        *,
        value_col: str | None = None,
        exposure_col: str | None = None,
        date_col: str | None = None,
        freq: str = "M",
        min_periods: int = 3,
        confidence: float = 0.95,
    ) -> TrendFit:
        """Fit an exponential trend to the bound experience by log-linear regression.

        Defaults to the bound ``expense`` (claims) over the bound ``exposure`` -- the
        per-exposure trend -- across the bound ``date``; pass ``value_col`` / ``exposure_col`` to
        override, or leave the exposure unbound to trend the raw amount. Returns a
        ``TrendFit`` (see :func:`fit_trend`). Run on completed, deseasonalized history.
        """
        data, resolved_value = self._data_with_amount(value_col)
        resolved_exposure = exposure_col if exposure_col is not None else self._single_exposure_or_none()
        return _fit_trend(
            data,
            value_col=resolved_value,
            date_col=self._resolve_date_col(date_col),
            exposure_col=resolved_exposure,
            freq=freq,
            min_periods=min_periods,
            confidence=confidence,
        )

    def components(
        self,
        component_cols: str | list[str],
        *,
        exposure_col: str | None = None,
        groupby: str | list[str] | None = None,
        date_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Explain component drivers between two periods."""
        # Use the bound date column only for date-range comparisons. If the
        # caller supplies period_col/prior_period/current_period, passing the
        # bound date column would create two comparison modes and incorrectly
        # raise an error.
        resolved_date = date_col if date_col is not None else self.date
        if "period_col" in kwargs and date_col is None:
            resolved_date = None
        return component_driver_analysis(
            self.data,
            component_cols=component_cols,
            exposure_col=exposure_col or self._single_exposure_or_none(),
            groupby=groupby,
            date_col=resolved_date,
            **kwargs,
        )

    def component_summary(
        self,
        component_cols: str | list[str],
        *,
        groupby: str | list[str] | None = None,
        exposure_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Summarize component amounts, per-exposure values, and shares."""
        return summarize_components(
            self.data,
            groupby=groupby,
            component_cols=component_cols,
            exposure_col=exposure_col or self._single_exposure_or_none(),
            **kwargs,
        )

    def actual_vs_expected(
        self,
        expected: str | list[str],
        *,
        actual: str | list[str] | None = None,
        groupby: str | list[str] | None = None,
        exposure: str | list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Summarize actual-versus-expected experience.

        If ``actual`` is omitted, the object's bound expense columns are used.
        """
        return summarize_actual_vs_expected(
            self.data,
            groupby=groupby,
            actual_cols=self.expense if actual is None else actual,
            expected_cols=expected,
            exposure_cols=self.exposure if exposure is None else exposure,
            **kwargs,
        )

    def claimants(
        self,
        claimant_col: str,
        *,
        amount_cols: str | list[str] | None = None,
        groupby: str | list[str] | None = None,
        exposure_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Aggregate the experience to claimant/member/risk level."""
        return summarize_claimants(
            self.data,
            claimant_col=claimant_col,
            amount_cols=self.expense if amount_cols is None else amount_cols,
            groupby=groupby,
            exposure_col=exposure_col,
            **kwargs,
        )

    def top_claimants(
        self,
        claimant_col: str,
        *,
        amount_cols: str | list[str] | None = None,
        amount_col: str | None = None,
        groupby: str | list[str] | None = None,
        n: int = 25,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return top claimants by amount."""
        return top_claimants(
            self.data,
            claimant_col=claimant_col,
            amount_cols=self.expense if amount_cols is None and amount_col is None else amount_cols,
            amount_col=amount_col,
            groupby=groupby,
            n=n,
            **kwargs,
        )

    def claimant_concentration(
        self,
        claimant_col: str,
        *,
        amount_cols: str | list[str] | None = None,
        groupby: str | list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Summarize how concentrated experience is among top claimants."""
        claimant_summary = summarize_claimants(
            self.data,
            claimant_col=claimant_col,
            amount_cols=self.expense if amount_cols is None else amount_cols,
            groupby=groupby,
        )
        return claim_concentration(claimant_summary, groupby=groupby, **kwargs)

    def cohort(
        self,
        *,
        entity_col: str,
        start_date_col: str,
        duration_months: int = 12,
        groupby: str | list[str] | None = None,
        date_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Summarize each entity's first N months or cohort-duration window."""
        return cohort_summary(
            self.data,
            entity_col=entity_col,
            date_col=self._resolve_date_col(date_col),
            start_date_col=start_date_col,
            duration_months=duration_months,
            groupby=groupby,
            expense_cols=kwargs.pop("expense_cols", kwargs.pop("expense", self.expense)),
            revenue_cols=kwargs.pop("revenue_cols", kwargs.pop("revenue", self.revenue)),
            exposure_cols=kwargs.pop("exposure_cols", kwargs.pop("exposure", self.exposure)),
            profile=kwargs.pop("profile", self.profile),
            **kwargs,
        )

    def duration(
        self,
        *,
        entity_col: str,
        start_date_col: str,
        max_duration_month: int | None = None,
        date_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Summarize experience by duration month since entity start."""
        return duration_summary(
            self.data,
            entity_col=entity_col,
            date_col=self._resolve_date_col(date_col),
            start_date_col=start_date_col,
            expense_cols=kwargs.pop("expense_cols", kwargs.pop("expense", self.expense)),
            revenue_cols=kwargs.pop("revenue_cols", kwargs.pop("revenue", self.revenue)),
            exposure_cols=kwargs.pop("exposure_cols", kwargs.pop("exposure", self.exposure)),
            max_duration_month=max_duration_month,
            **kwargs,
        )

    def by_status(self, status_col: str, *, entity_col: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Summarize experience by a status column."""
        return status_summary(
            self.data,
            status_col=status_col,
            entity_col=entity_col,
            expense_cols=kwargs.pop("expense_cols", kwargs.pop("expense", self.expense)),
            revenue_cols=kwargs.pop("revenue_cols", kwargs.pop("revenue", self.revenue)),
            exposure_cols=kwargs.pop("exposure_cols", kwargs.pop("exposure", self.exposure)),
            profile=kwargs.pop("profile", self.profile),
            **kwargs,
        )

    def with_status(
        self,
        *,
        effective_col: str,
        as_of: Any,
        termination_col: str | None = None,
        first_year_months: int = 12,
        status_col: str = "status",
        labels: dict[str, str] | None = None,
    ) -> "Experience":
        """Return a new ``Experience`` with a derived lifecycle status column.

        Derives active / first-year / termed from effective and termination dates
        as of a reference date (see :func:`actuarialpy.derive_status`). Summarize
        the result with :meth:`by_status`.
        """
        data = derive_status(
            self.data,
            effective_col=effective_col,
            as_of=as_of,
            termination_col=termination_col,
            first_year_months=first_year_months,
            status_col=status_col,
            labels=labels,
        )
        return self.with_roles(data=data, copy=False)

    def by_band(
        self,
        value_col: str,
        bands: Any,
        *,
        labels: Any = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Summarize experience by a size band on ``value_col`` (see ``summarize_by_band``)."""
        return summarize_by_band(
            self.data,
            value_col,
            bands,
            labels=labels,
            expense_cols=kwargs.pop("expense_cols", kwargs.pop("expense", self.expense)),
            revenue_cols=kwargs.pop("revenue_cols", kwargs.pop("revenue", self.revenue)),
            exposure_cols=kwargs.pop("exposure_cols", kwargs.pop("exposure", self.exposure)),
            profile=kwargs.pop("profile", self.profile),
            **kwargs,
        )

    def margin(
        self,
        groupby: str | list[str] | None = None,
        *,
        margin_col: str = "margin",
        ratio_col: str = "margin_ratio",
        per_exposure_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Underwriting margin (revenue net of expense) by optional grouping.

        Aggregates the bound expense and revenue roles with :meth:`by`, then adds
        the margin (``total_revenue - total_expense``), the margin ratio, and an
        optional per-exposure margin.
        """
        summary = self.by(groupby, **kwargs)
        summary[margin_col] = summary["total_revenue"] - summary["total_expense"]
        summary[ratio_col] = safe_divide(summary[margin_col], summary["total_revenue"])
        if per_exposure_col is not None:
            exposure = self._single_exposure_or_none()
            if exposure is None:
                raise ValueError("A single bound exposure is required for per_exposure_col.")
            summary[per_exposure_col] = per_exposure(summary[margin_col], summary[exposure])
        return summary

    def credibility_weighted(
        self,
        groupby: str | list[str],
        *,
        z: Any,
        metric: str = "loss_ratio",
        complement: float | None = None,
        out_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Blend each group's ``metric`` with a complement at credibility ``z``.

        Computes the grouped summary (:meth:`by`), then blends ``metric`` toward
        ``complement`` using ``z`` (see
        :func:`actuarialpy.credibility_weighted_estimate`). ``z`` may be a scalar
        or values aligned to the grouped rows. When ``complement`` is omitted the
        book-level value of ``metric`` is used as the complement of credibility.
        """
        summary = self.by(groupby, **kwargs)
        if metric not in summary.columns:
            raise ValueError(f"metric '{metric}' is not in the summary columns: {list(summary.columns)}")
        if complement is None:
            complement = self.by(**kwargs)[metric].iloc[0]
        name = out_col or f"credibility_weighted_{metric}"
        summary[name] = credibility_weighted_estimate(summary[metric], complement, z)
        return summary

    def pool_claimants(
        self,
        claimant_col: str,
        pooling_point: float,
        *,
        amount_cols: str | list[str] | None = None,
        groupby: str | list[str] | None = None,
        amount_name: str = "total_expense",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Aggregate to claimant level and split each claimant into pooled/excess.

        Summarizes the experience to claimant grain (:meth:`claimants`) and caps
        each claimant's total at ``pooling_point`` (see
        :func:`actuarialpy.pool_losses`), returning pooled and excess columns for
        capped experience and the excess hand-off to tail modeling.
        """
        claimant_totals = summarize_claimants(
            self.data,
            claimant_col=claimant_col,
            amount_cols=self.expense if amount_cols is None else amount_cols,
            groupby=groupby,
            amount_name=amount_name,
        )
        return pool_losses(claimant_totals, amount_name, pooling_point, **kwargs)

    def _resolve_date_col(self, date_col: str | None) -> str:
        resolved = date_col or self.date
        if resolved is None:
            raise ValueError("A date column is required. Pass date=... to Experience or date_col=... to this method.")
        return resolved

    def _resolve_count(self, count_col: str | None) -> str:
        resolved = count_col or self.count
        if resolved is None:
            raise ValueError(
                "A count column is required. Pass count=... to Experience or count_col=... to this method."
            )
        validate_columns(self.data, [resolved])
        return resolved

    def _resolve_exposure(self, exposure_col: str | None) -> str:
        if exposure_col is not None:
            validate_columns(self.data, [exposure_col])
            return exposure_col
        resolved = self._single_exposure_or_none()
        if resolved is None:
            raise ValueError(
                "An exposure column is required for this method. Pass exposure=... to Experience "
                "or exposure_col=... to this method."
            )
        return resolved

    def _single_exposure_or_none(self) -> str | None:
        exposures = as_list(self.exposure)
        if not exposures:
            return None
        if len(exposures) > 1:
            raise ValueError("Multiple exposures are bound. Pass exposure_col explicitly for this method.")
        return exposures[0]

    def _data_with_amount(self, amount_col: str | None) -> tuple[pd.DataFrame, str]:
        if amount_col is not None:
            validate_columns(self.data, [amount_col])
            return self.data, amount_col
        expenses = as_list(self.expense)
        if len(expenses) == 1:
            return self.data, expenses[0]
        temp = self.data.copy()
        amount_name = "_actuarialpy_total_expense"
        temp[amount_name] = sum_columns(temp, expenses)
        return temp, amount_name
