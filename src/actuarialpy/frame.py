"""Stateful facade for experience-analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from actuarialpy.claimants import claim_concentration, summarize_claimants, top_claimants
from actuarialpy.cohorts import cohort_summary, duration_summary
from actuarialpy.columns import as_list, sum_columns, validate_columns
from actuarialpy.components import component_driver_analysis, summarize_components
from actuarialpy.expected import summarize_actual_vs_expected
from actuarialpy.experience import status_summary, summarize_experience, summarize_views
from actuarialpy.rolling import rolling_summary
from actuarialpy.trend import trend_summary

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
    """

    data: pd.DataFrame
    expense: str | list[str]
    revenue: str | list[str]
    exposure: str | list[str] | None = None
    date: str | None = None
    profile: str | None = None
    copy: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "expense", as_list(self.expense))
        object.__setattr__(self, "revenue", as_list(self.revenue))
        object.__setattr__(self, "exposure", as_list(self.exposure))
        if self.copy:
            object.__setattr__(self, "data", self.data.copy())

        required = list(self.expense) + list(self.revenue) + list(self.exposure)
        if self.date is not None:
            required.append(self.date)
        validate_columns(self.data, required)
        _validate_exposure_names(list(self.exposure))
        _validate_numeric_columns(self.data, list(self.expense), role="Expense")
        _validate_numeric_columns(self.data, list(self.revenue), role="Revenue")
        _validate_numeric_columns(self.data, list(self.exposure), role="Exposure")

    def with_roles(
        self,
        *,
        data: pd.DataFrame | None = None,
        expense: str | list[str] | None = None,
        revenue: str | list[str] | None = None,
        exposure: str | list[str] | None = None,
        date: str | None = None,
        profile: str | None = None,
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
            data = self.data.loc[mask]
        if copy:
            data = data.copy()
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

    def views(self, views: dict[str, str | list[str] | None], **kwargs: Any) -> dict[str, pd.DataFrame]:
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

    def _resolve_date_col(self, date_col: str | None) -> str:
        resolved = date_col or self.date
        if resolved is None:
            raise ValueError("A date column is required. Pass date=... to Experience or date_col=... to this method.")
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
