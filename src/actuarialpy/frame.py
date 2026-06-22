"""The :class:`Experience` convenience facade.

``Experience`` binds the experience column roles (expense, revenue, exposure,
profile) once so each view is a single short call instead of repeating the
column arguments. It is an *optional* layer: every method delegates to the
existing free function, which remains available directly for anything the facade
does not cover. Any bound role can be overridden on a per-call basis.

    exp = Experience(df, expense=["claims", "rebates"], revenue="premium",
                     exposure="member_months", profile="health")
    exp.by("line_of_business")            # -> summarize_experience(...)
    exp.rolling(12, date_col="month", groupby="group_id")  # -> rolling_summary(...)
    exp.trend(amount_col="claims", period_col="year", prior_period=2025, current_period=2026)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from actuarialpy.banding import summarize_by_band
from actuarialpy.cohorts import cohort_summary, duration_summary
from actuarialpy.columns import as_list
from actuarialpy.experience import status_summary, summarize_experience, summarize_views
from actuarialpy.rolling import rolling_summary
from actuarialpy.trend import trend_summary

_UNSET: Any = object()


class Experience:
    """Bind expense/revenue/exposure/profile once and reuse across summaries."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        expense: str | Iterable[str],
        revenue: str | Iterable[str],
        exposure: str | Iterable[str] | None = None,
        profile: str | None = None,
    ) -> None:
        self.df = df
        self.expense = expense
        self.revenue = revenue
        self.exposure = exposure
        self.profile = profile

    # -- role resolution ----------------------------------------------------- #
    def _roles(self, expense: Any, revenue: Any, exposure: Any, profile: Any) -> dict[str, Any]:
        return {
            "expense_cols": self.expense if expense is None else expense,
            "revenue_cols": self.revenue if revenue is None else revenue,
            "exposure_cols": self.exposure if exposure is _UNSET else exposure,
            "profile": self.profile if profile is _UNSET else profile,
        }

    def _single_exposure(self, exposure: Any = _UNSET) -> str | None:
        value = self.exposure if exposure is _UNSET else exposure
        cols = as_list(value)
        return cols[0] if cols else None

    # -- delegating methods -------------------------------------------------- #
    def by(
        self,
        groupby: str | Iterable[str] | None = None,
        *,
        ratio_col: str | None = None,
        ratio_name: str | None = None,
        expense: Any = None,
        revenue: Any = None,
        exposure: Any = _UNSET,
        profile: Any = _UNSET,
    ) -> pd.DataFrame:
        """Grouped experience summary (delegates to ``summarize_experience``)."""
        return summarize_experience(
            self.df, groupby=groupby, ratio_col=ratio_col, ratio_name=ratio_name,
            **self._roles(expense, revenue, exposure, profile),
        )

    def views(
        self,
        views: dict[str, Any],
        *,
        expense: Any = None,
        revenue: Any = None,
        exposure: Any = _UNSET,
        profile: Any = _UNSET,
    ) -> dict[str, pd.DataFrame]:
        """Multiple grouped views in one call (delegates to ``summarize_views``)."""
        return summarize_views(self.df, views=views, **self._roles(expense, revenue, exposure, profile))

    def by_status(
        self,
        status_col: str,
        *,
        entity_col: str | None = None,
        expense: Any = None,
        revenue: Any = None,
        exposure: Any = _UNSET,
        profile: Any = _UNSET,
    ) -> pd.DataFrame:
        """Experience by status, with optional entity counts (``status_summary``)."""
        return status_summary(
            self.df, status_col=status_col, entity_col=entity_col,
            **self._roles(expense, revenue, exposure, profile),
        )

    def rolling(
        self,
        window: int = 12,
        *,
        date_col: str,
        groupby: str | Iterable[str] | None = None,
        freq: str | None = None,
        min_periods: int | None = None,
        drop_incomplete: bool = True,
        ratio_col: str = "loss_ratio",
        expense: Any = None,
        revenue: Any = None,
        exposure: Any = _UNSET,
    ) -> pd.DataFrame:
        """Calendar-aware rolling summary (delegates to ``rolling_summary``)."""
        roles = self._roles(expense, revenue, exposure, _UNSET)
        roles.pop("profile")  # rolling_summary takes no profile
        return rolling_summary(
            self.df, date_col=date_col, window=window, groupby=groupby, freq=freq,
            min_periods=min_periods, drop_incomplete=drop_incomplete, ratio_col=ratio_col, **roles,
        )

    def by_band(
        self,
        value_col: str,
        bands: Any,
        *,
        labels: Any = None,
        band_col: str = "band",
        ratio_col: str | None = None,
        right: bool = False,
        expense: Any = None,
        revenue: Any = None,
        exposure: Any = _UNSET,
        profile: Any = _UNSET,
    ) -> pd.DataFrame:
        """Experience by size band (delegates to ``summarize_by_band``)."""
        return summarize_by_band(
            self.df, value_col, bands, labels=labels, band_col=band_col,
            ratio_col=ratio_col, right=right, **self._roles(expense, revenue, exposure, profile),
        )

    def trend(
        self,
        *,
        amount_col: str,
        period_col: str | None = None,
        prior_period: Any = None,
        current_period: Any = None,
        groupby: str | Iterable[str] | None = None,
        prior_filter: Any = None,
        current_filter: Any = None,
        prior_label: str = "prior",
        current_label: str = "current",
        exposure: Any = _UNSET,
    ) -> pd.DataFrame:
        """Current-vs-prior trend (delegates to ``trend_summary``).

        ``amount_col`` is explicit because trend works on a single metric; the
        bound exposure is used for the per-exposure trend unless overridden.
        """
        return trend_summary(
            self.df, period_col=period_col, prior_period=prior_period, current_period=current_period,
            groupby=groupby, amount_col=amount_col, exposure_col=self._single_exposure(exposure),
            prior_filter=prior_filter, current_filter=current_filter,
            prior_label=prior_label, current_label=current_label,
        )

    def cohort(
        self,
        *,
        entity_col: str,
        date_col: str,
        start_date_col: str,
        duration_months: int = 12,
        groupby: str | Iterable[str] | None = None,
        expense: Any = None,
        revenue: Any = None,
        exposure: Any = _UNSET,
        profile: Any = _UNSET,
    ) -> pd.DataFrame:
        """First-N-months cohort summary (delegates to ``cohort_summary``)."""
        return cohort_summary(
            self.df, entity_col=entity_col, date_col=date_col, start_date_col=start_date_col,
            duration_months=duration_months, groupby=groupby,
            **self._roles(expense, revenue, exposure, profile),
        )

    def duration(
        self,
        *,
        entity_col: str,
        date_col: str,
        start_date_col: str,
        max_duration_month: int | None = None,
        expense: Any = None,
        revenue: Any = None,
        exposure: Any = _UNSET,
    ) -> pd.DataFrame:
        """Experience by duration month (delegates to ``duration_summary``)."""
        roles = self._roles(expense, revenue, exposure, _UNSET)
        roles.pop("profile")  # duration_summary takes no profile
        return duration_summary(
            self.df, entity_col=entity_col, date_col=date_col, start_date_col=start_date_col,
            max_duration_month=max_duration_month, **roles,
        )

    def __repr__(self) -> str:
        return (
            f"Experience(expense={self.expense!r}, revenue={self.revenue!r}, "
            f"exposure={self.exposure!r}, profile={self.profile!r}, rows={len(self.df)})"
        )
