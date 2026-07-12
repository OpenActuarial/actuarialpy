"""The canonical experience container: column roles, grain metadata, and
immutable transformations.

``Experience`` is the ecosystem's shared semantic wrapper for historical
actuarial data. It binds column roles (which columns are expense, revenue,
exposure, counts, dates, segmentation dimensions), grain metadata
(``exposure_keys``), and snapshot context (``valuation_date``, ``basis``)
once, so downstream packages -- experiencestudies, projectionmodels,
ratingmodels -- consume one object instead of re-declaring columns.

The class holds **no actuarial judgment**. Its public methods are immutable
*transformations*: each takes caller-supplied assumptions as arguments and
returns a new ``Experience`` (``complete``, ``adjust``, ``deseasonalize``,
``filter``, ``with_status``, ``with_roles``). Calculations and workflow
outputs -- summaries, trend fits, projections, rates -- belong to consuming
packages as functions that accept an ``Experience``. This split is enforced
by a test: no public method on this class may return anything other than
``Experience``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
from pandas.api.types import is_numeric_dtype

from actuarialpy.adjustments import adjust as _adjust
from actuarialpy.columns import as_list, sum_columns, validate_columns
from actuarialpy.lifecycle import derive_status
from actuarialpy.reserving import apply_completion as _apply_completion
from actuarialpy.seasonality import deseasonalize as _deseasonalize

__all__ = [
    "Experience",
    "Source",
    "Pivot",
    "single_role",
    "single_role_or_none",
    "resolve_date",
    "resolve_amount",
]

_ID_LIKE_EXPOSURE_NAMES = {
    "member_id",
    "subscriber_id",
    "group_id",
    "employee_id",
    "policy_id",
    "claim_id",
}

#: Basis value written by :meth:`Experience.complete`.
ULTIMATE = "ultimate"


def _as_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a role argument (str | iterable | None) to a tuple."""
    return tuple(as_list(value))


def _validate_exposure_names(exposures: Iterable[str]) -> None:
    bad = [
        col
        for col in exposures
        if col.lower() in _ID_LIKE_EXPOSURE_NAMES or col.lower().endswith("_id")
    ]
    if bad:
        raise ValueError(
            "Exposure columns must be numeric exposure measures, not identifiers. "
            f"Invalid exposure column(s): {bad}."
        )


def _validate_numeric_columns(df: pd.DataFrame, cols: Iterable[str], *, role: str) -> None:
    bad = [col for col in cols if not is_numeric_dtype(df[col])]
    if bad:
        raise ValueError(f"{role} columns must be numeric. Non-numeric column(s): {bad}.")


def single_role(roles: Iterable[str], role_name: str) -> str:
    """Return the single column bound to a role, or raise a helpful error.

    Consumers that need exactly one column for a role (one claims column, one
    exposure column) use this to turn the bound tuple into a column name.
    """
    cols = tuple(roles)
    if len(cols) == 1:
        return cols[0]
    if not cols:
        raise ValueError(
            f"No {role_name} column is bound. Bind {role_name}=... on the "
            f"Experience or pass the column explicitly to this function."
        )
    raise ValueError(
        f"Multiple {role_name} columns are bound ({list(cols)}). Pass the "
        f"column explicitly to this function."
    )


def single_role_or_none(roles: Iterable[str]) -> str | None:
    """Return the single bound column, ``None`` if unbound, or raise if several."""
    cols = tuple(roles)
    if not cols:
        return None
    if len(cols) > 1:
        raise ValueError(
            f"Multiple columns are bound for this role ({list(cols)}). "
            "Pass the column explicitly to this function."
        )
    return cols[0]


def resolve_date(exp: Experience, date_col: str | None = None) -> str:
    """Return an explicit date column or the bound ``date`` role."""
    resolved = date_col or exp.date
    if resolved is None:
        raise ValueError(
            "A date column is required. Bind date=... on the Experience or "
            "pass date_col=... explicitly."
        )
    return resolved


def resolve_amount(exp: Experience, amount_col: str | None = None) -> tuple[pd.DataFrame, str]:
    """Return ``(frame, column)`` for an amount: explicit, single expense, or summed.

    With one bound expense column the frame is returned as-is; with several,
    a temporary row-wise total is added so callers see one amount column.
    """
    if amount_col is not None:
        validate_columns(exp.data, [amount_col])
        return exp.data, amount_col
    expenses = list(exp.expense)
    if len(expenses) == 1:
        return exp.data, expenses[0]
    if not expenses:
        raise ValueError(
            "No expense column is bound. Bind expense=... on the Experience "
            "or pass the amount column explicitly."
        )
    temp = exp.data.copy()
    amount_name = "_actuarialpy_total_expense"
    temp[amount_name] = sum_columns(temp, expenses)
    return temp, amount_name


@dataclass(frozen=True, slots=True)
class Pivot:
    """Provenance of one ``wide_by`` pivot performed by :meth:`Experience.from_tables`.

    Records that categorical column ``by`` was pivoted so that measure column
    ``value`` became the wide ``columns`` under measure role ``role``. Stored
    on the resulting ``Experience`` so :meth:`Experience.melt` (and consumers
    such as ``projectionmodels.project``) can undo the reshape structurally --
    the inverse of a recorded pivot has exactly one right answer.
    """

    by: str
    role: str
    value: str
    columns: tuple[str, ...]


def _period_alias(freq: str) -> str:
    from pandas.tseries.frequencies import get_period_alias

    return get_period_alias(freq) or freq


_MEASURE_ROLES = ("expense", "revenue", "count")
_MAX_WIDE_CATEGORIES = 50


@dataclass(frozen=True)
class Source:
    """One measure table for :meth:`Experience.from_tables`.

    Declares which columns of ``data`` carry which measure role (the same
    role vocabulary as ``Experience``: ``expense``, ``revenue``, ``count``),
    plus how the table reaches the grain:

    ``wide_by``
        A categorical column (claim type, service line) to pivot: each
        category becomes its own column under the spec's measure role.
        Requires the spec to name exactly one measure column.
    ``date``
        This table's own date column (e.g. ``incurred_date``). It is floored
        to the constructor's ``period`` to produce the grain's date column --
        choosing *which* date (incurred vs paid) is the caller's judgment;
        flooring it is calendar arithmetic.
    ``agg``
        ``"sum"`` (default) or ``"count"`` (rows per grain cell, e.g. a claim
        count from claim IDs).
    ``keys``
        Maps this table's join-key column names onto the grain table's names
        (``keys={"mbr_id": "member_id"}``), for sources that spell the same
        key differently.
    ``name``
        Names this source as a listing member of an ``ExperienceSet``
        (``name="claims"`` -> ``book["claims"]``). Ignored by
        ``Experience.from_tables``.
    """

    data: pd.DataFrame
    expense: str | Iterable[str] = ()
    revenue: str | Iterable[str] = ()
    count: str | Iterable[str] = ()
    wide_by: str | None = None
    date: str | None = None
    agg: str = "sum"
    rename: Mapping[str, str] | None = None
    keys: Mapping[str, str] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        for role in _MEASURE_ROLES:
            object.__setattr__(self, role, _as_tuple(getattr(self, role)))
        named = [(role, col) for role in _MEASURE_ROLES for col in getattr(self, role)]
        if not named:
            raise ValueError(
                "A Source spec must name at least one measure column "
                "(expense=, revenue=, or count=)."
            )
        if self.agg not in ("sum", "count"):
            raise ValueError(f"agg must be 'sum' or 'count', got {self.agg!r}")
        if self.wide_by is not None and len(named) != 1:
            raise ValueError(
                "wide_by pivots exactly one measure column; this spec names "
                f"{[col for _, col in named]}. Split it into one spec per column."
            )
        required = [col for _, col in named]
        if self.wide_by is not None:
            required.append(self.wide_by)
        if self.date is not None:
            required.append(self.date)
        validate_columns(self.data, required)
        if self.keys is not None:
            validate_columns(self.data, list(self.keys))
        if self.rename is not None:
            if self.wide_by is not None:
                raise ValueError("rename does not apply to wide_by specs; the pivot names its own columns.")
            unknown = [c for c in self.rename if c not in [col for _, col in named]]
            if unknown:
                raise ValueError(f"rename references columns this spec does not name: {unknown}")

    def _named(self) -> tuple[str, str]:
        ((role, col),) = [
            (role, col) for role in _MEASURE_ROLES for col in getattr(self, role)
        ]
        return role, col


@dataclass(frozen=True, slots=True)
class Experience:
    """Bind an experience dataset to its actuarial column roles and grain.

    Role columns
    ------------
    ``expense``, ``revenue``, ``exposure``, ``count``
        Measure roles; each accepts one column name or several. At least one
        measure role (expense, revenue, or count) must be bound, but no
        particular one is mandatory -- a claims-only frame, a premium-only
        frame, and a count study are all legal.
    ``date``
        The experience date column (incurred month, policy month, ...).

    Grain metadata
    --------------
    ``dimensions``
        Segmentation columns -- group, product, claim type. Consumers use
        these as defaults for reporting groupings, assumption lookups, and
        projection grain. They say nothing about row grain.
    ``exposure_keys``
        Columns that identify one exposure unit, e.g.
        ``("member_id", "month")``. When bound, construction validates that
        the frame is unique on these columns, so repeated exposure units
        (long, service-line-grain data) are rejected instead of silently
        overcounting every per-exposure figure. Leave unbound to skip the
        guard; no grain safety is claimed without it.

    Snapshot context
    ----------------
    ``valuation_date``
        The paid-through / as-of date of the data. :meth:`complete` uses it
        as the default valuation date.
    ``basis``
        Per-column transformation state, e.g. ``{"paid_claims": "ultimate"}``.
        :meth:`complete` refuses columns already marked ``"ultimate"`` and
        marks the columns it completes, so accidental double development is
        an error rather than a silent overstatement. Any string is legal;
        only ``"ultimate"`` currently carries enforcement.

    The object is immutable and holds no actuarial judgment: every public
    method takes its assumptions as arguments (completion factors, adjustment
    factors, seasonal factors) and returns a new ``Experience``. Analytical
    consumers -- summaries, trend fits, projections, rates -- are functions in
    this package and downstream packages that accept an ``Experience``.

    The frame is not defensively copied on construction; transformations
    always return new objects over new frames.
    """

    data: pd.DataFrame
    expense: tuple[str, ...] = ()
    revenue: tuple[str, ...] = ()
    exposure: tuple[str, ...] = ()
    count: tuple[str, ...] = ()
    date: str | None = None
    dimensions: tuple[str, ...] = ()
    exposure_keys: tuple[str, ...] = ()
    valuation_date: pd.Timestamp | None = None
    basis: Mapping[str, str] = field(default_factory=dict)
    pivots: tuple[Pivot, ...] = ()

    def __post_init__(self) -> None:
        for role in ("expense", "revenue", "exposure", "count", "dimensions", "exposure_keys"):
            object.__setattr__(self, role, _as_tuple(getattr(self, role)))

        if not (self.expense or self.revenue or self.count):
            raise ValueError(
                "Bind at least one measure role (expense, revenue, or count)."
            )

        required = [
            *self.expense,
            *self.revenue,
            *self.exposure,
            *self.count,
            *self.dimensions,
            *self.exposure_keys,
        ]
        if self.date is not None:
            required.append(self.date)
        validate_columns(self.data, required)
        _validate_exposure_names(self.exposure)
        _validate_numeric_columns(self.data, self.expense, role="Expense")
        _validate_numeric_columns(self.data, self.revenue, role="Revenue")
        _validate_numeric_columns(self.data, self.exposure, role="Exposure")
        _validate_numeric_columns(self.data, self.count, role="Count")

        if self.valuation_date is not None:
            object.__setattr__(self, "valuation_date", pd.Timestamp(self.valuation_date))

        basis = dict(self.basis)
        unknown = [col for col in basis if col not in self.data.columns]
        if unknown:
            raise ValueError(f"basis references columns not in the frame: {unknown}")
        object.__setattr__(self, "basis", MappingProxyType(basis))

        object.__setattr__(self, "pivots", tuple(self.pivots))
        for pivot in self.pivots:
            missing_pivot = [c for c in pivot.columns if c not in self.data.columns]
            if missing_pivot:
                raise ValueError(
                    f"pivot {pivot.by!r} references columns not in the frame: {missing_pivot}"
                )

        if self.exposure_keys:
            duplicated = self.data.duplicated(list(self.exposure_keys))
            if bool(duplicated.any()):
                n = int(duplicated.sum())
                raise ValueError(
                    f"{n} row(s) repeat an exposure unit on exposure_keys="
                    f"{list(self.exposure_keys)}. Aggregate to the exposure "
                    "grain first, or bind keys that identify one row per unit."
                )

    # ------------------------------------------------------------------ #
    # Transformations: assumptions in, Experience out.                    #
    # ------------------------------------------------------------------ #

    def with_roles(
        self,
        *,
        data: pd.DataFrame | None = None,
        expense: str | Iterable[str] | None = None,
        revenue: str | Iterable[str] | None = None,
        exposure: str | Iterable[str] | None = None,
        count: str | Iterable[str] | None = None,
        date: str | None = None,
        dimensions: str | Iterable[str] | None = None,
        exposure_keys: str | Iterable[str] | None = None,
        valuation_date: Any | None = None,
        basis: Mapping[str, str] | None = None,
    ) -> Experience:
        """Return a new ``Experience`` with updated data, roles, or metadata."""
        return replace(
            self,
            data=self.data if data is None else data,
            expense=self.expense if expense is None else _as_tuple(expense),
            revenue=self.revenue if revenue is None else _as_tuple(revenue),
            exposure=self.exposure if exposure is None else _as_tuple(exposure),
            count=self.count if count is None else _as_tuple(count),
            date=self.date if date is None else date,
            dimensions=self.dimensions if dimensions is None else _as_tuple(dimensions),
            exposure_keys=(
                self.exposure_keys if exposure_keys is None else _as_tuple(exposure_keys)
            ),
            valuation_date=self.valuation_date if valuation_date is None else valuation_date,
            basis=dict(self.basis) if basis is None else dict(basis),
        )

    def filter(
        self,
        mask: Any | None = None,
        *,
        query: str | None = None,
        copy: bool = True,
    ) -> Experience:
        """Return a new ``Experience`` over a filtered dataset.

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
        return self.with_roles(data=data)

    def complete(
        self,
        factors: pd.Series,
        *,
        valuation_date: Any = None,
        columns: str | Iterable[str] | None = None,
        development_col: str | None = None,
        by: str | Iterable[str] | None = None,
        date_col: str | None = None,
    ) -> Experience:
        """Return a new ``Experience`` with paid amounts developed to ultimate.

        Grosses the expense (loss / claims) columns up to estimated ultimate in
        place under the same names -- ``completed = paid / completion_factor``.
        Each row's development period is ``development_months(date,
        valuation_date)`` (the convention :func:`make_completion_triangle`
        uses), or an explicit ``development_col``. ``valuation_date`` defaults
        to the object's bound valuation date. ``factors`` may be a flat Series
        (one pattern, from :func:`completion_factors`) or a tidy per-segment
        table from :func:`completion_factors_by`; with the latter, pass ``by``
        naming the grouping column(s). Only the numerator is developed --
        exposure is left untouched.

        Completing marks each developed column ``"ultimate"`` in :attr:`basis`,
        and completing a column already marked ``"ultimate"`` raises -- the
        double-development mistake is an error, not a silent overstatement.
        """
        cols = as_list(columns) if columns is not None else list(self.expense)
        if not cols:
            raise ValueError("No columns to complete; pass columns=... or bind an expense role.")
        already = [col for col in cols if self.basis.get(col) == ULTIMATE]
        if already:
            raise ValueError(
                f"Column(s) {already} are already on an ultimate basis; "
                "completing again would double-count development. Start from "
                "paid data, or pass columns=... naming undeveloped columns."
            )
        if valuation_date is None:
            valuation_date = self.valuation_date
        if development_col is None:
            resolved_date = resolve_date(self, date_col)
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
        new_basis = {**self.basis, **{col: ULTIMATE for col in cols}}
        return self.with_roles(data=data, basis=new_basis)

    def adjust(
        self,
        factors: float | int | pd.Series | pd.DataFrame,
        *,
        on: str | Iterable[str] | None = None,
        columns: str | Iterable[str] | None = None,
        by: str | Iterable[str] | None = None,
        how: str = "multiply",
        factor_col: str = "factor",
        audit_col: str | None = None,
        default: float | None = None,
    ) -> Experience:
        """Return a new ``Experience`` with an expense column restated by a factor.

        The general counterpart to :meth:`complete` and :meth:`deseasonalize`:
        joins a factor by the key ``on`` (a column already in the frame,
        optionally within ``by`` segments) and multiplies -- or, with
        ``how="divide"``, divides -- the selected column(s) in place under the
        same name. ``factors`` is a scalar, a Series indexed by ``on``, or a
        tidy DataFrame keyed by ``by + on``. This is the spine of
        experience-period restatement -- trend, benefit / area / demographic
        relativities, network discounts -- where the methodology is supplied as
        the factors rather than encoded here. Chain freely
        (``exp.complete(...).adjust(trend).adjust(area, on="region")``); with
        ``audit_col`` the cumulative restatement multiplier is carried across
        the chain. An absent key surfaces as ``NaN`` unless ``default`` is
        given (``default=1.0`` to mean "no adjustment for this key").
        """
        cols = as_list(columns) if columns is not None else list(self.expense)
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
        return self.with_roles(data=data)

    def deseasonalize(
        self,
        factors: pd.Series,
        *,
        columns: str | Iterable[str] | None = None,
        freq: str = "M",
        by: str | Iterable[str] | None = None,
        date_col: str | None = None,
    ) -> Experience:
        """Return a new ``Experience`` with the seasonal pattern divided out.

        Each selected column is divided by its row's seasonal factor (as
        produced by :func:`seasonality_factors`), in place under the same name.
        By default the expense columns are adjusted; pass ``columns`` to choose
        others. Only the numerator is touched. ``factors`` may be a flat Series
        (one pattern) or a tidy per-segment table from
        :func:`seasonality_factors_by`; with the latter pass ``by``. Estimate
        factors on the broader pool, not on this object's own (often thin)
        data. To put the pattern back, apply :func:`apply_seasonality` to
        ``.data``.
        """
        resolved_date = resolve_date(self, date_col)
        cols = as_list(columns) if columns is not None else list(self.expense)
        if not cols:
            raise ValueError(
                "No columns to deseasonalize; pass columns=... or bind an expense role."
            )
        validate_columns(self.data, cols + [resolved_date] + as_list(by))
        data = self.data.copy()
        for col in cols:
            data = _deseasonalize(
                data,
                factors,
                date_col=resolved_date,
                value_col=col,
                freq=freq,
                by=by,
                out_col=col,
                copy=False,
            )
        return self.with_roles(data=data)

    def with_status(
        self,
        *,
        effective_col: str,
        as_of: Any,
        termination_col: str | None = None,
        first_year_months: int = 12,
        status_col: str = "status",
        labels: dict[str, str] | None = None,
    ) -> Experience:
        """Return a new ``Experience`` with a derived lifecycle status column.

        Derives active / first-year / termed from effective and termination
        dates as of a reference date (see :func:`actuarialpy.derive_status`).
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
        return self.with_roles(data=data)

    def aggregate(self, by: str | Iterable[str] | None = None, *, freq: str | None = None) -> Experience:
        """Return a new ``Experience`` summed to a coarser grain.

        ``by`` names the grouping columns; ``freq`` (a pandas offset alias
        such as ``"MS"``, ``"QS"``, ``"YS"``) additionally floors the bound
        date role into the grouping. All measure-role columns are summed --
        aggregation is structural -- and non-measure, non-key columns are
        dropped, since they need not be constant at the coarser grain.

        Summing the exposure role is only provably safe when the input grain
        was validated, so an ``Experience`` with an exposure role must have
        ``exposure_keys`` bound. The result's ``exposure_keys`` are the new
        grain (uniqueness holds by construction of the groupby).
        """
        if by is None and freq is None:
            raise ValueError("Pass by=..., freq=..., or both -- nothing to aggregate to.")
        if self.exposure and not self.exposure_keys:
            raise ValueError(
                "aggregating would sum the exposure role over possibly repeated "
                "units; bind exposure_keys=... to prove one row per unit first, "
                "or drop the exposure role with with_roles(exposure=())."
            )
        by_cols = list(_as_tuple(by)) if by is not None else []
        validate_columns(self.data, by_cols)
        data = self.data
        keys = list(by_cols)
        new_date: str | None = None
        if freq is not None:
            date_col = resolve_date(self)
            if date_col in by_cols:
                raise ValueError(
                    f"the date role {date_col!r} is already in by=; pass either "
                    "freq= to floor it or include it in by=, not both."
                )
            data = data.assign(
                **{date_col: pd.to_datetime(data[date_col]).dt.to_period(_period_alias(freq)).dt.to_timestamp()}
            )
            keys.append(date_col)
            new_date = date_col
        elif self.date is not None and self.date in by_cols:
            new_date = self.date
        measures = list(
            dict.fromkeys([*self.expense, *self.revenue, *self.exposure, *self.count])
        )
        grouped = data.groupby(keys, dropna=False, as_index=False)[measures].sum()
        return replace(
            self,
            data=grouped,
            date=new_date,
            dimensions=tuple(d for d in self.dimensions if d in grouped.columns),
            exposure_keys=tuple(keys),
            basis={c: s for c, s in self.basis.items() if c in grouped.columns},
            pivots=tuple(
                p for p in self.pivots if all(c in grouped.columns for c in p.columns)
            ),
        )

    def melt(self, pivot: str | None = None) -> Experience:
        """Undo a recorded ``wide_by`` pivot, returning a long ``Experience``.

        The categorical column comes back (and joins ``dimensions``), the
        original measure column returns under its recorded role, and the wide
        columns disappear. Purely structural: only pivots recorded by
        :meth:`from_tables` can be melted, because only those have one right
        inverse.

        The melted frame repeats each exposure unit once per category, so
        ``exposure_keys`` is cleared -- summing the exposure role across
        categories on the result would overcount, and no grain safety is
        claimed. Per-category pipelines (projection base rates) consume it
        correctly.
        """
        recorded = {p.by: p for p in self.pivots}
        if not recorded:
            raise ValueError(
                "no recorded pivot to melt; this Experience was not built with "
                "a wide_by= Source spec."
            )
        if pivot is None:
            if len(recorded) > 1:
                raise ValueError(
                    f"multiple recorded pivots {sorted(recorded)}; pass pivot=... "
                    "to name the one to melt."
                )
            chosen = next(iter(recorded.values()))
        else:
            if pivot not in recorded:
                raise ValueError(
                    f"no recorded pivot {pivot!r}; recorded pivots: {sorted(recorded)}"
                )
            chosen = recorded[pivot]
        id_vars = [c for c in self.data.columns if c not in chosen.columns]
        long = self.data.melt(
            id_vars=id_vars,
            value_vars=list(chosen.columns),
            var_name=chosen.by,
            value_name=chosen.value,
        )
        role_cols = tuple(
            c for c in getattr(self, chosen.role) if c not in chosen.columns
        ) + (chosen.value,)
        states = {self.basis[c] for c in chosen.columns if c in self.basis}
        basis = {c: s for c, s in self.basis.items() if c not in chosen.columns}
        if len(states) == 1 and all(c in self.basis for c in chosen.columns):
            basis[chosen.value] = next(iter(states))
        dimensions = self.dimensions
        if chosen.by not in dimensions:
            dimensions = dimensions + (chosen.by,)
        return replace(
            self,
            data=long,
            **{chosen.role: role_cols},
            dimensions=dimensions,
            exposure_keys=(),
            basis=basis,
            pivots=tuple(p for p in self.pivots if p.by != chosen.by),
        )

    @classmethod
    def from_tables(
        cls,
        data: pd.DataFrame,
        *,
        grain: str | Iterable[str],
        exposure: str | Iterable[str] | None = None,
        sources: Iterable[Source] = (),
        date: str | None = None,
        period: str | None = None,
        dimensions: str | Iterable[str] = (),
        valuation_date: Any | None = None,
        basis: Mapping[str, str] | None = None,
        unmatched: str = "warn",
    ) -> Experience:
        """Build an ``Experience`` from source tables: multi-table at the
        doorway, single-table inside.

        ``data`` is the table that defines the grain -- one row per exposure
        unit (typically membership / eligibility). It is validated unique on
        ``grain``, contributes the ``exposure`` role, and keeps all its other
        columns (entity attributes ride along). Each :class:`Source` spec is
        then brought to the grain by one fixed, auditable algorithm:

        * tables at a *finer* grain are aggregated up (grouped and summed or
          counted) -- aggregation is structural, so the constructor may do it;
        * tables at a *coarser* grain (missing a grain column) are refused --
          allocation downward is judgment, so the caller must do it before
          binding;
        * grain cells with no rows get ``0.0`` (the absence of claims is zero
          claims), and rows whose keys don't exist in ``data`` are surfaced
          per ``unmatched`` (``"warn"`` or ``"raise"``) -- never dropped
          silently;
        * ``wide_by`` pivots are recorded as :class:`Pivot` provenance so they
          can be undone structurally by :meth:`melt`.

        The result is an ordinary single-grain ``Experience`` with
        ``exposure_keys`` set to ``grain`` (uniqueness was just proven).
        """
        if unmatched not in ("warn", "raise"):
            raise ValueError(f"unmatched must be 'warn' or 'raise', got {unmatched!r}")
        grain_cols = list(_as_tuple(grain))
        if not grain_cols:
            raise ValueError("grain must name at least one column.")
        exposure_cols = _as_tuple(exposure) if exposure is not None else ()
        required = [*grain_cols, *exposure_cols, *_as_tuple(dimensions)]
        if date is not None and date not in required:
            required.append(date)
        validate_columns(data, required)
        duplicated = data.duplicated(grain_cols)
        if bool(duplicated.any()):
            n = int(duplicated.sum())
            raise ValueError(
                f"{n} row(s) in the grain table repeat an exposure unit on "
                f"grain={grain_cols}. The grain table defines one row per unit; "
                "aggregate it first."
            )

        base = data.copy()
        base_keys = base[grain_cols].drop_duplicates()
        roles: dict[str, list[str]] = {role: [] for role in _MEASURE_ROLES}
        pivots: list[Pivot] = []
        joined_cols: set[str] = set(base.columns)

        for index, spec in enumerate(sources):
            if not isinstance(spec, Source):
                raise TypeError(
                    f"sources[{index}] is {type(spec).__name__}; wrap each measure "
                    "table in a Source(...) spec."
                )
            work = spec.data
            if spec.keys:
                # map this table's join-key names onto the grain's names --
                # renaming a merge key is structural, not judgment
                work = work.rename(columns=dict(spec.keys))
            if date is not None and date not in work.columns and spec.date is not None:
                if period is None:
                    raise ValueError(
                        f"sources[{index}] supplies its own date ({spec.date!r}); pass "
                        "period=... (e.g. 'M') so it can be floored to the grain date."
                    )
                work = work.assign(
                    **{date: pd.to_datetime(work[spec.date]).dt.to_period(_period_alias(period)).dt.to_timestamp()}
                )
            missing = [c for c in grain_cols if c not in work.columns]
            if missing:
                raise ValueError(
                    f"sources[{index}] lacks grain column(s) {missing}; amounts are "
                    "never allocated downward -- join it at its own grain or "
                    "allocate explicitly before binding."
                )

            orphan = work.merge(base_keys, on=grain_cols, how="left", indicator=True)
            n_orphan = int((orphan["_merge"] == "left_only").sum())
            if n_orphan:
                sample = (
                    orphan.loc[orphan["_merge"] == "left_only", grain_cols]
                    .drop_duplicates()
                    .head(3)
                    .to_dict("records")
                )
                message = (
                    f"sources[{index}]: {n_orphan} row(s) have grain keys not present "
                    f"in the grain table (e.g. {sample}); they will not join."
                )
                if unmatched == "raise":
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)

            if spec.wide_by is not None:
                role, value = spec._named()
                categories = work[spec.wide_by].nunique(dropna=False)
                if categories > _MAX_WIDE_CATEGORIES:
                    raise ValueError(
                        f"sources[{index}]: wide_by={spec.wide_by!r} has {categories} "
                        f"categories (limit {_MAX_WIDE_CATEGORIES}) -- that looks like "
                        "an identifier, not a category."
                    )
                wide = work.pivot_table(
                    index=grain_cols,
                    columns=spec.wide_by,
                    values=value,
                    aggfunc=spec.agg,
                    fill_value=0.0,
                )
                wide.columns = [str(c) for c in wide.columns]
                created = list(wide.columns)
                collision = [c for c in created if c in joined_cols]
                if collision:
                    raise ValueError(
                        f"sources[{index}]: pivoted column(s) {collision} collide with "
                        "existing columns; rename the categories or the source columns."
                    )
                base = base.merge(wide.reset_index(), on=grain_cols, how="left")
                base[created] = base[created].fillna(0.0)
                roles[role].extend(created)
                joined_cols.update(created)
                pivots.append(
                    Pivot(by=spec.wide_by, role=role, value=value, columns=tuple(created))
                )
                joined_cols.add(value)  # reserve: melt() recreates it
            else:
                spec_cols = [
                    (role, col) for role in _MEASURE_ROLES for col in getattr(spec, role)
                ]
                cols = [col for _, col in spec_cols]
                out = [(spec.rename or {}).get(col, col) for col in cols]
                collision = [c for c in out if c in joined_cols]
                if collision:
                    raise ValueError(
                        f"sources[{index}]: column(s) {collision} collide with existing "
                        "columns; rename them before binding."
                    )
                grouped = (
                    work.groupby(grain_cols, dropna=False, as_index=False)[cols]
                    .agg(spec.agg)
                )
                out_names = {col: (spec.rename or {}).get(col, col) for col in cols}
                grouped = grouped.rename(columns=out_names)
                base = base.merge(grouped, on=grain_cols, how="left")
                base[list(out_names.values())] = base[list(out_names.values())].fillna(0.0)
                for role, col in spec_cols:
                    roles[role].append(out_names[col])
                joined_cols.update(out_names.values())

        return cls(
            base,
            expense=tuple(roles["expense"]),
            revenue=tuple(roles["revenue"]),
            count=tuple(roles["count"]),
            exposure=exposure_cols,
            date=date,
            dimensions=dimensions,
            exposure_keys=tuple(grain_cols),
            valuation_date=valuation_date,
            basis=basis or {},
            pivots=tuple(pivots),
        )
