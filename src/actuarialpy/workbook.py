"""The workbook layer: one construction call, grain-honest members.

``ExperienceSet`` owns coordinated construction over source tables and
exposes materialized, inspectable ``Experience`` members -- the worksheet
(``tab``) plus one listing per *named* :class:`Source` spec, each at its
own declared grain. One construction call is universal; one instance never
pretends to hold two grains.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import pandas as pd

from actuarialpy.frame import Experience, Source, _as_tuple

__all__ = ["ExperienceSet"]


@dataclass(frozen=True)
class ExperienceSet:
    """A coordinated bundle of grain-honest ``Experience`` members.

    Build with :meth:`from_tables`. ``tab`` is the worksheet at the declared
    grain (via ``Experience.from_tables``); each *named* ``Source`` spec
    also yields a listing member at its own source grain, reachable by
    ``book["claims"]``. Members are ordinary, materialized ``Experience``
    objects: what a member's ``.data`` shows is exactly what a consumer
    receives.

    ``cohort(...)`` is the only cross-member operation: it filters the grain
    table on its own columns (the population authority) and re-derives every
    member from the filtered sources -- propagation by reconstruction, never
    by mutation. Worksheet-local transformations stay on the members and
    return plain ``Experience`` objects.
    """

    tab: Experience
    listings: Mapping[str, Experience]
    manifest: Mapping[str, Any]
    _sources: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "listings", MappingProxyType(dict(self.listings)))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    def __getitem__(self, name: str) -> Experience:
        try:
            return self.listings[name]
        except KeyError:
            raise KeyError(
                f"no listing named {name!r}; named listings: "
                f"{sorted(self.listings)}"
            ) from None

    @property
    def member_names(self) -> tuple[str, ...]:
        return ("tab", *sorted(self.listings))

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
        unmatched: str = "warn",
    ) -> ExperienceSet:
        """One construction call: the tab plus a listing per named spec.

        Takes exactly the arguments of ``Experience.from_tables`` -- the
        ``Source`` declarations already carry everything both members
        need (roles, the table's own date, the pivot categorical).
        """
        sources = tuple(sources)
        tab = Experience.from_tables(
            data, grain=grain, exposure=exposure, sources=sources, date=date,
            period=period, dimensions=dimensions,
            valuation_date=valuation_date, unmatched=unmatched,
        )
        grain_cols = list(_as_tuple(grain))
        listings: dict[str, Experience] = {}
        for spec in sources:
            if spec.name is None:
                continue
            if spec.name in listings or spec.name == "tab":
                raise ValueError(f"duplicate member name {spec.name!r}")
            listing_date = spec.date or (
                date if date is not None and date in spec.data.columns else None
            )
            dims = [
                c for c in (*_as_tuple(dimensions), spec.wide_by)
                if c is not None and c in spec.data.columns
            ]
            listings[spec.name] = Experience(
                spec.data,
                expense=spec.expense, revenue=spec.revenue, count=spec.count,
                date=listing_date,
                dimensions=tuple(dict.fromkeys(dims)),
                valuation_date=valuation_date,
            )
        manifest = {
            "grain": tuple(grain_cols),
            "tab": {"rows": len(tab.data), "roles": {
                "expense": tab.expense, "revenue": tab.revenue,
                "exposure": tab.exposure, "count": tab.count}},
            "sources": {
                name: {
                    "rows": len(exp.data),
                    "columns": tuple(exp.data.columns),
                    **({"date_range": (str(exp.data[exp.date].min()),
                                       str(exp.data[exp.date].max()))}
                       if exp.date else {}),
                }
                for name, exp in listings.items()
            },
        }
        sources = {
            "data": data, "grain": grain, "exposure": exposure,
            "sources": sources, "date": date, "period": period,
            "dimensions": dimensions, "valuation_date": valuation_date,
            "unmatched": unmatched,
        }
        return cls(tab=tab, listings=listings, manifest=manifest,
                   _sources=sources)

    def cohort(self, **predicates: Any) -> ExperienceSet:
        """A new ``ExperienceSet`` restricted to a population.

        Predicates name columns of the grain table (the population
        authority) with a value or list of values. Every member is
        re-derived: the grain table is filtered directly, and each source
        table is semi-joined to the surviving grain keys on the grain
        columns it shares -- propagation by reconstruction, so nothing can
        go stale.
        """
        src = dict(self._sources)
        data: pd.DataFrame = src["data"]
        unknown = [c for c in predicates if c not in data.columns]
        if unknown:
            raise ValueError(
                f"cohort predicates must name grain-table columns; unknown: "
                f"{unknown}. The grain table has: {list(data.columns)}"
            )
        mask = pd.Series(True, index=data.index)
        for col, value in predicates.items():
            values = value if isinstance(value, (list, tuple, set)) else [value]
            mask &= data[col].isin(list(values))
        filtered = data[mask].copy()
        grain_cols = list(_as_tuple(src["grain"]))
        new_sources = []
        for spec in src["sources"]:
            shared = [c for c in grain_cols if c in spec.data.columns]
            if shared:
                keys = filtered[shared].drop_duplicates()
                sub = spec.data.merge(keys, on=shared, how="inner")
            else:
                sub = spec.data
            new_sources.append(replace(spec, data=sub))
        src.update(data=filtered, sources=tuple(new_sources))
        return type(self).from_tables(
            src.pop("data"), grain=src.pop("grain"), **src
        )

    def reconcile(self) -> pd.DataFrame:
        """Tie each named listing's measure totals to the tab.

        Returns one row per (listing, measure): source total, tab total,
        difference, and whether they tie. A nonzero difference is the
        surfaced exclusions (orphan keys that never joined) -- the check an
        actuary does by hand between the claims extract and the worksheet.
        """
        rows = []
        for name, exp in self.listings.items():
            spec = next(s for s in self._sources["sources"] if s.name == name)
            for role in ("expense", "revenue", "count"):
                for col in getattr(spec, role):
                    source_total = float(pd.to_numeric(
                        exp.data[col], errors="coerce").sum())
                    if spec.wide_by is not None:
                        pivot = next(p for p in self.tab.pivots
                                     if p.value == col)
                        tab_total = float(
                            self.tab.data[list(pivot.columns)].to_numpy().sum())
                    else:
                        out = (spec.rename or {}).get(col, col)
                        tab_total = (float(self.tab.data[out].sum())
                                     if spec.agg == "sum"
                                     else float(self.tab.data[out].sum()))
                    rows.append({
                        "listing": name, "measure": col, "role": role,
                        "source_total": source_total, "tab_total": tab_total,
                        "difference": source_total - tab_total,
                        "ties": abs(source_total - tab_total) < 1e-6,
                    })
        return pd.DataFrame(rows)
