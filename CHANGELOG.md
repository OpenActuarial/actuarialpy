# Changelog

## 0.8.0

Ergonomics release: bind the experience column roles once instead of repeating
them on every call, plus a completion convenience and a trend fix.

### Added

- **`Experience` facade** (`actuarialpy.Experience`): an optional convenience
  layer that remembers the expense, revenue, exposure, and profile arguments so
  each view is a single short call. Methods delegate to the existing free
  functions (`by` -> `summarize_experience`, `views` -> `summarize_views`,
  `by_status` -> `status_summary`, `rolling` -> `rolling_summary`, `by_band` ->
  `summarize_by_band`, `trend` -> `trend_summary`, `cohort`/`duration` ->
  `cohort_summary`/`duration_summary`). The free functions are unchanged and
  remain the substrate; any bound role can be overridden per call.
- **`completed_experience`** in `completion`: completes claim components and then
  summarizes experience on the completed basis in one call. The `*_completed`
  columns become the expense numerator together with any `additional_expense_cols`
  (e.g. rebates, non-fee-for-service expense). Convenience for the common
  completion -> experience chain.

### Fixed

- **`trend_summary` no longer leaks the period column.** Previously the
  `period_col` (e.g. `year`) was summed through the group-by and collided into
  `year_x`/`year_y` in the output. The period column is now used only to build
  the prior/current masks and is no longer carried into the summary.


## 0.7.0

This release generalizes the package across lines of business (life, health, and
pension under the SOA; property/casualty under the CAS) and adds a lifecycle
layer and a set of experience-analysis modules. Health remains a first-class
application via the `health` profile.

### Changed

- **Default ratio is now `loss_ratio`** (general across lines), not the prior
  `expense_revenue_ratio` placeholder. `summarize_experience` and
  `rolling_summary` name the post-aggregation ratio column `loss_ratio` unless a
  profile or explicit name overrides it. The `health` profile still produces
  `mlr` and the `life` profile `benefit_ratio`, so existing health/life calls
  that pass `profile=...` are unaffected. Calls that relied on the old default
  column name should read `loss_ratio` (or pass `ratio_name=`/`ratio_col=`).

### Added

- **Lifecycle module** (`actuarialpy.lifecycle`): derive status and tenure from
  effective/termination dates rather than requiring a precomputed label, and
  clip exposure to the in-force window.
  - `derive_status` (active / first-year / termed, with `first_year_months` as a
    parameter and optional label remapping), `add_tenure`, `is_in_force`,
    `add_months_in_force`, and `earned_exposure` (prorated). The module derives
    the distinction and provides the windowing levers; it deliberately does not
    encode differential cohort treatment, which is a pricing-methodology choice.
- **Banding module** (`actuarialpy.banding`): `assign_band` and
  `summarize_by_band` for configurable size bands by any numeric column
  (subscriber/member count, exposure, premium, TIV).
- **Concentration module** (`actuarialpy.concentration`): `concentration_curve`,
  `concentration_summary`, and `top_n_share` — grain-agnostic, for group-level
  or member/claimant-level concentration.
- **Margins module** (`actuarialpy.margins`): `margin`, `margin_ratio`, and
  `add_margin` (premium net of losses and expense loadings, with optional ratio
  and per-exposure margin).
- **Large-loss / pooling module** (`actuarialpy.pooling`): `flag_large_losses`,
  `pool_losses` (cap at a pooling point → pooled/excess split), and
  `large_loss_summary`, plus `excess_over_threshold` — the deterministic
  excess-over-threshold hand-off that feeds tail/severity/aggregate modeling in
  the satellite packages (`extremeloss`, `lossmodels`, `risksim`).
- **`permissible_loss_ratio`** in `metrics` — the permissible / target /
  zero-margin loss ratio, `1 - expense_ratio - profit_provision` (the CAS PLR on
  a premium basis).


## 0.6.1

### Changed

- Repository migrated to a `src/` layout (`src/actuarialpy/`) to match the rest
  of the ecosystem. This is a repository-structure change only — the installed
  package, its import name, and its public API are unchanged. Development now
  requires an editable install (`pip install -e .`) for the test suite to import
  the package, which is the intended behavior of the `src` layout.
- README: the capabilities summary and package-structure tree now include the
  credibility module (the dedicated credibility section was added in 0.6.0), and
  credibility is removed from the list of planned additions.

## 0.6.0

### Added

- **Credibility module** (`actuarialpy.credibility`), exposed at the top level.
  - `Buhlmann` and `BuhlmannStraub` greatest-accuracy credibility models —
    **moved from the `lossmodels` package** (their behavior is unchanged; the
    test suite is ported verbatim to confirm parity). This relocation puts
    credibility next to the experience and ratemaking workflows that consume it,
    and lets `lossmodels` step back from being a complete Klugman implementation
    toward being a Klugman-anchored loss-distribution toolkit.
  - `credibility_weighted_estimate(observed, complement, z)` — the type-stable
    credibility-weighting primitive (scalar → float, Series → Series, array →
    ndarray), for blending an experience estimate with its complement at a `Z`
    from any source.

## 0.5.0

Corrections and consistency pass. Two items change output and are called out as
behavior changes; the rest are additive, fixes, or tooling.

### Behavior changes

- **Metric primitives now preserve pandas types.** `safe_divide` (and every
  metric that builds on it, e.g. `loss_ratio`, `pmpm`, `ratio`,
  `actual_to_expected`) returns a `pandas.Series` with the original index and
  name when given a Series, instead of a bare `numpy.ndarray`. Scalar inputs
  return a native `float`. Series are still combined positionally and are
  assumed aligned. Code that relied on receiving an `ndarray` from Series input
  should call `.to_numpy()` explicitly.
- **`rolling_summary` is now calendar-aware.** The window is measured in
  calendar periods rather than rows. Each group is reindexed onto a dense,
  gap-free period grid (filled with zero amounts/exposure) before rolling, so a
  missing month no longer stretches the window across extra calendar time, and
  `period_start`/`period_end` are correct in the presence of gaps. A new `freq`
  argument (pandas offset alias, e.g. `"MS"`) controls the grid; it is inferred
  when omitted, and gapped data requires an explicit `freq`.

### Fixes

- `combined_ratio` no longer coerces scalar inputs to `numpy.float64`; scalar in
  → `float` out, matching the other primitives. Python lists are still summed
  element-wise rather than concatenated.
- `make_completion_triangle` had a docstring promising a "cumulative" triangle
  while only summing within cells. The docstring now states the snapshot
  assumption, and a new `cumulative=True` argument accumulates incremental
  amounts across lag.
- `as_list` now returns set inputs in a deterministic (string-sorted) order
  instead of arbitrary iteration order, since these become group-by keys and
  column orders.

### Additions

- Most workflow functions are re-exported from the top level, so
  `from actuarialpy import summarize_experience, rolling_summary, trend_summary`
  works without knowing the submodule layout.
- `relative_change` is the single canonical `a / b - 1` implementation;
  `percent_change`, `variance_pct`, and `period_change` delegate to it.
- A shared `EXPOSURE_SUFFIX` map and `per_exposure_name` helper in `columns`
  replace the per-exposure naming logic previously duplicated across
  `experience`, `rolling`, and `components` (output names unchanged).

### Tooling

- Added a `py.typed` marker so the existing type hints are visible to
  downstream type checkers.
- Added `ruff` and `mypy` configuration and a GitHub Actions CI workflow that
  runs the test suite, the linter, and the type checker across Python
  3.9–3.12.
- README package structure synced with the actual modules (it had listed a
  nonexistent `validation.py` and omitted several real modules).
