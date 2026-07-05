# Changelog

## 0.40.1

### Added

- openpyxl added as a dev dependency.

## 0.40.0

### Added
- **Mack standard errors on the chain ladder.** `ChainLadder.mack_sigma_squared`
  estimates the development-variance parameters (with Mack's log-linear
  extrapolation for the final period, where only one link ratio exists), and
  `ChainLadder.mack_standard_errors` returns per-origin and total reserve
  standard errors by the distribution-free formulas of Mack (1993) --
  parameter and process error per origin, plus the cross-origin covariance
  of shared estimation error in the total. Verified against the published
  Taylor-Ashe results (total reserve 18,680,856; total s.e. 2,447,095) and
  an independent transcription of the recursion. The total's
  cross-origin term is computed pairwise over the development periods
  unobserved by both origins of each pair, so it is independent of the
  triangle's row order (property-tested under row reversal; the familiar
  textbook loop assumes rows sorted most-developed first). Defined for the
  volume-weighted estimator only (Mack's model *is* that estimator;
  `method="simple"` refuses rather than reporting errors the assumptions
  do not support), and a fitted `tail` is treated as deterministic --
  ultimates and standard errors scale by it exactly, and this is stated in
  the docstring rather than hidden.

## 0.39.1

### Changed

- Documentation-only vocabulary sweep completing the 0.37--0.38 domain-agnostic
  rename. Core docstrings no longer use unattributed health vocabulary
  ("PMPM", "membership", "eligibility", "medical") as the default illustration
  of per-exposure quantities; domain terms remain only where explicitly
  attributed as a dialect (a health shop's `mlr`, member-months from
  eligibility) or inside the `labels` / `profiles` opt-in layer, which is
  unchanged. README, examples README, example walkthrough framings, and stale
  test comments updated to match (`decompose_pmpm_trend` references corrected
  to `decompose_per_exposure_trend`; the removed `pmpm` primitive corrected to
  `per_exposure`). No code paths changed.

## 0.39.0

### Added

- Example scripts are now executed by the test suite. Three examples that
  had silently broken at the 0.37 rename (`experience_basics`,
  `rolling_trend_monitor`, `trend_decomposition`) are updated to the
  current frequency/severity per-exposure API.

### Changed

- More descriptive package `description` metadata.

## 0.38.0

### Changed

- Decomposition components renamed to match the `frequency_severity_summary`
  columns: `util_trend` / `cost_trend` / `util_effect` / `cost_effect` are
  now `frequency_trend` / `severity_trend` / `frequency_effect` /
  `severity_effect` (`mix_*` unchanged). Identities hold as before:
  `loss_per_exposure_trend == frequency_trend * severity_trend (* mix_trend)`
  and `loss_per_exposure_change == frequency_effect + severity_effect
  (+ mix_effect)`. The utilization / unit-cost dialect is a labels-level
  rename like any other domain vocabulary.

## 0.37.0

### Removed

- The domain-sugar metrics leave the core: `pmpm`, `pspm`, `pepm` (use
  `per_exposure`), `medical_loss_ratio` (use `loss_ratio`; the `health`
  profile labels it `mlr` on output views), and `utilization_per_1000`
  (assumed month-based exposure and per-1,000 scaling; compute from
  `frequency` where needed). `frequency_severity_summary` drops the
  `util_per_1000` column and the `annualization` parameter with it.
- The automatic `_pmpm` / `_pspm` / `_pepm` per-exposure suffixes keyed off
  column names are removed everywhere (experience, rolling, claimants,
  components, actual-vs-expected, underwriting). Every per-exposure output
  is now the mechanical `{name}_per_{exposure_col}`, produced by one shared
  `per_exposure_name` helper in `columns`; domain names are opt-in via
  `labels` / explicit output-name parameters, never inferred from column
  names. Per-exposure columns now also follow custom total names
  (`claims_total` -> `claims_total_per_member_months`).

### Changed

- `decompose_pmpm_trend` -> `decompose_per_exposure_trend`; the summary
  column `pmpm` and the keys `pmpm_prior` / `pmpm_current` / `pmpm_trend` /
  `pmpm_change` become `loss_per_exposure` / `loss_per_exposure_*`. The
  identities are unchanged: `loss_per_exposure == frequency * severity` and
  `loss_per_exposure_trend == util_trend * cost_trend (* mix_trend)`.

## 0.36.0

### Changed

- The underwriting module is renamed to domain-agnostic vocabulary before
  its first release (0.35.0 was built but never published). `benefit` ->
  `losses`, `admin` -> `expenses`, `member_months` -> `exposure`, `mcr` ->
  `loss_ratio`, `aer` -> `expense_ratio`, `from_pmpm` ->
  `from_per_exposure`, and every `*_pmpm` output becomes
  `*_per_exposure`. A `combined_ratio` is added, mirroring the
  loss/expense/combined trio in `metrics`. Domain naming now flows through
  the profile system instead of the calculation: `to_frame`, `statement`,
  and `underwriting_summary` accept `profile`/`labels`, renaming only the
  loss-ratio column to the domain's ratio name (`health` -> `mlr`, `life`
  -> `benefit_ratio`), exactly like `summarize_experience`. The grouped
  version keeps the month-based per-exposure suffixes (`_pmpm` / `_pspm` /
  `_pepm`) purely as sugar when the exposure column is literally named
  `member_months` / `subscriber_months` / `employee_months`, and uses
  `_per_{exposure_col}` otherwise -- the same convention as the experience
  summaries.
- `weighted` docstrings drop group-insurance phrasing; behavior unchanged.

## 0.35.0

### Added

- `UnderwritingSummary` and `underwriting_summary` in a new `underwriting`
  module: the two-tier underwriting income statement. Labeled revenue,
  benefit, and admin components go in; total revenue, total benefit,
  gross margin (the benefit-tier margin, admin excluded), gain/(loss)
  (gross margin less admin), MCR, AER, and the margin ratios come out,
  with PMPM variants when member months are supplied. Ratio denominators
  are explicit parameters (`"total_revenue"` or `"premium"`) rather than
  assumptions, because real exhibits mix them -- MCR over net revenue,
  AER over gross premium -- and `reconciliation()` reports the resulting
  gap in `gain% = 1 - MCR - AER` so convention drift is visible. The
  grouped frame version follows the `summarize_experience` contract:
  components are summed first and every ratio is a ratio of sums.
  Component labels are the caller's vocabulary; the library only sums
  them. These are management/pricing metrics -- the ACA rebate MLR is a
  separate regulated calculation and is documented as out of scope.
- `weighted_mean` and `weighted_summary` in a new `weighted` module, for
  quantities that are already rates at the row level (rate actions,
  persistency, trend assumptions) and therefore cannot be summed. The
  weight is a required argument everywhere -- an unweighted mean of rate
  actions silently equal-weights a 50-life group with a 5,000-life group
  -- and the weight total is reported alongside every grouped average.
  NaN values propagate by default (`skipna=False`) so missing data
  surfaces instead of silently shrinking the base.

## 0.34.0

### Changed
- `safe_divide` and `combined_ratio` preserve pandas inputs: Series in,
  Series out (index and name retained), instead of coercing to bare numpy
  arrays.
- Grouped operations (claimants, decomposition, reserving, seasonality,
  exposure, credibility internals) zip keys and columns strictly, surfacing
  length mismatches instead of silently truncating.

## 0.31.0

### Added

- `retained_cv` and `retention_for_target_cv` in the pooling module. `retained_cv`
  gives the coefficient of variation of the retained (capped) aggregate of `n`
  independent units; `retention_for_target_cv` inverts it to find the retention
  level at which that CV meets a target, which is the basis for a size-graded
  retention rule (larger `n` tolerates a higher retention). Both are general
  retention-stability primitives over a per-unit outcome sample -- no domain
  vocabulary -- and use a sorted prefix-sum so the capped-CV curve is computed in
  one pass.

## 0.30.0

Simplified the library back to a single contract: **one tidy table in, views out.**
`Experience` takes a frame at the grain you're analysing, you name its columns, and you ask
for views — `by`, `rolling`, `trend`, completion, seasonality, and the rest — alongside the
free functions for one-off math.

### Removed

- The keyed measure facts engine (`Fact`, `Count`, and `Experience.bind`) introduced in
  0.29.0. It let you bind several warehouse tables (claims / eligibility / premium) at
  different grains and enforced a cut-by-keys law so member-months couldn't be summed off
  claim rows. In practice it added a second way to use the library and a layer of grain
  reasoning (canonical keys, the cut-by-keys law, the broadcast rule) for a data-plumbing
  job that pandas already does well. Building the analysis frame — aggregating claims to a
  grain, counting member-months from eligibility, joining premium — is left to the caller,
  where it belongs. The single-frame `Experience` and every free function are unchanged.

## 0.29.0

Adds the **keyed measure fact** engine: a multi-table companion to the single-frame
`Experience` (which is unchanged and remains the primary entry point for one-frame data),
for warehouse-shaped data where claims, eligibility, and premium live in separate tables at
different grains. You declare each table as a `Fact` (its measures,
counts, optional exposure, and the keys it is carried by) and bind them into one object;
the engine aggregates each fact only *up* to grains its keys support, joins facts on
shared keys, and computes rates and decompositions from the existing verified primitives.

The single law governing every operation: **a measure can be cut only by keys its fact
carries.** The one principled exception is exposure -- the declared population denominator
-- which is broadcast across reporting axes it does not carry (so per-line utilization
divides each line's count by *total* membership, never an apportioned share). Every other
measure must carry the requested grain's keys or the cut is refused: a member-level loss
ratio is unavailable when premium is only group-grain, and the engine raises rather than
fabricate a per-member premium. This is purely additive; the single-frame `Experience`
and all free functions are unchanged.

### Added

- `Fact` -- one keyed table. Declares `measures` (summed), `counts` (severity
  denominators / utilization numerators), optional `exposure` (`Count(col)` to count a
  column, e.g. member-months from an eligibility table, or a measure name), `keys` (a list
  or a `{canonical: physical}` map so a claims table's `incurred_month` and an eligibility
  table's `month` both fill the `month` role and join), `category` (a within-fact
  reporting axis, folded into keys), and `labels` (human-readable names that travel with a
  key, lossless via functional dependence).
- `Count` -- marks exposure as the count of a column over a cut (never summed off claim
  rows), the fix for the member-months overcount footgun.
- `BoundExperience` -- the engine returned by `Experience.bind`. Methods:
  - `rate(numerator, denominator="exposure", by=, scale=)` -- the single rate primitive.
    PMPM is `rate(measure)`; utilization is `rate(count, scale=...)`; severity is
    `rate(measure, count)`; loss ratio is `rate(paid, premium)`. Convenience wrappers
    `pmpm`, `utilization`, `severity`, `loss_ratio` call it. `numerator` and `denominator`
    each accept a **list** of measure names, summed -- the multi-table analogue of binding
    several expense columns. The pieces may live in different facts at different grains
    (e.g. FFS `paid` from claims plus a member-month `care_mgmt` fee plus a group-month
    `rebate`); each is aggregated to the grain and added, and the cut-by-keys law applies to
    every measure. Non-FFS expenses are modeled as their own facts at their own grain, not
    as fake service lines in claims.
  - `trend(numerator, by=, prior=, current=, denominator=, period=)` -- period-over-period
    change in a per-exposure (or any) rate; `period` defaults to the bound `time` key.
  - `decompose(measure, count, by=, mix_by=, prior=, current=)` -- utilization x unit cost
    (x mix) decomposition of the PMPM change, delegating the LMDI math to
    `decompose_pmpm_trend`. `mix_by` must be keys of the exposure fact (the partition
    test): mix measures composition shift across a partition of the population, so it is
    only defined on dimensions that partition exposure.
  - `flag_entities(entity, measure, threshold, return_detail=)` -- decides *who* crosses a
    threshold on the aggregate, then returns those entities' original rows intact (every
    date, group, and category preserved). The large-claimant pattern that must not collapse
    detail.
  - `concentration(measure, entity, by=, ...)` -> `claim_concentration`; `pool(measure,
    entity, pooling_point, by=)` -> `pool_losses`; `summary(by=)` for totals and PMPM.
- `Experience.bind(facts, time=None)` -- classmethod entry point returning a
  `BoundExperience`. `Fact`, `Count`, and `BoundExperience` are exported at the top level.

### Examples

- New `keyed_facts.py` -- a runnable end-to-end demo on three synthetic warehouse tables
  (claims / eligibility / premium): binding via `Experience.bind`, `pmpm`, per-line
  `utilization` (showing exposure broadcast and the absence of member-month overcount),
  `severity`, group `loss_ratio` and the refused member-level one, `trend`, two- and
  three-way `decompose` (reconciling to machine precision), and `flag_entities` with claim
  detail preserved.
- New `binding_example.py` -- a `build_experience(claims, eligibility, premium,
  premium_has_member_id=...)` template to copy and point at your own tables, spotlighting
  the column mapping and the one group- vs member-grain premium decision.
- New `sample_warehouse()` generator in `_sample_data.py` returning the three keyed tables
  at their natural grains; `examples/README.md` updated.

### Tests

- New `test_facts.py` (31 tests): `Fact` and `bind` validation; every core rate against
  hand-computed values; the cut-by-keys law on both numerator and denominator; exposure
  broadcast and the member-months no-overcount guarantee; trend; decompose reconciliation
  (`util_trend * cost_trend * mix_trend == pmpm_trend`), factor recovery, and the partition
  test; `flag_entities` detail preservation, summary mode, and empty result; concentration,
  pool, and summary. Full suite: 288 passing.

## 0.28.0

Wires the frequency-severity family into the `Experience` facade so columns are bound
once, closing a consistency gap: `frequency_severity_summary`, `decompose_pmpm_trend`,
and `fit_trend` were free functions with no facade method because `Experience` had no
count role. `Experience` now binds an optional `count`, and three methods delegate to
those functions using the bound roles. The free functions are unchanged.

### Added

- `count` role on `Experience` (optional, a single claim/service count column), validated
  numeric like the other roles and preserved across `filter` / `with_roles`.
- `Experience.frequency_severity(...)` -> `frequency_severity_summary` using the bound
  count, expense (as the loss), and exposure.
- `Experience.decompose_trend(...)` -> `decompose_pmpm_trend`. Splits the bound frame into
  prior/current with the same comparison modes as `Experience.trend` (period_col +
  prior/current_period, a date_col with ranges, or explicit filters; the bound `date` is
  used when no date_col is passed), then decomposes utilization x unit cost, with `mix_by`
  for the third LMDI mix term and `groupby` for per-group rows.
- `Experience.fit_trend(...)` -> `fit_trend`, defaulting to the bound expense over the
  bound exposure (PMPM trend) across the bound date; returns a `TrendFit`.

### Changed

- `decompose_pmpm_trend` now raises a clear `ValueError` when `on` and `mix_by` share a
  column (mix is undefined when the mix dimension is also a reporting group), instead of a
  cryptic pandas grouper error.

### Examples

- `trend_decomposition.py` gains an `Experience.decompose_trend` section showing it
  reproduces the free-function split with columns bound once; `sample_trend_cells()` gains
  a `premium` column so the facade (which binds a revenue role) is clean.

### Tests

- New `test_experience_count_methods.py` (11 tests): count-role validation, each facade
  method equals its free function (period mode, bound-date ranges, mix_by, groupby), the
  unbound-count and overlapping-on/mix_by errors, and role persistence through `filter` /
  `with_roles`. Full suite: 257 passing.

## 0.27.0

Extends `decompose_pmpm_trend` with an optional `mix_by`, adding a third **mix**
component to the PMPM split: utilization × unit cost × mix. The two-way split is an
exact identity, so book-wide utilization and unit-cost trends silently absorb shifts in
membership composition; `mix_by` measures utilization and unit cost *within* each cell
and reports the composition shift as its own term. Omitting `mix_by` is unchanged --
the existing two-way behaviour is byte-for-byte preserved. The library measures the
decomposition; it does not decide which cells define the mix.

### Added

- `mix_by` parameter on `decompose_pmpm_trend` (a column or list of columns). When
  given, PMPM is decomposed three ways via LMDI (logarithmic mean Divisia index), which
  is order-free and leaves no residual: `pmpm_trend == util_trend * cost_trend *
  mix_trend` and `pmpm_change == util_effect + cost_effect + mix_effect`, both exact.
  Output gains `mix_trend` and `mix_effect`; utilization and unit cost are then measured
  within cell rather than book-wide.
- A list in `mix_by` defines the cells as their cross -- one blended mix term, not a
  per-dimension attribution (for that, run the decomposition once per dimension). `on`
  and `mix_by` are orthogonal: `on` groups the output rows, `mix_by` defines the mix
  cells within each group.
- Every mix cell must have positive count, loss, and exposure in both periods (the
  within-cell multiplicative split is undefined otherwise); a clear `ValueError` names
  the offending cells, with guidance to combine sparse cells or filter entrants/exits.

### Examples

- New `trend_decomposition.py`: the two-way split, the three-way `mix_by=` split with
  exact multiplicative and dollar reconciliation, mix over a single dimension vs the
  cross of two (showing the cross is not the sum of the marginals), and `on=` + `mix_by=`
  together. Backed by a new deterministic `sample_trend_cells()` generator (segment ×
  region cells with uniform within-cell trend and an enrollment shift toward the High
  segment).

### Tests

- 12 new tests in `test_decomposition.py`: exact multiplicative and additive
  reconciliation (including on 25 random books), the worked-example values, two-way
  preserved when `mix_by` is omitted, pure-mix isolation, single-cell mix → 1.0 matching
  the two-way factors, the list/cross behaviour and cross ≠ sum of marginals, `on=` +
  `mix_by=` composition, and the positive-cells guard. Full suite: 246 passing.

## 0.26.0

Adds `fit_trend` -- developing a trend from history by log-linear regression, the
counterpart to the existing two-point and application trend tools. Returns the fitted
annual trend with goodness of fit and a confidence interval; stays dependency-free
(NumPy only, including the Student-t critical value). The library fits the trend; it does
not select it.

### Added

- `fit_trend(df, *, value_col, date_col, exposure_col=None, freq="M", min_periods=3,
  confidence=0.95)` in `trend`: aggregates to the period grain, forms the rate
  (`value / exposure` when `exposure_col` is given, else `value`), and fits
  `log(rate) = intercept + slope * t` (t in years from the first period) by OLS. The
  fitted annual trend is `exp(slope) - 1`. Time is taken from real dates, so an occasional
  missing period is handled correctly; non-positive rates raise.
- `TrendFit` result (frozen dataclass): `annual_trend`, `r_squared`, `std_error`
  (delta-method SE of the annual trend), `ci` / `ci_low` / `ci_high` (confidence interval,
  asymmetric -- transformed from the log-scale slope interval), `confidence`, `n_periods`,
  `slope`, `intercept`, and a `.factor(months)` bridge to application
  (`(1 + annual_trend) ** (months / 12)`).
- Dependency-free Student-t and inverse-normal quantiles (Acklam + Abramowitz-Stegun
  26.7.5) backing the confidence interval -- verified against tabulated critical values
  (within ~0.001 for df >= 5).

### Examples

- `trend_and_forecast.py` gains a `fit_trend` section: it fits on raw vs deseasonalized
  history (showing the recommended `complete -> deseasonalize -> fit_trend` order sharpens
  the fit, R^2 ~0.27 -> ~0.99), and contrasts the regression's robustness to a single odd
  month against a two-point CAGR.

## 0.25.0

Expands the deterministic reserving methods: alongside chain ladder, `develop_ultimate`
adds Bornhuetter-Ferguson, Benktander, and Cape Cod, selected by a `method=` parameter.
All blend emerged-to-date with an a priori, which the library takes as input -- it applies
a method, it does not pick the a priori or the exposure base. Stochastic reserving (Mack,
bootstrap) is intentionally out of scope: it shifts the library from a deterministic
point-estimate engine toward a stochastic-reserving package, beyond the intent of basic
day-to-day tools.

### Added

- `develop_ultimate(df, factors, *, method, value_col, date_col=None, valuation_date=None,
  development_col=None, apriori_col=None, exposure_col=None, by=None, ...)` in `reserving`.
  Methods:
  - `"chain_ladder"` -- `paid / emerged` (equivalent to `apply_completion`).
  - `"bornhuetter_ferguson"` -- `paid + apriori * (1 - emerged)`; stable for immature
    periods. Requires `apriori_col` (an expected ultimate per row).
  - `"benktander"` -- one BF iteration using the BF ultimate as the a priori; a credibility
    blend sitting between BF and chain ladder. Requires `apriori_col`.
  - `"cape_cod"` -- BF with the a priori derived from the data: one expected loss ratio per
    segment, `sum(paid) / sum(exposure * emerged)`. Requires `exposure_col`.
  All four share the completion-factor join (flat or per-segment via `by=`, beyond-the-
  triangle rows fully emerged), and BF/Benktander/Cape Cod accept `by=` per-segment tables.

### Changed

- The proportion-emerged-per-row computation is now a shared internal helper behind both
  `apply_completion` and `develop_ultimate`, so the development-period join, the
  beyond-the-triangle rule, and the grouped/flat factor handling are identical across
  methods. `apply_completion` behaviour is unchanged.

### Examples

- `reserving_ibnr.py` gains a method-comparison section: chain ladder vs Bornhuetter-
  Ferguson vs Benktander vs Cape Cod on the latest diagonal, showing chain ladder swinging
  for the greenest months while the a-priori-anchored methods stay stable.

## 0.24.0

A maintenance and confidence release: an end-to-end renewal example, a fix to
`development_months` for mixed scalar/Series arguments, and property-based tests pinning
the library's advertised invariants.

### Fixed

- `development_months` (and its `lag_months` alias) now accepts a scalar, a Series, or
  array-like for *either* argument in any combination -- e.g.
  `development_months(df["incurred"], "2024-12-31")` (a column of dates against a single
  valuation date), which previously raised `AttributeError`. Internal callers are
  unaffected (they already passed aligned Series).

### Added

- `examples/renewal.py`: a complete group renewal threading the individual surfaces into
  one study -- complete immature months to ultimate, trend, apply area / benefit
  relativities (`adjust`), pool large claimants, credibility-blend the pooled experience
  with the manual, load for retention, and read off the indicated rate change, with the
  projected claims PMPM printed as it builds.
- Property-based tests (`tests/test_properties.py`, `hypothesis`) asserting invariants over
  many generated inputs: deseasonalize/reseasonalize are inverses; `adjust` multiply/divide
  are inverses; an `audit_col` accumulates exactly the product of the factors applied; the
  factor join is by value and order-preserving regardless of index; chain-ladder completion
  factors are bounded in `(0, 1]`, monotone, and reach `1.0`; and `development_months`
  counts the months added to an origin. `hypothesis` added to the `dev` extra.

## 0.23.0

Adds `adjust` -- the general factor-application lens behind experience-period
restatement (trend, benefit / area / demographic relativities, network discounts). Joins a
factor to each row by a key and multiplies or divides the value by it, with an optional
cumulative audit trail. Completion and seasonality are recast as specializations of the
same move, all now sharing one validated join primitive.

### Added

- `adjust(df, factors, *, value_col, on=None, by=None, how="multiply", factor_col="factor",
  out_col=None, audit_col=None, default=None)` in the new `adjustments` module. ``factors``
  is a scalar (one factor for all rows), a Series indexed by ``on``, or a tidy DataFrame
  keyed by ``by + on``. ``how="divide"`` backs a factor out; ``default`` controls the
  treatment of keys absent from the table (``NaN`` surfaced by default, ``1.0`` for "no
  adjustment"); ``audit_col`` accumulates the net multiplier applied across a chain of
  adjustments, one value per row.
- `Experience.adjust(factors, *, on=None, columns=None, by=None, how="multiply",
  factor_col="factor", audit_col=None, default=None)`: restates the expense column(s) in
  place under the same name, so a renewal reads as one composable chain
  (`exp.complete(...).adjust(trend).adjust(relativity, on="plan")`), with the audit trail
  carried across it. The general counterpart to `complete` and `deseasonalize`.
- `factor_lookup(df, factors, keys, *, factor_col, default=None)` in `columns`: the single
  factor-join primitive (join by value on one or more existing key columns, fan-out
  guarded, index-independent, surfaced gaps) now shared by `adjust`, grouped completion,
  and grouped seasonality.

### Changed

- `grouped_factor_lookup` is now a thin wrapper over `factor_lookup` for the derived-key
  case (season from a date, development period); behaviour is unchanged. Completion and
  seasonality are unchanged behaviourally but now sit on the shared primitive --
  `deseasonalize`/`apply_seasonality` are `adjust` on a derived season key, and
  `apply_completion` is `adjust` on a derived development-period key plus its
  beyond-the-triangle "fully complete" rule (verified by equivalence tests).

## 0.22.0

Adds grouped (per-segment) factor joins for both completion and seasonality, so factors
that differ by line of business, product, or any grouping -- and at *different*
granularities for completion vs. seasonality -- join correctly on group plus period.
Renames the reserving "lag" vocabulary to "development", the term recognized across life,
health, and P&C (the old `lag_months` name is kept as an alias).

### Added

- `completion_factors_by(...)` factor tables and `seasonality_factors_by(...)` now feed
  grouped application: `apply_completion(df, factors, *, by=..., factor_col=..., 
  development_name=...)` and `deseasonalize`/`apply_seasonality(df, factors, *, by=...,
  factor_col=..., season_name=...)` accept a tidy per-segment factor table and join each
  row on its grouping column(s) plus development period / season.
- `seasonality_factors_by(df, *, groupby, ...)` in `seasonality`: seasonal factors per
  segment as a tidy `[*groupby, season, seasonal_factor]` table, mirroring
  `completion_factors_by`. Seasons absent from a segment's history are omitted (surface as
  `NaN` on join); `warn=False` silences the thin-history warning per segment.
- `Experience.complete` and `Experience.deseasonalize` accept `by=` to apply per-segment
  factor tables in place; because each adjusts the expense column under the same name,
  completion and seasonality compose even when keyed at different groupings (e.g.
  `exp.complete(cf_by_lob, by="line_of_business").deseasonalize(sf_by_lob_product,
  by=["line_of_business", "product"])`).
- `grouped_factor_lookup(...)` in `columns`: the shared join primitive behind the above.
  Matches by value (frame index irrelevant, order preserved), validates the factor table
  is unique on `by + [key]` so a duplicate cannot silently fan out the data, and leaves an
  absent `(group, key)` as `NaN`.

### Changed

- **Reserving "lag" renamed to "development".** `development_months(incurred, valuation)`
  replaces `lag_months` (kept as a backwards-compatible alias); the default triangle
  column label is now `development_month` (was `lag_month`); `apply_completion`'s and
  `Experience.complete`'s maturity parameter is now `development_col` (was `lag_col`); the
  `make_completion_triangle` / `completion_factors_by` parameter is now `development_name`
  (was `lag_name`); and `ChainLadder.project` returns `latest_development` (was
  `latest_lag`). Prose and messages updated to "development period" throughout.
- For grouped completion, the "fully complete beyond the triangle" rule is applied per
  segment: a row past *its own group's* last development period is complete; a group
  absent from the factor table stays `NaN`.

### Notes

- Breaking: code passing `lag_col=` / `lag_name=`, or reading the `lag_month` triangle
  column or the `latest_lag` projection column, must use the `development*` names. The
  `lag_months` function alias is the one old name retained.

## 0.21.0

Adds application of completion factors -- developing paid claims to estimated ultimate
-- as a free function and an `Experience` lens, the reserving counterpart to the
seasonality tools.

### Added

- `apply_completion(df, factors, *, value_col, date_col, valuation_date, lag_col=None,
  ...)` in `reserving`: grosses a paid amount up to estimated ultimate,
  `completed = paid / completion_factor[lag]`. Each row's development lag is
  `lag_months(date, valuation_date)` (the convention `make_completion_triangle` uses, so
  factors from `completion_factors` join by construction) or an explicit `lag_col`. The
  join is by lag value -- never index alignment -- so the frame's index is irrelevant; a
  convention mismatch surfaces as `NaN` rather than silent corruption. Rows past the
  triangle's last lag are taken as fully complete (factor 1.0); a negative lag raises. On
  the latest diagonal this reproduces `ChainLadder.project`'s per-origin ultimates
  exactly.
- `Experience.complete(factors, *, valuation_date, columns=None, lag_col=None,
  date_col=None)` applies the above to the expense (loss / claims) columns in place under
  the same names and returns a new `Experience`, so downstream views run on the completed
  series. Only the numerator is developed; exposure is left untouched. Run completion
  before deseasonalizing and trending.

### Changed

- The `reserving` module previously declined to apply completion factors on the grounds
  that the join was the caller's to define. `apply_completion` now provides that
  application under one explicit, well-defined join contract; the module documentation
  reflects the change.

## 0.20.0

Adds a deseasonalize lens to the `Experience` facade.

### Added

- `Experience.deseasonalize(factors, columns=None, freq="M", date_col=None)` returns a
  new `Experience` with the seasonal pattern divided out of the expense (loss / claims)
  columns in place under the same names, so every downstream view (`trend`, `rolling`,
  `by`, ...) then operates on the deseasonalized series without a per-view flag. Only
  the numerator is adjusted -- exposure is left untouched -- so a deseasonalized PMPM is
  deseasonalized claims over unchanged member months. `factors` are expected to be
  estimated on the broader pool (via `seasonality_factors`) and applied to the group.

## 0.19.0

Adds seasonality and working-day (business-day) adjustment for monthly and quarterly
series.

### Added

- `seasonality` module. `business_days_in_period(periods, freq, holidays, weekmask)`
  counts weekdays minus holidays in each period (US federal holidays via pandas by
  default, with `None` or a custom date list also accepted), and `add_business_days`
  attaches the count as a column so a paid amount can be put on a per-working-day basis.
- `seasonality_factors(df, date_col, value_col, exposure_col=None, freq="M", ...)`
  estimates one multiplier per calendar period (12 monthly or 4 quarterly), normalized
  to average 1.0, by the classical ratio-to-moving-average decomposition (a centered
  2x12 / 2x4 moving average removes trend and level). Passing `exposure_col` computes
  the factors on a rate (e.g. PMPM), the right basis for health. Options: a simpler
  `method="period_share"`, `aggregate="median"` for robustness to outlier months,
  `exclude=[years]` to drop distorted years (e.g. COVID), and a `min_years` data-
  sufficiency warning.
- `deseasonalize` and `apply_seasonality` divide the seasonal pattern out of a value
  column or multiply it back when projecting a flat annual figure onto specific periods.

## 0.18.0

Adds limited-fluctuation (classical) credibility and utilization/PMPM trend
decomposition.

### Added

- `limited_fluctuation_z(exposure, full_credibility_standard)` in `credibility`: the
  square-root rule, `Z = min(1, sqrt(exposure / standard))`, used by most group
  experience rating. Works on scalars and per-group Series, and feeds
  `credibility_weighted_estimate`. `full_credibility_claims(confidence, tolerance,
  severity_cv)` derives a classical full-credibility standard (~1082 claims at
  90%/5%) for those not using a filed value.
- `decomposition` module: `frequency_severity_summary` (per-group frequency,
  severity, `util_per_1000`, and PMPM, with `pmpm == frequency * severity`) and
  `decompose_pmpm_trend`, which splits the PMPM change between two periods into
  utilization and unit-cost effects -- both multiplicatively (`pmpm_trend ==
  util_trend * cost_trend`) and additively (`pmpm_change == util_effect +
  cost_effect`), each exact.
- `utilization_per_1000(claim_count, exposure, annualization=12)` metric.

## 0.17.0

Surfaces cohort coverage so partial (not-yet-mature) cohorts can be identified and
excluded.

### Added

- `cohort_summary` (and `Experience.cohort`) now report, per cohort, how much of the
  duration window is actually present: `months_observed` (distinct duration months
  present), `last_month` (latest experience month, giving the available range with
  `first_month`), and `complete` (`months_observed == duration_months`). Filter to
  mature cohorts with `cohorts[cohorts["complete"]]` before trending across vintages.

### Changed

- `cohort_summary` output now leads with the entity identity columns (`entity`,
  `first_month`, `cohort_year`), then the coverage columns, then the metrics.

## 0.16.0

Extends the consistent, readable column ordering to the actual-vs-expected
comparison outputs, so variances and ratios sit predictably on the right.

### Changed

- `summarize_actual_vs_expected` now emits columns in a fixed, readable order: date-
  like grouping columns first, then other groups, then exposure, then the actual
  block (components, total, per-exposure rate), then the expected block, then the
  comparison metrics on the right -- variance and its per-exposure rate(s), variance
  percent, and finally the actual-to-expected ratio. Each total now sits beside its
  own per-exposure rate (e.g. actual beside actual PMPM) instead of the rates being
  stranded at the far right.
- `forecast.compare_actual_to_expected` orders its appended metrics as variance,
  variance percent, then the actual-to-expected ratio (ratio last), matching
  `summarize_actual_vs_expected`.
- The date-detection helper used for column ordering now lives in `columns` as
  `is_date_like` and is shared by the experience and actual-vs-expected summaries.

## 0.15.0

Makes experience-summary view output consistent and easier to read, and removes the
single-table actual-vs-forecast helper in favor of the two-table comparison.

### Changed

- `summarize_experience` (and everything built on it -- `summarize_views`,
  `Experience.by`, `Experience.views`, `status_summary`) now emits columns in a
  fixed, readable order, identical across every view: date-like grouping columns
  first, then other grouping columns, then exposure, then the full expense block
  (components, total, then per-exposure rates), then the full revenue block, then the
  ratio. Each total now sits next to its own per-exposure rate (e.g. total expense
  beside expense PMPM) instead of being split by the ratio and the other measure.

### Removed

- `compare_actual_vs_forecast` and `actual_vs_forecast_views` (added in 0.13.0). Use
  `forecast.compare_actual_to_expected` with two tables (actuals and forecast)
  instead; for a single long table with an actual/forecast indicator column, split it
  into two with a boolean mask first.

## 0.14.0

Fixes two-table actual-vs-forecast comparison when both tables use the same value
column name, and makes the resulting column names meaningful.

### Fixed

- `forecast.compare_actual_to_expected` no longer raises `KeyError` when the actual
  and expected frames share an amount-column name (e.g. both call it `amount`).
  Previously the merge produced `amount_x`/`amount_y` and the metric step failed.

### Added

- `suffixes` parameter (default `("actual", "expected")`) on
  `compare_actual_to_expected`: on a name collision the output columns become
  `{actual_col}_{suffixes[0]}` / `{expected_col}_{suffixes[1]}` -- e.g.
  `amount_actual` / `amount_expected`, or pass `("actual", "forecast")` for
  `amount_forecast`. Distinct column names are left unchanged (backward compatible).

## 0.13.0

Adds actual-versus-forecast comparison for long-format tables -- compare a single
table that flags each row as an actual or a forecast, without pivoting it to wide
form first.

### Added

- `compare_actual_vs_forecast(df, *, indicator_col, actual_label, forecast_label,
  amount_col, groupby=None, ...)` in `expected`: splits a long table on an
  actual/forecast indicator column, aggregates the amount within each grouping,
  aligns actuals and forecasts on the grouping keys, and returns the actual,
  forecast, variance, variance percent, and actual-to-forecast ratio. All-missing
  groups stay NaN (so unavailable actuals are distinguishable from zero).
- `actual_vs_forecast_views(df, *, ..., views={...})`: runs several named
  comparisons (total, by month, by grouping) in one call, mirroring
  `Experience.views`.

## 0.12.0

Adds per-segment completion factors -- fit a separate development pattern for each
line of business (or any split) from one dataset, instead of looping by hand.

### Added

- `chain_ladder_by(df, *, groupby, origin_col, valuation_col, amount_col, ...)` in
  `reserving`: groups the development data, builds a triangle per segment, and fits
  a `ChainLadder` to each, returning `{segment: ChainLadder}`.
- `completion_factors_by(...)`: the same, returned as a tidy
  `[<groupby>, lag_month, completion_factor]` table for review or joining.
- `InsufficientDataWarning`.

Segments too small to fit are handled by `on_insufficient`: `"raise"` (default,
names the failing segment), `"skip"` (omit them), or `"aggregate"` (use the pooled
whole-data pattern for them). A `warn` flag plus the standard `warnings` filters
control whether skipped/aggregated segments are reported -- use
`on_insufficient="skip", warn=False` to ignore thin segments silently. Both
functions are re-exported at the top level.

## 0.11.0

Adds chain-ladder estimation, so completion factors can now be **calculated** from
a development triangle rather than only supplied. This fills the gap between
building a triangle and applying factors.

### Added

- `ChainLadder` in the `reserving` module. `ChainLadder.fit(triangle, method=,
  tail=)` estimates a development pattern from a cumulative triangle (e.g. the
  output of `make_completion_triangle(..., cumulative=True)`) and exposes
  `age_to_age` (link factors), `cdf` (cumulative development factors to ultimate),
  and `completion_factors` (`1 / cdf`, divide-convention, in (0, 1]). `method` is
  volume-weighted (default) or simple-average; `tail` (>= 1) extends development
  past the latest observed lag. `ChainLadder.project(triangle)` returns per-origin
  latest, ultimate, and IBNR.
- `completion_factors(triangle, *, method=, tail=)` -- a one-line convenience
  returning just the completion-factor series.

Both are re-exported at the top level. The estimator is deterministic, so it lives
in `reserving` next to the triangle tooling rather than in a modeling satellite.

## 0.10.0

Sharpens the boundary around claims completion. Completing claims from a supplied
factor is a one-line multiply best done upstream, and a factor arriving in a
separate table needs a join only the caller can define unambiguously -- so factor
application is no longer part of the library. `Experience` is unchanged: bind
completed claims with the incurred date for the experience view, or paid claims
with the paid date for a finance / paid-basis view (every method already accepts a
`date_col` override, so one frame can even carry both dates).

### Removed (breaking)

- `complete_claims`, `complete_claim_components`, `completed_from_factor`, and
  `completed_experience`. Complete claims in your own pipeline (a single multiply,
  or a join you control), then pass the completed claims to `Experience`.
- The `completion` module is gone.

### Changed

- New `reserving` module holds the claims-development primitives that are *not* a
  one-line multiply: `make_completion_triangle`, `lag_months`, the `ibnr` identity,
  and `validate_completion_factors` (all still re-exported at the top level). These
  consume a compact development aggregate (origin x valuation), not line-level data.
- `make_completion_triangle` now returns a **cumulative** triangle by default (the
  usual basis for estimating development factors), with `cumulative=False` for the
  incremental triangle. The previous behavior was incremental but documented as
  cumulative; the flag makes the choice explicit.

## 0.9.0

Restores the capabilities removed during the Experience-centric refactor and
wires them into the new `Experience` object, plus fixes. The redesign itself
(the immutable `Experience` dataclass with `.filter()` / `.with_roles()`, the
`claimants` and `expected` modules) is kept.

### Added (restored)

- **Credibility** (`credibility`): `Buhlmann`, `BuhlmannStraub`, and
  `credibility_weighted_estimate`, plus a new `Experience.credibility_weighted`
  method that blends each group's metric with a complement at a given Z.
- **Lifecycle** (`lifecycle`): `derive_status`, `add_tenure`, `is_in_force`,
  `add_months_in_force`, `earned_exposure`, and the status constants, plus a new
  `Experience.with_status(...)` that returns a new `Experience` carrying a
  derived active/first-year/termed column (then summarize with `.by_status`).
- **Banding** (`banding`): `assign_band` and `summarize_by_band`, plus
  `Experience.by_band(...)`.
- **Margins** (`margins`): `margin`, `margin_ratio`, `add_margin`, plus
  `Experience.margin(...)` (computes margin from the bound roles, with an
  optional per-exposure margin).
- **Large-loss pooling** (`pooling`): `pool_losses` (cap at a pooling point →
  pooled/excess split) and `excess_over_threshold` (the per-claim excess sample
  that feeds tail/severity modeling in `extremeloss`/`lossmodels`/`risksim`),
  plus `Experience.pool_claimants(...)`. Descriptive large-claim flagging and
  concentration continue to live in `claimants` (`large_claimant_flags`,
  `claim_concentration`), so only the capping transform and the modeling
  hand-off were re-added.
- **`permissible_loss_ratio`** in `metrics` (CAS permissible / zero-margin loss
  ratio).
- **`completed_experience`** in `completion` (completion → experience in one
  call).

### Fixed

- **Default ratio name restored to `loss_ratio`.** `summarize_experience` had
  reverted to `expense_revenue_ratio` and `rolling_summary` to `mlr`; both now
  default to the general `loss_ratio` again (`profile="health"` still yields
  `mlr`).
- **Type-checking**: the package is mypy-clean again. Fixed pre-existing
  annotation issues in `metrics.safe_divide`, `trend` date-range narrowing,
  `forecast` merge `how`, and several `Experience` field/`filter`/`views`
  annotations.

## 0.8.0

Ergonomics: the `Experience` facade (bind the expense/revenue/exposure/profile
roles once, then `.by`/`.rolling`/`.trend`/`.by_band`/`.by_status`),
`completed_experience`, and a `trend_summary` fix so the period column no longer
leaks as `year_x`/`year_y`.

## 0.7.0

Generalized terminology across SOA and CAS lines (default ratio `loss_ratio`,
health available via the `health` profile) and added lifecycle, banding,
concentration, margins, and large-loss pooling, plus `permissible_loss_ratio`.
