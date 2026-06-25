# ActuarialPy

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

An experience-centered Python toolkit for actuarial analysis on tabular data.

---

## Overview

ActuarialPy turns a claims / exposure / premium table into a set of standard
actuarial analyses with very little ceremony. You define the actuarial *roles* of
your columns once — what is expense, revenue, exposure, and the date — and then ask
the resulting `Experience` object for summaries, rolling views, trends, driver
analyses, actual-versus-expected, concentration reviews, cohort/duration tables,
and more. The same role assignments are reused across every analysis, so you never
re-specify which column is premium or how to compute a loss ratio.

Alongside the `Experience` object, every calculation is also available as a plain
function (loss ratios, PMPM/PSPM/PEPM, trend factors, chain-ladder completion
factors, credibility formulas), so the library works equally well for one-off
calculations and for building larger pipelines.

It is deterministic and pandas-native: the only dependencies are `numpy` and
`pandas`, and outputs are ordinary DataFrames and Series you can inspect, join, and
export.

## Highlights

- **The `Experience` object** — assign column roles once, then produce grouped
  summaries, rolling windows, trends, component drivers, actual-vs-expected,
  claimant concentration, cohort/duration tables, size bands, and lifecycle views.
- **Ratios and per-exposure metrics** — loss ratio, medical loss ratio, combined
  and expense ratios, pure premium, frequency, severity, and PMPM / PSPM / PEPM /
  generic per-exposure metrics.
- **Reserving** — development triangles, completion factors, ultimates, and IBNR
  (overall or per segment), and a choice of development method — chain ladder,
  Bornhuetter-Ferguson, Benktander, Cape Cod.
- **Credibility** — greatest-accuracy credibility (Bühlmann and Bühlmann-Straub)
  and direct credibility-weighting of any estimate.
- **Trend and forecasting** — fit a trend from history (log-linear, with goodness of fit
  and a confidence interval), apply trend factors, project values forward, period-over-
  period comparisons, and rate-based forecasts.
- **Seasonality and working days** — per-month working-day counts (holidays included)
  and classical seasonal factors (overall or per segment) to deseasonalize history or
  project a pattern forward.
- **Adjustments and restatement** — `adjust` joins a factor by key (scalar, per-key, or
  per-segment) and multiplies or divides a column in place, with an optional cumulative
  audit trail — the general spine behind completion, trend, and relativity loads.
- **Lifecycle and exposure** — in-force determination, earned exposure, tenure,
  and active / first-year / termed status.
- **Pooling, banding, and margins** — pooling points and excess losses, size
  banding, and underwriting margins.
- **Reporting** — write a dictionary of analysis views to a multi-sheet workbook.

## Installation

```bash
pip install actuarialpy
```

From a source checkout:

```bash
pip install -e .
```

Requires Python `>=3.10`; depends only on `numpy` and `pandas`.

## Package structure

```text
actuarialpy/
├── frame.py         # the Experience object (role-aware analysis facade)
├── experience.py    # grouped experience summaries and multi-view summaries
├── rolling.py       # rolling-window experience summaries
├── trend.py         # trend factors, projections, period-over-period comparisons
├── forecast.py      # rate-based forecasting and actual-vs-forecast
├── components.py    # component / driver decompositions
├── decomposition.py # frequency-severity & PMPM trend decomposition
├── contribution.py  # share-of-total and contribution-to-change
├── expected.py      # actual-versus-expected summaries
├── claimants.py     # large-claimant flags and concentration
├── pooling.py       # pooling points and excess losses
├── cohorts.py       # cohort and duration summaries
├── lifecycle.py     # in-force, earned exposure, tenure, status
├── banding.py       # size banding and banded summaries
├── margins.py       # underwriting margins
├── metrics.py       # ratios and per-exposure metrics
├── credibility.py   # Bühlmann, Bühlmann-Straub & limited-fluctuation credibility
├── reserving.py     # triangles, chain ladder, completion factors, IBNR
├── seasonality.py   # working-day counts and seasonal factors
├── periods.py       # period / duration column helpers
├── profiles.py      # naming profiles (e.g. health vs. P&C terminology)
└── reporting.py     # multi-sheet workbook export
```

## The Experience object

Define the roles of your dataset once:

```python
import actuarialpy as ap

exp = ap.Experience(
    experience_data,             # a pandas DataFrame of claims, premium, and exposure
    expense="total_claims",      # the loss / claim amount column(s)
    revenue="premium",           # the premium / revenue column(s)
    exposure="member_months",    # the exposure column(s)
    date="incurred_month",       # the time column
    count="claim_count",         # a claim/service count -> frequency-severity views
    profile="health",            # naming profile (health -> MLR, PMPM terminology)
)
```

Then reuse it across analyses:

```python
# grouped experience summary (totals + loss ratio / MLR by group)
summary = exp.by(["group_id", "product_code"])

# trailing-12-month rolling view per product
rolling = exp.rolling(window=12, groupby="product_code")

# period-over-period trend
trend = exp.trend(
    prior_start="2025-01-01", prior_end="2025-12-31",
    current_start="2026-01-01", current_end="2026-12-31",
    groupby="product_code",
)

# frequency / severity / PMPM, and the PMPM trend split into utilization x unit cost x mix
fs = exp.frequency_severity(groupby="product_code")
drivers = exp.decompose_trend(period_col="year", prior_period=2025, current_period=2026, mix_by="segment")
slope = exp.fit_trend()        # log-linear PMPM trend with goodness of fit (developed, not received)

# several named cuts at once
views = exp.views({"overall": None, "by_group": "group_id", "by_product": "product_code"})
```

`Experience` is immutable-friendly: `exp.filter(query="product_code == 'A'")` and
`exp.with_roles(...)` return new objects without mutating the original.

### Experience methods

| Method | Produces |
| --- | --- |
| `by(groupby)` | grouped totals and ratios |
| `views(views)` | a dict of named grouped summaries |
| `rolling(window, groupby)` | rolling-window summaries |
| `trend(prior/current windows, groupby)` | period-over-period trend |
| `components(component_cols, ...)` / `component_summary(...)` | component / driver breakdowns |
| `actual_vs_expected(expected, actual, ...)` | actual-versus-expected with variances |
| `claimants(...)` / `top_claimants(...)` / `claimant_concentration(...)` | large-claimant and concentration views |
| `pool_claimants(claimant_col, pooling_point)` | pooled vs. excess by claimant |
| `cohort(...)` / `duration(...)` | cohort and duration summaries |
| `by_band(value_col, bands)` | banded summaries |
| `with_status(...)` / `by_status(...)` | lifecycle status assignment and summary |
| `margin(...)` | underwriting margins |
| `credibility_weighted(groupby, z, metric)` | credibility-blended estimates by group |
| `deseasonalize(factors)` / `complete(factors, valuation_date)` | new Experience with seasonality removed / claims developed to ultimate |
| `adjust(factors, on, by, how, audit_col)` | new Experience with a column restated by a joined factor (trend, relativity, discount) |
| `filter(...)` / `with_roles(...)` | derive a new Experience |

## Reserving

Build a development triangle from transactional data, fit a chain ladder, and read
off ultimates and IBNR. Origin and development periods are derived for you.

```python
from actuarialpy import make_completion_triangle, ChainLadder, completion_factors

triangle = make_completion_triangle(
    experience_data,
    origin_col="incurred_month",
    valuation_col="paid_month",
    amount_col="paid",
    cumulative=True,
)

cl = ChainLadder.fit(triangle, method="volume", tail=1.0)
projection = cl.project(triangle)
# projection columns: latest_development, latest, development_factor, ultimate, ibnr

# or just the completion factors (1 / cumulative development factor)
factors = completion_factors(triangle, method="volume", tail=1.0)
```

`ChainLadder.fit` exposes `age_to_age`, `cdf`, `completion_factors`, `tail`, and
`method`. Segment-level reserving is available without manually splitting data —
`chain_ladder_by` returns a `{segment: ChainLadder}` mapping, and `completion_factors_by`
returns a **tidy table** of factors, one row per `(segment, development_month)`:

```python
from actuarialpy import chain_ladder_by, completion_factors_by

cf_by_lob = completion_factors_by(
    experience_data, groupby="line_of_business",
    origin_col="incurred_month", valuation_col="paid_month", amount_col="paid",
    on_insufficient="skip",   # "raise", "skip", or "aggregate"
)
# columns: line_of_business, development_month, completion_factor
```

Supporting helpers: `development_months(incurred, valuation)` (still available under the
older name `lag_months`), `ibnr(completed, paid)`, and `validate_completion_factors(...)`
(which checks a factor column is in range and warns on `InsufficientDataWarning`).

**Applying completion factors.** Estimating factors and applying them are kept separate,
because applying hinges on a join — each row's development period matched to the right
factor. `apply_completion` commits to one well-defined contract: factors keyed by
development period, each row's period taken as `development_months(incurred, valuation)`
(the same convention `make_completion_triangle` uses, so factors from the pipeline above
join by construction), matched **by value** so the frame's index is irrelevant. Rows past
the triangle's last development period are taken as fully complete; only recent, immature
months move.

```python
from actuarialpy import apply_completion

# latest diagonal: one row per incurred month, claims paid-to-date as of the valuation
completed = apply_completion(
    latest_diagonal, factors,
    value_col="claims", date_col="incurred_month", valuation_date="2024-12-31",
)
# completed["claims_completed"] == paid / completion_factor; equals ChainLadder's
# per-origin ultimate. Pass development_col=... instead if you already carry a
# maturity column.
```

**Per-segment factors (grouped join).** When the pattern differs by segment, pass the
tidy table from `completion_factors_by` together with `by=` naming the grouping
column(s). Each row is joined on its group **and** development period: a row past *its own
group's* last development period is complete, an absent `(group, period)` stays `NaN` (a
surfaced gap, never silently filled), and a duplicated `(group, period)` in the factor
table is rejected rather than allowed to fan out the data.

```python
completed = apply_completion(
    latest_diagonal, cf_by_lob, by="line_of_business",
    value_col="claims", date_col="incurred_month", valuation_date="2024-12-31",
)
# each segment is developed by its own pattern, reproducing that segment's own chain ladder
```

On the `Experience` facade, `complete` grosses the expense column up to ultimate in
place under the same name, so every downstream view runs on completed claims — the
reserving counterpart to `deseasonalize`. It takes the same `by=` for per-segment factors:

```python
clean = exp.complete(factors, valuation_date="2024-12-31")
clean.rolling(12)        # runs on completed claims; only the green months changed

# or per line of business, each developed by its own completion pattern:
clean = exp.complete(cf_by_lob, valuation_date="2024-12-31", by="line_of_business")
```

**Other development methods.** Chain ladder takes the entire unemerged tail from the data,
which is volatile for green periods -- a thin latest diagonal drives the whole estimate.
`develop_ultimate` offers the standard alternatives that blend emerged-to-date with an a
priori, selected by a `method=` parameter and all sharing the same completion-factor join:

```python
from actuarialpy import develop_ultimate

# Bornhuetter-Ferguson: paid + apriori * (1 - emerged); stable for immature periods
bf = develop_ultimate(
    latest_diagonal, factors, method="bornhuetter_ferguson",
    value_col="claims", date_col="incurred_month", valuation_date="2024-12-31",
    apriori_col="expected_ultimate",   # a plan / budget / manual x exposure -- an input
)
# bf["claims_ultimate"] is paid where fully emerged, anchored to the a priori where green
```

`method=` also takes `"chain_ladder"` (equivalent to `apply_completion`), `"benktander"`
(one BF iteration, sitting between BF and chain ladder), and `"cape_cod"` (BF with the a
priori derived from the data as a single expected loss ratio per segment,
`sum(paid) / sum(exposure * emerged)`, taking `exposure_col` instead of `apriori_col`).
BF, Benktander, and Cape Cod all accept `by=` for per-segment factor tables. As everywhere,
the library applies the method; it does not pick the a priori or the exposure base.

Run completion **before** deseasonalizing and trending — `complete → deseasonalize →
trend` — since deseasonalizing paid claims would tangle claims runout with the calendar
effect. Only the numerator is developed; exposure is left untouched.

## Adjustments and restatement

Much of experience rating is one move repeated: take a base amount and carry it through a
chain of factors — completion to ultimate, trend, benefit relativity, area, demographic
loads, network discounts. `adjust` is that move, once: join a factor to each row by a key
and multiply (or divide) the value by it. It is the general case of which `complete` and
`deseasonalize` are specializations — those derive their key from a date (a development
period, a season); `adjust` keys on an ordinary column. All three share the same validated
join (unique-key / fan-out guard, surfaced gaps, index-independent).

```python
from actuarialpy import adjust

# a scalar (one trend factor for every row)
adjust(experience, 1.072, value_col="claims")

# a Series indexed by the key column
area = pd.Series({"urban": 1.08, "suburban": 1.00, "rural": 0.94})
adjust(experience, area, value_col="claims", on="region")

# a tidy per-segment table (factor varies by segment), joined on by + on
adjust(experience, benefit_relativity, value_col="claims", on="plan", by="line_of_business")
```

`how="multiply"` (default) loads the value up; `how="divide"` backs a factor out. An absent
key surfaces as `NaN` (never silently filled); pass `default=1.0` when a key missing from
the table should mean "no adjustment". The methodology stays *input* — `adjust` takes the
trend, the relativity, the discount as given and applies them mechanically; it does not
encode any particular method.

On the `Experience` facade, `adjust` restates the expense column in place under the same
name, so a renewal reads as one composable chain — and `audit_col` carries the cumulative
restatement multiplier across the chain, one value per row, for a reviewable trail:

```python
restated = (
    exp.complete(cf_by_lob, valuation_date="2024-12-31", by="line_of_business")
       .adjust(1.072, audit_col="restatement")                       # trend
       .adjust(benefit_relativity, on="plan", by="line_of_business", audit_col="restatement")
       .adjust(area, on="region", audit_col="restatement")
)
# restated.data["restatement"] is the total factor applied to each row's claims
restated.by()   # grouped loss ratios on the fully restated claims
```



Greatest-accuracy credibility, either fit empirically from per-risk observations or
constructed from known structural parameters.

```python
from actuarialpy import Buhlmann, BuhlmannStraub, credibility_weighted_estimate

# Bühlmann: rows are risks, columns are observed periods
observations = [[10, 12, 9, 11], [20, 18, 22, 19], [5, 6, 4, 7]]
model = Buhlmann.fit(observations)
print(model.z, model.k)               # credibility factor and credibility constant
print(model.premium(risk_mean=11.0))  # credibility-weighted premium for a risk

# Bühlmann-Straub: unequal exposures via weights
ws = BuhlmannStraub.fit(observations, weights=[[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]])
print(ws.z(weight=6), ws.premium(risk_mean=11.0, weight=6))

# or blend any observed estimate with a complement directly
est = credibility_weighted_estimate(observed=0.82, complement=0.75, z=0.6)
```

You can also construct the models directly from parameters —
`Buhlmann(overall_mean, epv, vhm, n_obs)` and
`BuhlmannStraub(overall_mean, epv, vhm, weights)` — when EPV and VHM are already
known.

For the limited-fluctuation (classical) credibility most group experience rating
uses — the square-root rule against a full-credibility standard, often a filed value
— use `limited_fluctuation_z`:

```python
from actuarialpy import limited_fluctuation_z, full_credibility_claims, credibility_weighted_estimate

# Z = min(1, sqrt(exposure / full_credibility_standard)); exposure is claim counts,
# member months, life-years, etc. Works per group on a Series.
groups["z"] = limited_fluctuation_z(groups["claim_count"], full_credibility_standard=1082)
groups["blended_lr"] = credibility_weighted_estimate(groups["experience_lr"], manual_lr, groups["z"])

# derive a standard from first principles instead of using a filed one:
n_full = full_credibility_claims(confidence=0.90, tolerance=0.05)            # ~1082 claims
n_full_agg = full_credibility_claims(confidence=0.90, tolerance=0.05, severity_cv=2.0)  # inflated for severity
```

## Trend and forecasting

```python
from actuarialpy import trend_factor, annualized_trend, project_forward, trend_summary

trend_factor(0.06, months=18)            # (1 + 0.06) ** (18/12)
project_forward(1000.0, 0.06, months=18) # trend a value forward 18 months
annualized_trend(current=1.1, prior=1.0, months_between=12)

# trend a metric between two windows of a transactional frame
summary = trend_summary(
    experience_data, date_col="incurred_month",
    prior_start="2025-01-01", prior_end="2025-12-31",
    current_start="2026-01-01", current_end="2026-12-31",
    amount_col="paid", exposure_col="member_months", groupby="product_code",
)
```

**Fitting trend from data.** The functions above *apply* a trend, or measure it between
two points. To *develop* a trend from history, `fit_trend` regresses `log(rate)` on time
(log-linear OLS) over the whole series -- using every period, so one noisy month doesn't
swing it -- and returns a `TrendFit` with the fitted annual trend, goodness of fit, and a
confidence
interval, which is what a developed (rather than received) trend is judged on:

```python
from actuarialpy import fit_trend

fit = fit_trend(history, value_col="claims", date_col="month", exposure_col="member_months")
fit.annual_trend          # e.g. 0.072  (exp(slope) - 1)
fit.r_squared, fit.ci     # goodness of fit, and the confidence interval
fit.factor(18)            # bridge to application: (1 + annual_trend) ** (18/12)
```

It fits on the rate (`claims / member_months`) when an exposure is given, otherwise on
`value_col` directly; unlike a two-point `annualized_trend`, one odd month barely moves it,
and it reports how well the exponential fits. Time is measured from real dates, so a
missing period is handled correctly. Run it on completed, deseasonalized history
(`complete → deseasonalize → fit_trend`) so runout and seasonality don't bias the slope,
then apply the result with `trend_factor` / `fit.factor(...)` or `adjust`. As always, the
library fits the trend; it does not select it -- the window, the rate basis, benefit
leveraging, and the blend with external trends remain judgment.

Rate-based forecasting lives in the forecast module: `forecast_experience(...)`
applies a trended per-exposure rate to projected exposure, with
`forecast_from_rate(...)`, `expected_from_rate(...)`, and
`compare_actual_to_expected(...)` as supporting helpers.

## Actual versus forecast

To compare actuals against a forecast, supply the two as separate tables — for
example a database pull for paids and a finance workbook for the forecast — and
align them with `compare_actual_to_expected`:

```python
from actuarialpy.forecast import compare_actual_to_expected

variance = compare_actual_to_expected(
    actuals_table,
    forecast_table,
    on=["month", "segment", "product", "category"],
    actual_col="amount",                 # value column in the actuals table
    expected_col="amount",               # value column in the forecast table
    how="outer",                         # keep forecast months without actuals yet
    suffixes=("actual", "forecast"),     # -> amount_actual, amount_forecast
)
# columns: <keys...>, amount_actual, amount_forecast,
#          variance, variance_pct, actual_to_expected
```

The two frames are joined on `on` and the variance, variance percent, and
actual-to-expected ratio are computed. When both tables call their value column `amount`, a
plain merge would collide them into `amount_x` / `amount_y`; instead the output
disambiguates them as `amount_actual` / `amount_forecast` (controlled by `suffixes`,
default `("actual", "expected")`). When the two value columns already have distinct
names, they are left unchanged. With `how="outer"`, a key present in only one table —
such as a future month that has a forecast but no actual yet — is kept, with the
missing side left as `NaN`, so unavailable actuals stay distinguishable from a true
zero.

**Units caveat:** when `category` mixes units (dollars for expenses/revenue, counts
for member months), a grand total over all categories adds unlike quantities and is
not meaningful — keep `category` in the keys, or filter to a single category. For
coarser time buckets (quarter, year), derive a period column first with
`actuarialpy.periods.add_period_column(df, date_col, freq)`.

## Ratios and per-exposure metrics

All of these accept scalars, NumPy arrays, or pandas Series, and divide safely
(returning NaN rather than raising on a zero denominator):

| Function | Definition |
| --- | --- |
| `loss_ratio(losses, revenue)` | losses ÷ revenue |
| `medical_loss_ratio(claims, premium)` | claims ÷ premium |
| `expense_ratio(expenses, revenue)` | expenses ÷ revenue |
| `combined_ratio(losses, expenses, revenue)` | (losses + expenses) ÷ revenue |
| `pure_premium(losses, exposure)` | losses ÷ exposure |
| `frequency(claim_count, exposure)` | claims ÷ exposure |
| `severity(losses, claim_count)` | losses ÷ claim count |
| `pmpm` / `pspm` / `pepm(amount, months)` | per-member / -subscriber / -employee per month |
| `per_exposure(amount, exposure)` | generic per-exposure rate |
| `permissible_loss_ratio(expense_ratio, profit_provision)` | 1 − expense ratio − profit |
| `required_revenue(expense, target_ratio)` | expense ÷ target ratio |
| `indicated_change(required, current)` | required ÷ current − 1 |
| `actual_to_expected(actual, expected)` | actual ÷ expected |

## Utilization and PMPM trend decomposition

Split a per-member cost into its drivers and decompose movement between two periods
into utilization versus unit cost — the standard "how much of the trend is util vs
cost" exhibit. It needs a claim (or service) count alongside losses and exposure.

```python
from actuarialpy import frequency_severity_summary, decompose_pmpm_trend

# per-group panel; pmpm == frequency * severity holds for every row
panel = frequency_severity_summary(
    df, count_col="claim_count", loss_col="claims", exposure_col="member_months",
    groupby="plan",
)
# columns: plan, member_months, claim_count, claims, frequency, severity, util_per_1000, pmpm

# decompose the change between two period slices
trend = decompose_pmpm_trend(
    prior_year, current_year,
    count_col="claim_count", loss_col="claims", exposure_col="member_months",
    on="plan",        # optional; omit for a single total row
)
# pmpm_trend == util_trend * cost_trend (exact), and
# pmpm_change == util_effect + cost_effect (exact, symmetric dollar split)
```

The multiplicative view answers "PMPM grew X%, of which U% utilization and C% unit
cost"; the additive `util_effect` / `cost_effect` give the same split in dollars and
sum exactly to the PMPM change. `util_per_1000` is annualized claims per 1,000
members (pass `annualization=1` if exposure is already in member-years).

**Adding a mix term (`mix_by`).** Pass `mix_by` to split PMPM three ways —
utilization × unit cost × **mix** — when your book is a blend of cells whose
composition is shifting. The two-way above is an exact identity, so a book-wide
utilization or unit-cost trend silently absorbs membership-mix shifts: when enrollment
tilts toward a higher-cost segment, both look like inflation even when nothing inside
any segment changed. `mix_by` measures utilization and unit cost *within* each cell and
reports the composition shift as its own term.

```python
trend = decompose_pmpm_trend(
    prior_year, current_year,
    count_col="claim_count", loss_col="claims", exposure_col="member_months",
    mix_by="segment",        # the cell dimension whose weights can shift
)
# pmpm_trend == util_trend * cost_trend * mix_trend (exact), and
# pmpm_change == util_effect + cost_effect + mix_effect (exact, LMDI dollars)
```

The split uses LMDI (logarithmic mean Divisia index), which is order-free and leaves no
residual. `mix_by` takes a column or a list — a list defines the cells as their cross
(one blended mix term, not a per-column attribution; for separate attribution, run it
once per dimension). `on` and `mix_by` are orthogonal: `on` groups the output rows,
`mix_by` defines the mix cells within each group. Every cell must have positive count,
loss, and exposure in both periods. See
[`examples/trend_decomposition.py`](examples/trend_decomposition.py).

On an `Experience`, the same split is `exp.decompose_trend(...)`: it uses the bound
`count`, `expense` (as the loss), and `exposure` roles and slices prior/current by period
or date range exactly like `exp.trend`, so you specify the columns once.

## Seasonality and working-day adjustment

Two months of claims are rarely comparable as-is. They differ for reasons that have
nothing to do with real changes in your book: a month may have fewer working days
(weekends and holidays close offices), and the time of year matters on its own (flu
season runs hot, summer light). These are separate effects, so there are two tools.

**Working days.** `business_days_in_period` counts weekdays minus holidays in each
period (US federal holidays via pandas, by default — no extra dependency). Dividing a
paid amount by this makes a short month and a long month comparable.

```python
from actuarialpy import business_days_in_period

business_days_in_period(pd.date_range("2024-01-01", "2024-12-01", freq="MS"))
# 2024-01-01    21
# 2024-02-01    20
# ...
# 2024-06-01    19   # fewer working days -> looks lighter on its own
# 2024-11-01    19
# 2024-12-01    21
```

`add_business_days(df, date_col)` adds the count as a column; divide your paid column
by it to get an amount-per-working-day series. `holidays` accepts `"us_federal"`,
`None` (weekdays only), or your own list of dates; `weekmask` sets which weekdays count.

**Seasonal factors.** `seasonality_factors` learns one multiplier per calendar period
from a few years of history — the classical ratio-to-moving-average decomposition,
normalized to average 1.0. Pass `exposure_col` to compute the factors on a rate (PMPM),
which is the right basis for health: it removes membership growth, so only the
time-of-year pattern remains.

```python
from actuarialpy import seasonality_factors, deseasonalize, apply_seasonality

factors = seasonality_factors(
    exp, date_col="month", value_col="claims", exposure_col="member_months",
)
# season
# 1     1.180   # January ~18% hot
# 6     0.886   # June ~11% light
# 12    1.131
```

`deseasonalize` divides the pattern out so you can see the underlying trend;
`apply_seasonality` multiplies it back when projecting a flat annual number onto
specific months. December's raw claims below look alarming, but deseasonalized they
sit right on trend — the jump was just the winter factor.

```python
deseasonalize(recent, factors, date_col="month", value_col="claims")
#      month     claims  member_months  claims_deseasonalized
# 2024-09-01  6,648,743         13,691              6,975,622
# 2024-10-01  7,040,449         13,732              7,024,422
# 2024-11-01  7,298,236         13,773              7,072,117
# 2024-12-01  8,049,170         13,814              7,118,962   # not real growth
```

Seasonal factors are the workhorse and stand on their own — they already absorb each
month's *typical* working-day count. Working-day normalization is an optional
refinement for the residual year-to-year wrinkle (an unusually holiday-heavy month).
When you use both, normalize by working days first, then fit factors on the normalized
series. Other knobs: `freq="Q"` for quarterly factors, `method="period_share"` for the
simpler share-of-year-average estimate, `aggregate="median"` to blunt outlier months,
and `exclude=[2020, 2021]` to keep distorted years out of the estimate.

**On the `Experience` object.** `Experience.deseasonalize(factors)` applies the pattern
once and returns a new `Experience` with the expense column adjusted under the same
name, so every downstream view composes on the deseasonalized series — no per-view flag
needed. Estimate the factors on the broader pool, then apply them to the group:

```python
factors = seasonality_factors(pool, date_col="month", value_col="claims",
                              exposure_col="member_months")   # estimated on the block, once
clean = exp.deseasonalize(factors)   # new Experience; claims divided by its factor, same column
clean.rolling(12)                    # every view now runs on the deseasonalized series
clean.trend(date_col="month", prior_start="2023-01-01", prior_end="2023-12-01",
            current_start="2024-01-01", current_end="2024-12-01")
```

Only the numerator is adjusted (exposure is left alone), so a deseasonalized PMPM is
just deseasonalized claims over unchanged member months. A rolling-12 or any like-for-
like full-year comparison already cancels seasonality on its own, so reach for this
mainly when fitting trend on month-level data or annualizing a partial period.

**Per-segment factors (grouped join).** Seasonality often differs by segment, and the
granularity varies — sometimes line of business, sometimes line × product.
`seasonality_factors_by` fits the pattern within each segment and returns a tidy table,
one row per `(segment, season)`:

```python
from actuarialpy import seasonality_factors_by

sf_by_lob = seasonality_factors_by(
    pool, groupby="line_of_business",
    date_col="month", value_col="claims", exposure_col="member_months",
)
# columns: line_of_business, season, seasonal_factor
```

Pass that table to `deseasonalize`, `apply_seasonality`, or `Experience.deseasonalize`
with `by=` naming the grouping column(s). The join carries the same guarantees as the
reserving side: matched by value (index irrelevant), an absent `(group, season)` left as
`NaN`, and a duplicated key rejected rather than fanned out.

```python
clean = exp.deseasonalize(sf_by_lob, by="line_of_business")
# or at a finer grain, factors keyed by line of business AND product:
clean = exp.deseasonalize(sf_by_lob_product, by=["line_of_business", "product"])
```

Because completion and seasonality each adjust the expense column in place, they compose
even when keyed at *different* groupings — complete by line of business, then
deseasonalize by line × product:

```python
clean = (
    exp.complete(cf_by_lob, valuation_date="2024-12-31", by="line_of_business")
       .deseasonalize(sf_by_lob_product, by=["line_of_business", "product"])
)
```

## Lifecycle, pooling, banding, and margins

- **Lifecycle** (`lifecycle`): `is_in_force(...)`, `earned_exposure(...)`,
  `add_months_in_force(...)`, `add_tenure(...)`, and `derive_status(...)` (which
  labels rows `STATUS_ACTIVE` / `STATUS_FIRST_YEAR` / `STATUS_TERMED`).
- **Pooling** (`pooling`): `pool_losses(df, loss_col, pooling_point)` splits each
  loss into pooled and excess amounts; `excess_over_threshold(...)` isolates the
  excess layer.
- **Banding** (`banding`): `assign_band(df, value_col, bands)` and
  `summarize_by_band(...)` for size-band experience.
- **Margins** (`margins`): `add_margin(...)` / `margin(...)` / `margin_ratio(...)`.
- **Contribution** (`contribution`): `share_of_total(...)`,
  `contribution_to_change(...)`, and `top_contributors(...)`.

## Reporting

Write a set of named analysis views to a multi-sheet Excel workbook:

```python
from actuarialpy.reporting import to_excel_report

views = exp.views({"overall": None, "by_group": "group_id"})
to_excel_report(views, "experience_report.xlsx")
```

## The ActuarialPy ecosystem

ActuarialPy is the deterministic, experience-and-data layer of a small family of
actuarial packages. It is standalone (only `numpy` and `pandas`) and focuses on
turning real data into summaries, triangles, trends, and credibility-weighted
estimates. Three companion packages cover the distributional and simulation side
and interoperate through a simple `.sample()` / `.mean()` interface:

- **`lossmodels`** — frequency and severity distributions, aggregate
  (collective-risk) loss models, coverage modifications, and model fitting.
- **`risksim`** — portfolio loss simulation and aggregate reinsurance program
  evaluation.
- **`extremeloss`** — extreme value theory: tail fitting (peaks-over-threshold /
  GPD, block maxima / GEV), tail risk measures, and threshold diagnostics.

## Testing

```bash
pytest -q
```

## License

MIT License
