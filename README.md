# actuarialpy

Standard actuarial analyses on claims, exposure, and premium data: loss ratios and per-exposure rates,
development triangles and IBNR, trend, credibility, seasonality, and frequency-severity
decomposition, computed from long, transactional tables. The only dependencies are `numpy`
and `pandas`, and every result is a DataFrame or Series.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The `Experience` object](#the-experience-object)
- [Ratios and per-exposure metrics](#ratios-and-per-exposure-metrics)
- [Reserving](#reserving)
- [Trend and forecasting](#trend-and-forecasting)
- [Frequency-severity and per-exposure decomposition](#frequency-severity-and-per-exposure-decomposition)
- [Seasonality and working days](#seasonality-and-working-days)
- [Adjustments and restatement](#adjustments-and-restatement)
- [Actual versus forecast](#actual-versus-forecast)
- [Credibility](#credibility)
- [Lifecycle, pooling, banding, margins](#lifecycle-pooling-banding-margins)
- [Underwriting summary and weighted rollups](#underwriting-summary-and-weighted-rollups)
- [Reporting](#reporting)

## Overview

**`actuarialpy`** is a calculation library for loss ratios, per-exposure rates, chain-ladder
development, credibility, seasonal factors, and the LMDI trend decomposition, applied to
claims, exposure, and premium data. It does not perform data preparation or encode filed
methodology: the caller supplies the table and selects the method.

There are two interfaces:

- **Free functions** — `loss_ratio`, `per_exposure`, `severity`, `trend_factor`, `fit_trend`,
  `seasonality_factors`, `decompose_per_exposure_trend`, the credibility models, and others. Each
  accepts scalars, NumPy arrays, or pandas Series and returns the same type. They operate on
  any frame at any grain.
- **The `Experience` object** — binds the column roles (expense, revenue, exposure, date)
  once and reuses them across grouped summaries, rolling windows, trends, completion,
  seasonality, and more. Use it when running several analyses at one grain.

## Installation

```bash
pip install actuarialpy
```

## Quick start

Pass one table at the grain you are analysing (one row per unit, e.g. per group per month),
name its columns, and request views:

```python
import actuarialpy as ap

exp = ap.Experience(
    data,
    expense="claims",            # the loss / claim amount column(s)
    revenue="premium",           # the premium / revenue column(s)
    exposure="member_months",    # the exposure column(s)
    date="month",                # the time column
    profile="health",            # naming profile (health -> mlr on output views)
)

exp.by("group_id")                          # grouped totals + loss ratio (mlr under the health profile)
exp.rolling(12, groupby="group_id")         # trailing-12 view per group
exp.trend(date_col="month",                 # period-over-period trend
          prior_start="2024-01-01", prior_end="2024-12-01",
          current_start="2025-01-01", current_end="2025-12-01")
```

Build `data` with pandas. Typically this is a single `groupby` that aggregates claims to the
grain, counts exposure from a correctly-grained table (here a health book's
member-months from eligibility), and joins premium:

```python
g = ["group_id", "month"]
data = (claims.groupby(g)["paid_amount"].sum().rename("claims").to_frame()
        .join(eligibility.groupby(g).size().rename("member_months"))   # counted, not summed
        .join(premium.groupby(g)["premium"].sum().rename("premium"))
        .reset_index())
```

Member-months are counted from eligibility rather than summed from claims, because the
eligibility count does not repeat across a member's claim rows. The remaining steps are
standard aggregation. Choose the grain to match the question: add `"service_type"` for a
per-line view, or keep `member_id` for member-level analysis. For single calculations, use
the [free functions](#ratios-and-per-exposure-metrics) directly on any aggregate.

## The `Experience` object

The bound roles are reused by every method, so the expense, revenue, exposure, and date
columns are specified once. Binding `count` (a claim or service count) enables the
frequency-severity views (`frequency_severity`, `decompose_trend`). `filter(...)` and
`with_roles(...)` return new objects without mutating the original.

### `Experience` methods

| Method | Produces |
| --- | --- |
| `by(groupby)` | grouped totals and ratios |
| `views(views)` | a dict of named grouped summaries |
| `rolling(window, groupby)` | rolling-window summaries |
| `trend(prior/current windows, groupby)` | period-over-period trend |
| `components(...)` / `component_summary(...)` | component / driver breakdowns |
| `decompose_trend(...)` | frequency × severity (× mix) split |
| `actual_vs_expected(expected, actual, ...)` | actual-versus-expected with variances |
| `claimants(...)` / `top_claimants(...)` / `claimant_concentration(...)` | large-claimant and concentration views |
| `pool_claimants(claimant_col, pooling_point)` | pooled vs. excess by claimant |
| `cohort(...)` / `duration(...)` | cohort and duration summaries |
| `by_band(value_col, bands)` | banded summaries |
| `with_status(...)` / `by_status(...)` | lifecycle status assignment and summary |
| `margin(...)` | underwriting margins |
| `credibility_weighted(groupby, z, metric)` | credibility-blended estimates by group |
| `deseasonalize(factors)` / `complete(factors, valuation_date)` | new Experience with seasonality removed / claims developed to ultimate |
| `adjust(factors, on, by, how, audit_col)` | new Experience with a column restated by a joined factor |
| `filter(...)` / `with_roles(...)` | derive a new Experience |

## Ratios and per-exposure metrics

All of these accept scalars, NumPy arrays, or pandas Series, and divide safely (returning
NaN rather than raising on a zero denominator):

| Function | Definition |
| --- | --- |
| `loss_ratio(losses, revenue)` | losses ÷ revenue |
| `expense_ratio(expenses, revenue)` | expenses ÷ revenue |
| `combined_ratio(losses, expenses, revenue)` | (losses + expenses) ÷ revenue |
| `pure_premium(losses, exposure)` | losses ÷ exposure |
| `frequency(claim_count, exposure)` | claims ÷ exposure |
| `severity(losses, claim_count)` | losses ÷ claim count |
| `per_exposure(amount, exposure)` | generic per-exposure rate |
| `permissible_loss_ratio(expense_ratio, profit_provision)` | 1 − expense ratio − profit |
| `required_revenue(expense, target_ratio)` | expense ÷ target ratio |
| `indicated_change(required, current)` | required ÷ current − 1 |
| `actual_to_expected(actual, expected)` | actual ÷ expected |

## Reserving

Build a development triangle from transactional claims, fit a chain ladder, and obtain
ultimates and IBNR. Origin and development periods are derived automatically.

```python
from actuarialpy import make_completion_triangle, ChainLadder, completion_factors

triangle = make_completion_triangle(
    claims, origin_col="incurred_month", valuation_col="paid_month",
    amount_col="paid", cumulative=True,
)

cl = ChainLadder.fit(triangle, method="volume", tail=1.0)
projection = cl.project(triangle)
# columns: latest_development, latest, development_factor, ultimate, ibnr

factors = completion_factors(triangle, method="volume", tail=1.0)   # 1 / cumulative DF
```

`ChainLadder.fit` exposes `age_to_age`, `cdf`, `completion_factors`, `tail`, and `method`.
For segment-level reserving, `chain_ladder_by` returns a `{segment: ChainLadder}` mapping,
and `completion_factors_by` returns a table of factors, one row per `(segment,
development_month)`:

```python
from actuarialpy import completion_factors_by

cf_by_lob = completion_factors_by(
    claims, groupby="line_of_business",
    origin_col="incurred_month", valuation_col="paid_month", amount_col="paid",
    on_insufficient="skip",   # "raise", "skip", or "aggregate"
)
```

Applying factors is separate from estimating them. `apply_completion` matches by value (the
frame's index is not used), computes each row's period as `development_months(incurred,
valuation)`, and treats rows past the triangle's last period as fully complete, so only
immature months are adjusted:

```python
from actuarialpy import apply_completion

completed = apply_completion(
    latest_diagonal, factors,
    value_col="claims", date_col="incurred_month", valuation_date="2024-12-31",
)
# completed["claims_completed"] == paid / completion_factor

# per-segment factors: pass the tidy table + by=, joined on group AND development period
completed = apply_completion(
    latest_diagonal, cf_by_lob, by="line_of_business",
    value_col="claims", date_col="incurred_month", valuation_date="2024-12-31",
)
```

An absent `(group, period)` returns `NaN`; a duplicated key in the factor table raises
rather than producing a many-to-many join. Methods that blend emerged-to-date experience
with an a priori are available through `develop_ultimate(..., method=...)`: `"chain_ladder"`,
`"bornhuetter_ferguson"` (with `apriori_col`), `"benktander"`, and `"cape_cod"` (a priori
derived from the data, taking `exposure_col`). All accept `by=`. The method is applied as
specified; the a priori and exposure base are caller-supplied. On the `Experience` object,
`complete(factors, valuation_date=...)` returns a new Experience with the expense column
developed to ultimate. Run completion before deseasonalizing and trending.

## Trend and forecasting

```python
from actuarialpy import trend_factor, annualized_trend, project_forward

trend_factor(0.06, months=18)             # (1 + 0.06) ** (18/12)
project_forward(1000.0, 0.06, months=18)  # trend a value forward 18 months
annualized_trend(current=1.1, prior=1.0, months_between=12)
```

The functions above apply a trend or measure it between two points. To estimate a trend from
history, `fit_trend` regresses `log(rate)` on time (log-linear OLS) over the full series and
returns the fitted annual trend, goodness of fit, and a confidence interval:

```python
from actuarialpy import fit_trend

fit = fit_trend(history, value_col="claims", date_col="month", exposure_col="member_months")
fit.annual_trend          # e.g. 0.072  (exp(slope) - 1)
fit.r_squared, fit.ci     # goodness of fit, confidence interval
fit.factor(18)            # (1 + annual_trend) ** (18/12)
```

It fits on the rate (`claims / member_months`) when an exposure is given, otherwise on the
value. Time is measured from dates, so missing periods are handled correctly. Fit on
completed, deseasonalized history (`complete` → `deseasonalize` → `fit_trend`) so runout and
seasonality do not bias the slope. Rate-based projection is in the forecast module:
`forecast_experience(...)` applies a trended per-exposure rate to projected exposure.

## Frequency-severity and per-exposure decomposition

Split a per-exposure loss into frequency and severity, and decompose the change between two
periods into frequency and severity effects. It requires a claim or service count
alongside losses and exposure:

```python
from actuarialpy import frequency_severity_summary, decompose_per_exposure_trend

panel = frequency_severity_summary(
    df, count_col="claim_count", loss_col="claims", exposure_col="member_months",
    groupby="plan",
)
# columns: plan, member_months, claim_count, claims, frequency, severity, loss_per_exposure
# loss_per_exposure == frequency * severity for every row

trend = decompose_per_exposure_trend(
    prior_year, current_year,
    count_col="claim_count", loss_col="claims", exposure_col="member_months",
    on="plan",            # optional; omit for a single total row
)
# loss_per_exposure_trend == frequency_trend * severity_trend (exact); loss_per_exposure_change == frequency_effect + severity_effect
```

Pass `mix_by` to split the per-exposure trend into frequency, severity, and mix when the book is a blend of
cells whose composition changes between periods; otherwise a book-wide frequency or severity
trend absorbs exposure-mix shifts. The split uses LMDI (logarithmic mean Divisia index),
which is order-independent and leaves no residual. Every cell must have positive count, loss,
and exposure in both periods. On an `Experience`, the same split is `exp.decompose_trend(...)`,
using the bound `count`, `expense`, and `exposure` roles. See
[`examples/trend_decomposition.py`](examples/trend_decomposition.py).

## Seasonality and working days

Two effects make months non-comparable: differing working-day counts, and the time of year.
Two separate tools address them.

`seasonality_factors` estimates one multiplier per calendar period from several years of
history (ratio-to-moving-average decomposition, normalized to average 1.0). Pass
`exposure_col` to fit on a rate (per exposure), which removes exposure growth so only the
time-of-year pattern remains:

```python
from actuarialpy import seasonality_factors, deseasonalize, apply_seasonality

factors = seasonality_factors(history, date_col="month", value_col="claims",
                              exposure_col="member_months")
deseasonalize(recent, factors, date_col="month", value_col="claims")   # pattern divided out
apply_seasonality(annual_plan, factors, date_col="month", value_col="budget")  # multiplied back
```

`business_days_in_period` counts weekdays minus holidays (US federal by default) for the
working-day effect; when using both, normalize by working days first, then fit factors on the
normalized series. On the `Experience` object, `deseasonalize(factors)` divides the pattern
out of the expense column and returns a new `Experience`, so downstream views use the
deseasonalized series. `seasonality_factors_by` fits per segment and returns a table indexed
by `(segment, season)`; pass it with `by=` to `deseasonalize`. A rolling-12 or full-year
comparison already cancels seasonality; this is mainly needed when fitting trend on monthly
data.

## Adjustments and restatement

Experience rating applies a chain of factors to a base amount: completion, trend, benefit
relativity, area, demographic loads, network discounts. `adjust` joins a factor to each row
by a key and multiplies (or divides). `complete` and `deseasonalize` derive the key from a
date; `adjust` keys on a column. All use the same validated join (unique-key check, NaN on
missing keys, index-independent).

```python
from actuarialpy import adjust

adjust(experience, 1.072, value_col="claims")                    # a scalar trend factor
area = pd.Series({"urban": 1.08, "suburban": 1.00, "rural": 0.94})
adjust(experience, area, value_col="claims", on="region")        # a Series keyed by a column
adjust(experience, benefit_relativity, value_col="claims",       # a tidy per-segment table
       on="plan", by="line_of_business")
```

`how="multiply"` (default) applies the factor; `how="divide"` removes it. An absent key
returns `NaN`; pass `default=1.0` to treat a missing key as no adjustment. On the
`Experience` object, `adjust` returns a new Experience with the expense column restated, and
`audit_col` records the cumulative restatement multiplier across the chain:

```python
restated = (
    exp.complete(cf_by_lob, valuation_date="2024-12-31", by="line_of_business")
       .adjust(1.072, audit_col="restatement")                                     # trend
       .adjust(benefit_relativity, on="plan", by="line_of_business", audit_col="restatement")
       .adjust(area, on="region", audit_col="restatement")
)
restated.by()   # grouped loss ratios on the fully restated claims
```

## Actual versus forecast

Compare actuals against a forecast supplied as two separate tables, joined on shared keys:

```python
from actuarialpy.forecast import compare_actual_to_expected

variance = compare_actual_to_expected(
    actuals_table, forecast_table,
    on=["month", "segment", "product", "category"],
    actual_col="amount", expected_col="amount",
    how="outer",                         # keep forecast months without actuals yet
    suffixes=("actual", "forecast"),     # -> amount_actual, amount_forecast
)
# columns: <keys...>, amount_actual, amount_forecast, variance, variance_pct, actual_to_expected
```

With `how="outer"`, a forecast month with no actual is kept with the missing side `NaN`, so
an unavailable actual is distinguishable from zero. When `category` mixes units (dollars and
counts), keep it in the keys or filter to one category before totalling.

## Credibility

Greatest-accuracy (Bühlmann) credibility, fit from per-risk observations or constructed from
known structural parameters:

```python
from actuarialpy import Buhlmann, BuhlmannStraub, credibility_weighted_estimate

observations = [[10, 12, 9, 11], [20, 18, 22, 19], [5, 6, 4, 7]]   # rows are risks
model = Buhlmann.fit(observations)
model.z, model.k                      # credibility factor and constant
model.premium(risk_mean=11.0)         # credibility-weighted premium

ws = BuhlmannStraub.fit(observations, weights=[[1,1,1,1],[2,2,2,2],[1,1,1,1]])  # unequal exposures
credibility_weighted_estimate(observed=0.82, complement=0.75, z=0.6)            # blend directly
```

For limited-fluctuation (classical) credibility — the square-root rule against a
full-credibility standard — use `limited_fluctuation_z`:

```python
from actuarialpy import limited_fluctuation_z, full_credibility_claims, credibility_weighted_estimate

groups["z"] = limited_fluctuation_z(groups["claim_count"], full_credibility_standard=1082)
groups["blended_lr"] = credibility_weighted_estimate(groups["experience_lr"], manual_lr, groups["z"])

n_full = full_credibility_claims(confidence=0.90, tolerance=0.05)   # ~1082 claims, from first principles
```

`Buhlmann(overall_mean, epv, vhm, n_obs)` and `BuhlmannStraub(...)` can also be constructed
directly when EPV and VHM are already known.

## Lifecycle, pooling, banding, margins

- **Lifecycle** (`lifecycle`): `is_in_force(...)`, `earned_exposure(...)`,
  `add_months_in_force(...)`, `add_tenure(...)`, and `derive_status(...)` (labels rows
  active / first-year / termed).
- **Pooling** (`pooling`): `pool_losses(df, loss_col, pooling_point)` splits each loss into
  pooled and excess portions; `excess_over_threshold(...)` returns the excess layer.
  `retained_cv(outcomes, retention, n_units)` returns the coefficient of variation of the
  retained aggregate of `n_units` independent units capped at `retention`, and
  `retention_for_target_cv(outcomes, n_units, target_cv)` inverts it to the retention at
  which that CV meets a target (the basis for a size-graded retention).
- **Banding** (`banding`): `assign_band(df, value_col, bands)` and `summarize_by_band(...)`.
- **Margins** (`margins`): `add_margin(...)` / `margin(...)` / `margin_ratio(...)`.
- **Contribution** (`contribution`): `share_of_total(...)`, `contribution_to_change(...)`,
  `top_contributors(...)`.

## Underwriting summary and weighted rollups

The two-tier underwriting income statement — **gross margin** (revenue less
loss expense, operating expense excluded, which is also why operating expense
never enters a loss ratio) and **gain/(loss)** (gross margin less operating
expense). Ratio denominators are explicit parameters because real exhibits mix
them (a loss ratio over net revenue beside an expense ratio over gross
premium); `reconciliation()` reports the resulting gap in
`gain% = 1 − combined ratio` so the convention is visible instead of silent.
Domain naming flows through `profile` / `labels` on the output views (a health
shop's `mlr`), never the calculation. These are management/pricing metrics;
regulated ratio calculations (e.g. a statutory rebate loss ratio) are out of
scope.

```python
from actuarialpy import UnderwritingSummary, underwriting_summary

uw = UnderwritingSummary.from_per_exposure(
    revenue_per_exposure={"premium": 400.0, "refund": -1.4},
    loss_per_exposure={"claims": 340.0, "other_loss": 16.4},
    expense_per_exposure=37.4,
    exposure=300_000,
)
uw.loss_ratio, uw.expense_ratio, uw.combined_ratio, uw.gain_per_exposure
uw.to_frame(profile="health")   # loss_ratio -> mlr; math unchanged

# grouped: components summed first, every ratio a ratio of sums
underwriting_summary(df, groupby="cohort",
                     revenue_cols=["premium", "refund"], loss_cols=claim_cols,
                     expense_cols="expense", exposure_col="member_months",
                     premium_col="premium")
```

Quantities that are already rates at the row level (rate actions, persistency)
cannot be summed; `weighted_mean` / `weighted_summary` average them with a
**required, named weight** and report the weight total beside every average:

```python
from actuarialpy import weighted_summary

weighted_summary(book, value_cols="rate_action", weight_col="premium",
                 groupby="cohort")
```

## Reporting

Write a set of named analysis views to a multi-sheet Excel workbook:

```python
from actuarialpy.reporting import to_excel_report

views = exp.views({"overall": None, "by_group": "group_id"})
to_excel_report(views, "experience_report.xlsx")
```

## Testing

```bash
pytest -q
```

## License

MIT License
