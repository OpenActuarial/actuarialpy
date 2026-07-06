# actuarialpy

[![CI](https://github.com/OpenActuarial/actuarialpy/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/actuarialpy/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/actuarialpy)](https://pypi.org/project/actuarialpy/)

Shared actuarial primitives and general tooling on claims, exposure, and premium data:
loss ratios and per-exposure rates, development triangles and IBNR, credibility, trend,
seasonality, financial mathematics, exposure and lifecycle bases, size banding, pooling,
margins, weighted rollups, and the underwriting income statement. The only dependencies
are `numpy` and `pandas`, and every result is a DataFrame or Series.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Ratios and per-exposure metrics](#ratios-and-per-exposure-metrics)
- [Reserving](#reserving)
- [Trend](#trend)
- [Seasonality and working days](#seasonality-and-working-days)
- [Adjustments and restatement](#adjustments-and-restatement)
- [Comparison and contribution](#comparison-and-contribution)
- [Credibility](#credibility)
- [Financial mathematics (time value of money)](#financial-mathematics-time-value-of-money)
- [Lifecycle, pooling, banding, margins](#lifecycle-pooling-banding-margins)
- [Underwriting summary and weighted rollups](#underwriting-summary-and-weighted-rollups)
- [Reporting](#reporting)

## Overview

**`actuarialpy`** is a calculation library of actuarial primitives: loss ratios and
per-exposure rates, chain-ladder development and IBNR, credibility, trend, seasonal
factors, financial mathematics, exposure and lifecycle bases, size banding, pooling, and
the underwriting income statement, applied to claims, exposure, and premium data. It does
not perform data preparation or encode filed methodology: the caller supplies the table
and selects the method.

Functions accept scalars, NumPy arrays, or pandas Series — or a DataFrame at any grain —
and return the same type: `loss_ratio`, `per_exposure`, `severity`, `trend_factor`,
`fit_trend`, `seasonality_factors`, `completion_factors`, the credibility models, and
others. They operate on any frame at any grain, so you choose the grain to match the
question.

## Installation

```bash
pip install actuarialpy
```

## Quick start

Pass an aggregate at the grain you are analysing and call the primitive you need:

```python
import actuarialpy as ap

# ratios and per-exposure rates on any aggregate
ap.loss_ratio(1_240_000, 1_500_000)            # 0.827
ap.per_exposure(1_240_000, 12_000)             # 103.33 per exposure unit

# chain-ladder development to ultimate + IBNR from a triangle
tri = ap.make_completion_triangle(dev, origin_col="origin",
                                  valuation_col="valuation", amount_col="paid")
cf = ap.completion_factors(tri)
completed = ap.apply_completion(latest, cf, value_col="claims",
                                date_col="origin", valuation_date="2024-12-31")

# fit a trend and project it forward
fit = ap.fit_trend(monthly, date_col="month", value_col="loss_ratio")
ap.project_forward(fit, periods=12)
```

Build the aggregate with pandas at the grain that matches the question. Typically this is a
single `groupby` that sums claims, counts exposure from a correctly-grained table (e.g. a
health book's member-months from eligibility, **counted** rather than summed, because the
count does not repeat across a member's claim rows), and joins premium:

```python
g = ["group_id", "month"]
data = (claims.groupby(g)["paid_amount"].sum().rename("claims").to_frame()
        .join(eligibility.groupby(g).size().rename("member_months"))   # counted, not summed
        .join(premium.groupby(g)["premium"].sum().rename("premium"))
        .reset_index())
```

Choose the grain to match the question: add `"service_type"` for a per-line view, or keep
`member_id` for member-level analysis. For single calculations, call the free functions
directly on any aggregate.

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
specified; the a priori and exposure base are caller-supplied. Run completion before
deseasonalizing and trending.

## Trend

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
completed, deseasonalized history so runout and seasonality do not bias the slope.

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
normalized series. `seasonality_factors_by` fits per segment and returns a table indexed by
`(segment, season)`; pass it with `by=` to `deseasonalize`. A rolling-12 or full-year
comparison already cancels seasonality; this is mainly needed when fitting trend on monthly
data.

## Adjustments and restatement

Experience rating applies a chain of factors to a base amount: completion, trend, benefit
relativity, area, demographic loads, network discounts. `adjust` joins a factor to each row
by a key and multiplies (or divides). The `apply_completion` and `deseasonalize` primitives
derive the key from a date; `adjust` keys on a column. All use the same validated join
(unique-key check, NaN on missing keys, index-independent).

```python
from actuarialpy import adjust

adjust(experience, 1.072, value_col="claims")                    # a scalar trend factor
area = pd.Series({"urban": 1.08, "suburban": 1.00, "rural": 0.94})
adjust(experience, area, value_col="claims", on="region")        # a Series keyed by a column
adjust(experience, benefit_relativity, value_col="claims",       # a tidy per-segment table
       on="plan", by="line_of_business")
```

`how="multiply"` (default) applies the factor; `how="divide"` removes it. An absent key
returns `NaN`; pass `default=1.0` to treat a missing key as no adjustment. `audit_col`
accumulates the net restatement multiplier across a chain of `adjust` calls, so the total
effect of trend and relativity loads is auditable on one column:

```python
step1 = adjust(experience, 1.072, value_col="claims", audit_col="restatement")   # trend
step2 = adjust(step1, benefit_relativity, value_col="claims",
               on="plan", by="line_of_business", audit_col="restatement")
step3 = adjust(step2, area, value_col="claims", on="region", audit_col="restatement")
# step3["restatement"] == 1.072 * benefit_relativity * area   (per row)
```

## Comparison and contribution

Small primitives for period-over-period movement and share-of-total attribution, on
scalars, arrays, or Series:

```python
from actuarialpy import (
    absolute_change, percent_change, basis_point_change, variance, variance_pct,
    share_of_total, contribution_to_change, top_contributors,
)

percent_change(prior=1.00, current=1.08)          # 0.08
basis_point_change(prior=0.822, current=0.831)    # 90.0 (bps)
variance(actual=1_050_000, expected=1_000_000)    # 50_000  (actual - expected)

share_of_total(df["claims"])                       # each row's share of the column total
contribution_to_change(prior_df, current_df, value_col="claims", on="plan")
top_contributors(df, value_col="claims", on="plan", n=5)
```

`variance`/`variance_pct` and `actual_to_expected` express the same actual-versus-expected
gap in different units — use them when you already have an actual and an expected column to
compare.

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

## Financial mathematics (time value of money)

The interest-theory primitives every reserve, premium, and valuation depends on:
interest-rate fundamentals and their conversions, present and accumulated values,
annuities-certain, cash-flow analysis (NPV/IRR), loan amortization, discounting against a
spot curve, and day-count year fractions. With `i` the effective annual rate,
`v = 1/(1+i)` is the discount factor, `d = i/(1+i)` the effective rate of discount, and
`δ = ln(1+i)` the force of interest.

The element-wise functions follow the same type contract as the [ratio and per-exposure
metrics](#ratios-and-per-exposure-metrics): a scalar rate returns a `float`, and a NumPy
array or pandas Series returns the same kind, with the index and name preserved — so a
per-scenario or per-period rate column maps straight to a result column you can assign back:

```python
import actuarialpy as ap
import pandas as pd

ap.discount_factor(0.05, 10)        # 0.6139…  — scalar in, scalar out
ap.annuity_immediate(0.05, 20)      # 12.4622… — a-angle-n at 5% for 20 years

# a rate curve (or scenario set) → a column of factors, index preserved
scenarios = pd.DataFrame({"rate": [0.03, 0.05, 0.07]})
scenarios["discount_10y"] = ap.discount_factor(scenarios["rate"], 10)
scenarios["annuity_20y"]  = ap.annuity_immediate(scenarios["rate"], 20)
```

Rate fundamentals and conversions — `discount_factor(i, t)`, `accumulation_factor(i, t)`,
`effective_discount(i)`, `force_of_interest(i)` and its inverse `rate_from_force(delta)`, and
the nominal conversions `nominal_interest(i, m)` / `nominal_discount(i, m)` with their inverses
`rate_from_nominal_interest(nominal, m)` / `rate_from_nominal_discount(nominal, m)` — all take
a scalar, array, or Series rate:

```python
from actuarialpy import force_of_interest, nominal_interest, rate_from_nominal_interest

force_of_interest(0.05)                             # δ = ln(1.05)
i_m = nominal_interest(0.05, 12)                    # i^(12) convertible monthly
rate_from_nominal_interest(i_m, 12)                 # back to the effective 0.05
```

Annuities-certain — present and accumulated values, immediate and due, plus the continuous,
`m`-thly, increasing, decreasing, geometric, and deferred variants. Each is a closed form, so
each vectorizes over the rate, and the `i = 0` limit is handled element-wise:

```python
from actuarialpy import (
    annuity_immediate, annuity_due, accumulated_immediate,
    annuity_continuous, increasing_annuity_immediate, geometric_annuity_immediate,
    perpetuity_immediate,
)

annuity_due(0.05, 20)                         # ä-angle-n
accumulated_immediate(0.05, 20)               # s-angle-n
geometric_annuity_immediate(0.05, 20, 0.02)   # payments growing 2% a year
perpetuity_immediate(0.05)                    # 1/i
```

Cash-flow analysis and loans. `net_present_value` and `internal_rate_of_return` take a
sequence of cash flows (at times `0, 1, 2, …` by default, or explicit `times`) and return a
scalar; `level_payment`, `outstanding_balance`, and `amortization_schedule` handle loan
amortization:

```python
from actuarialpy import net_present_value, internal_rate_of_return, amortization_schedule

net_present_value(0.06, [-1000, 300, 400, 500])   # discounted at 6%
internal_rate_of_return([-1000, 300, 400, 500])   # the rate making NPV = 0

sched = amortization_schedule(principal=100_000, i=0.05, n=30)
# columns: period, payment, interest, principal, balance
```

Discounting against a spot curve and day-count fractions. `discount_factors(spot_rates, times)`
returns the vector of `(1+s_t)^{-t}`, `present_value_curve(cashflows, spot_rates, times)` prices a
stream on that curve, and `year_fraction(start, end, convention)` computes day-count fractions
(`"actual/365"`, `"actual/360"`, `"30/360"`, `"actual/actual"`):

```python
from actuarialpy import discount_factors, present_value_curve, year_fraction

spot = [0.03, 0.035, 0.04]
present_value_curve([100, 100, 1100], spot_rates=spot, times=[1, 2, 3])
year_fraction("2024-01-15", "2024-07-15", convention="30/360")
```

The cash-flow functions (`net_present_value`, `internal_rate_of_return`, `present_value_curve`)
and `discount_factors` are reductions or curve operations over a whole stream — they take the
sequence and return a scalar or the matching factor vector. Everything else is element-wise over
the rate and mirrors input type as shown above.

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
- **Banding**: `assign_band(df, value_col, bands)` buckets rows into ordered size bands
  (a trailing `float("inf")` captures the open top band; the result is an ordered
  categorical, so downstream group-bys keep band order).
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

Write a set of named analysis views to a multi-sheet Excel workbook (one sheet per key).
The values are plain DataFrames, so anything you compute — grouped summaries, triangles,
trend tables — can be a sheet:

```python
from actuarialpy import to_excel_report

views = {
    "overall": overall_summary,      # a DataFrame per sheet
    "by_group": by_group_summary,
}
to_excel_report(views, "report.xlsx")   # requires the `excel` extra (openpyxl)
```

## Testing

```bash
pytest -q
```

## License

MIT License