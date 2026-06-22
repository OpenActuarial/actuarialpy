# ActuarialPy

ActuarialPy is a Python toolkit for actuarial experience analysis. It provides reusable functions for common actuarial calculations, grouped experience summaries, completion (loss development) factor application, rolling experience views, trend comparisons, component-level driver analysis, lifecycle/in-force handling, size banding, concentration, margins, and large-loss pooling.

The package is general across lines of business — life, health, and pension under the SOA, and property/casualty under the CAS. Terminology defaults to the line-agnostic vocabulary (loss ratio, exposure, premium, losses); health is supported as a first-class application through the `health` profile (which produces familiar names such as `mlr` and per-member-per-month metrics). It is designed to sit on top of pandas rather than replace it, focusing on calculations where actuarial meaning matters, such as aggregating before calculating ratios, completing claims with development factors, normalizing by exposure, and comparing experience across periods.

## Installation

For local development:

```bash
pip install -e .
```

## Core capabilities

ActuarialPy currently supports:

- Basic actuarial metrics such as loss ratio, per-exposure metrics (PMPM/PSPM/PEPM as health conveniences), frequency, severity, pure premium, combined ratio, actual-to-expected, and the permissible (target/zero-margin) loss ratio.
- Claim completion (loss development) calculations using paid completion factors, IBNR, and a `completed_experience` one-call completion-to-experience summary.
- Grouped experience summaries by fields such as group, product, line of business, and incurred period (ratio column defaults to `loss_ratio`).
- An optional `Experience` facade that binds the expense/revenue/exposure/profile roles once, so each view (`.by`, `.rolling`, `.trend`, `.by_band`, `.by_status`, `.cohort`, `.duration`) is a single short call instead of repeating the column arguments.
- Rolling-window summaries, including rolling 12-period loss ratio and per-exposure views.
- Trend summaries comparing two periods.
- Component-level driver analysis for arbitrary cost categories.
- Credibility weighting, including Bühlmann and Bühlmann-Straub models and a credibility-weighting primitive.
- Lifecycle handling: derive active / first-year / termed status and tenure from effective and termination dates, in-force flags, and exposure clipping / earned exposure.
- Size banding by any numeric measure, with per-band experience summaries.
- Concentration analysis (top-N cumulative and share) at group or member/claimant grain.
- Underwriting margin (premium net of losses and loadings) and per-exposure margin.
- Large-loss pooling: flagging, capping at a pooling point (pooled/excess split), large-loss summaries, and the excess-over-threshold hand-off to the modeling satellites.
- Validation utilities for common actuarial data issues.

## Package structure

```text
src/actuarialpy/
├── metrics.py       # Core ratios, exposure-normalized metrics, and actuarial primitives
├── completion.py    # Completion factors, completed claims, IBNR, and claim triangles
├── experience.py    # Grouped experience summaries and views
├── frame.py         # Experience facade: bind column roles once, delegate to the functions above
├── rolling.py       # Calendar-aware rolling-window experience summaries
├── trend.py         # Trend factors, projection, and period-over-period summaries
├── components.py    # Component summaries and driver analysis
├── contribution.py  # Share-of-total and contribution-to-change primitives
├── credibility.py   # Bühlmann / Bühlmann-Straub models and the weighting primitive
├── cohorts.py       # Cohort and duration summaries
├── lifecycle.py     # Status (active/first-year/termed), tenure, in-force, earned exposure
├── banding.py       # Configurable size bands and per-band summaries
├── concentration.py # Top-N cumulative and share (group or member/claimant grain)
├── margins.py       # Underwriting margin and per-exposure margin
├── pooling.py       # Large-loss flagging, pooling/capping, excess-over-threshold hand-off
├── forecast.py      # Rate-based forecasting and actual-to-expected comparison
├── periods.py       # Period and duration helpers
├── compare.py       # Variance and change calculations
├── columns.py       # Column validation and small DataFrame helpers
├── profiles.py      # Light-touch domain profile defaults (e.g. health -> mlr)
└── reporting.py     # Basic Excel report export
```

Most workflow functions are also importable directly from the top level, e.g.
`from actuarialpy import summarize_experience, rolling_summary, trend_summary`.

## Basic metrics

```python
from actuarialpy import loss_ratio, pmpm, actual_to_expected

loss_ratio(850_000, 1_000_000)
# 0.85

pmpm(1_000_000, 2_000)
# 500.0

actual_to_expected(1_100_000, 1_000_000)
# 1.10
```

The metric primitives are vectorized and type-stable: scalar inputs return a
native `float`, and `pandas.Series` inputs return a `Series` with the original
index preserved (Series are combined positionally, so they are assumed aligned).

## Completion factors and IBNR

ActuarialPy assumes completion factors are supplied by the user. Joins between claims and factor tables should generally be handled directly with pandas.

```python
claims = claims.merge(
    factors,
    on=["line_of_business", "incurred_date"],
    how="left",
    validate="many_to_one",
)
```

After the factors are joined, ActuarialPy can calculate completed claims and IBNR.

```python
from actuarialpy.completion import complete_claim_components

claims = complete_claim_components(
    claims,
    component_factor_map={
        "inpatient_claims": "inpatient_completion_factor",
        "outpatient_claims": "outpatient_completion_factor",
        "professional_claims": "professional_completion_factor",
        "pharmacy_claims": "pharmacy_completion_factor",
    },
)
```

This adds completed and IBNR columns for each component, such as:

```text
inpatient_claims_completed
inpatient_claims_ibnr
outpatient_claims_completed
outpatient_claims_ibnr
```

## Experience summaries

For member-level monthly data, create a true exposure field before summarizing.

```python
claims["member_months"] = 1
```

Create a total expense column using pandas. For example:

```python
claims["total_expense"] = claims[
    [
        "inpatient_claims_completed",
        "outpatient_claims_completed",
        "professional_claims_completed",
        "pharmacy_claims_completed",
        "pharmacy_rebates",
        "non_ffs_expenses",
    ]
].sum(axis=1)
```

Then summarize the experience.

```python
from actuarialpy.experience import summarize_experience

summary = summarize_experience(
    claims,
    groupby=["group_id", "product_code"],
    expense_cols=["total_expense"],
    revenue_cols=["premium"],
    exposure_cols=["member_months"],
    ratio_name="mlr",
)
```

The default output uses generic field names:

```text
total_expense
total_revenue
mlr
expense_pmpm
revenue_pmpm
```

ActuarialPy intentionally does not automatically rename `total_expense` to `total_claims` or `total_revenue` to `total_premium`, since the numerator and denominator may include items beyond claims or premium.

## Rolling summaries

Rolling summaries are useful for reviewing longer-term experience patterns, such as rolling 12-month MLR. The window is measured in calendar periods: each group is reindexed onto a dense, gap-free grid before rolling, so a missing month does not silently stretch the window across extra calendar time. Provide `freq` (a pandas offset alias such as `"MS"`) when the data has gaps or is not regularly spaced.

```python
from actuarialpy import rolling_summary

rolling = rolling_summary(
    claims,
    date_col="incurred_date",
    window=12,
    freq="MS",
    expense_cols=["total_expense"],
    revenue_cols=["premium"],
    exposure_cols=["member_months"],
)
```

Rolling summaries include:

```text
period_start
period_end
total_expense
total_revenue
member_months
mlr
expense_pmpm
revenue_pmpm
```

Incomplete rolling windows are omitted by default.

## Trend summaries

Trend summaries compare experience between two periods. Comparisons can be based on a period column, such as calendar year.

```python
from actuarialpy.trend import trend_summary

claims["year"] = claims["incurred_date"].dt.year

trend = trend_summary(
    claims,
    period_col="year",
    prior_period=2025,
    current_period=2026,
    groupby=["product_code"],
    amount_col="total_expense",
    exposure_col="member_months",
)
```

This compares PMPM experience between the prior and current periods.

## Component driver analysis

Component driver analysis explains which categories drove a change in total PMPM.

```python
from actuarialpy.components import component_driver_analysis

drivers = component_driver_analysis(
    claims,
    period_col="year",
    prior_period=2025,
    current_period=2026,
    groupby=["product_code"],
    component_cols=[
        "inpatient_claims_completed",
        "outpatient_claims_completed",
        "professional_claims_completed",
        "pharmacy_claims_completed",
        "pharmacy_rebates",
        "non_ffs_expenses",
    ],
    exposure_col="member_months",
)
```

The output shows prior PMPM, current PMPM, PMPM change, component trend, and contribution to the total PMPM change.

## Credibility

ActuarialPy provides the credibility-weighting primitive plus greatest-accuracy
(Bühlmann and Bühlmann-Straub) credibility models. These models were previously
part of `lossmodels` and were moved here, where credibility sits next to the
experience and ratemaking workflows that consume it.

```python
from actuarialpy import Buhlmann, credibility_weighted_estimate

# Fit a Bühlmann model from a (n_risks, n_obs) array of observations.
model = Buhlmann.fit(observations)
model.z          # credibility factor Z = n / (n + K)
model.premium(risk_mean)   # Z * risk_mean + (1 - Z) * overall_mean

# The blend is also available directly, for a Z from any source
# (a model above, a filed credibility formula, or your own calculation).
credibility_weighted_estimate(observed=actual_pmpm, complement=manual_pmpm, z=0.6)
```

`BuhlmannStraub` is the exposure-weighted analogue, with `z(weight)` and
`premium(risk_mean, weight)` taking per-risk exposure weights.

## Development status

ActuarialPy is in early development. The current focus is reliable core
experience-analysis workflows. Planned additions, kept deliberately separate
from this corrections-focused release, include seasonality indices,
enrollment-to-member-month exposure construction, and deterministic
chain-ladder factor development. Stochastic loss modeling, simulation, and
extreme-value/tail work live in the companion packages (`lossmodels`,
`risksim`, and `extremeloss`).