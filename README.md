# ActuarialPy

ActuarialPy is an experience-centered Python toolkit for actuarial analysis. It provides a lightweight `Experience` object for working with claims, losses, benefits, revenue, premium, exposure, and time-based experience data.

The core workflow is to define the actuarial roles of a dataset once, then use the object to produce common analyses such as experience summaries, rolling views, period-over-period trends, component drivers, actual-versus-expected summaries, claimant concentration reviews, and cohort/duration summaries.

## Installation

For local development:

```bash
pip install -e .
```

## Basic usage

```python
import actuarialpy as ap

exp = ap.Experience(
    claims,
    expense="total_expense",
    revenue="premium",
    exposure="member_months",
    date="incurred_date",
    profile="health",
)
```

Once the experience object is created, the same column roles are reused across analyses.

```python
summary = exp.by(["group_id", "product_code"])
rolling = exp.rolling(window=12, groupby="product_code")
trend = exp.trend(
    prior_start="2025-01-01",
    prior_end="2025-12-31",
    current_start="2026-01-01",
    current_end="2026-12-31",
    groupby="product_code",
)
```

## Core capabilities

ActuarialPy currently supports:

- grouped experience summaries
- loss ratio / MLR-style ratios
- PMPM, PSPM, PEPM, and generic per-exposure metrics
- completion factor and IBNR calculations
- rolling-window experience summaries
- period-over-period trend comparisons
- component-level driver analysis
- actual-versus-expected summaries
- claimant/member concentration review
- cohort and duration summaries
- validation utilities for common data issues

## Package structure

```text
actuarialpy/
├── frame.py        # Experience facade
├── metrics.py      # Ratios, per-exposure metrics, frequency, severity, A/E
├── experience.py   # Grouped experience summaries
├── completion.py   # Completion factor and IBNR calculations
├── rolling.py      # Rolling-window summaries
├── trend.py        # Period and date-range trend summaries
├── components.py   # Component summaries and driver analysis
├── expected.py     # Actual-versus-expected summaries
├── claimants.py    # Claimant and concentration summaries
├── cohorts.py      # Cohort and duration summaries
├── periods.py      # Period and duration helpers
└── columns.py      # Validation and column helpers
```

## Experience summaries

```python
summary = exp.by(["group_id", "product_code"])
```

Typical output includes:

```text
group_id
product_code
total_expense
total_revenue
member_months
mlr
expense_pmpm
revenue_pmpm
```

You can also create multiple views from the same experience object:

```python
views = exp.views({
    "overall": None,
    "by_group": "group_id",
    "by_product": "product_code",
    "by_group_product": ["group_id", "product_code"],
})
```

## Rolling summaries

```python
rolling = exp.rolling(
    window=12,
    groupby="product_code",
)
```

Rolling summaries include `period_start` and `period_end`. Incomplete windows are omitted by default.

## Trend summaries

Trend comparisons can be based on date ranges:

```python
trend = exp.trend(
    prior_start="2025-01-01",
    prior_end="2025-12-31",
    current_start="2026-01-01",
    current_end="2026-12-31",
    groupby="product_code",
)
```

They can also be based on a period column:

```python
claims["year"] = claims["incurred_date"].dt.year

trend = exp.trend(
    period_col="year",
    prior_period=2025,
    current_period=2026,
    groupby="product_code",
)
```

## Component driver analysis

Component driver analysis explains which categories drove the change in total experience.

```python
drivers = exp.components(
    component_cols=[
        "inpatient_claims",
        "outpatient_claims",
        "professional_claims",
        "pharmacy_claims",
        "pharmacy_rebates",
        "non_ffs_expenses",
    ],
    prior_start="2025-01-01",
    prior_end="2025-12-31",
    current_start="2026-01-01",
    current_end="2026-12-31",
    groupby="product_code",
)
```

## Actual versus expected

```python
ae = exp.actual_vs_expected(
    expected="expected_expense",
    groupby="product_code",
)
```

This produces aggregated actual, expected, actual-to-expected, variance, and variance percentage fields.

## Claimant review

Claimant-level summaries are descriptive and do not apply pooling, capping, or stop-loss adjustments.

```python
claimants = exp.claimants(
    claimant_col="member_id",
    groupby="group_id",
)

top = exp.top_claimants(
    claimant_col="member_id",
    groupby="group_id",
    n=25,
)

concentration = exp.claimant_concentration(
    claimant_col="member_id",
    groupby="group_id",
)
```

## Cohort and duration summaries

```python
cohort = exp.cohort(
    entity_col="group_id",
    start_date_col="group_effective_date",
    duration_months=12,
    groupby="product_code",
)

duration = exp.duration(
    entity_col="group_id",
    start_date_col="group_effective_date",
    max_duration_month=24,
)
```

## Functional API

The `Experience` object is the recommended workflow interface, but the underlying functions are also available directly:

```python
from actuarialpy.experience import summarize_experience
from actuarialpy.trend import trend_summary
from actuarialpy.components import component_driver_analysis
from actuarialpy.expected import summarize_actual_vs_expected
from actuarialpy.claimants import summarize_claimants, top_claimants, claim_concentration
```

## Development status

ActuarialPy is in early development. The current focus is on reliable experience-analysis workflows before expanding into more complex forecasting, seasonality, credibility, or reserving methods.
