# ActuarialPy

ActuarialPy is a Python toolkit for actuarial experience analysis. It provides reusable functions for common actuarial calculations, grouped experience summaries, completion factor application, rolling experience views, trend comparisons, and component-level driver analysis.

The package is designed to sit on top of pandas. It does not attempt to replace pandas or wrap ordinary DataFrame operations. Instead, it focuses on calculations and workflows where actuarial meaning matters, such as aggregating before calculating ratios, completing claims with valuation factors, calculating PMPM metrics, and comparing experience across periods.

## Installation

For local development:

```bash
pip install -e .
```

## Core capabilities

ActuarialPy currently supports:

- Basic actuarial metrics such as loss ratio, PMPM, PSPM, frequency, severity, pure premium, and actual-to-expected.
- Claim completion calculations using paid completion factors.
- IBNR calculation from paid and completed claims.
- Grouped experience summaries by fields such as group, product, line of business, and incurred period.
- Rolling-window summaries, including rolling 12-month MLR and PMPM views.
- Trend summaries comparing two periods.
- Component-level driver analysis for categories such as inpatient, outpatient, professional, pharmacy, rebates, and non-fee-for-service expenses.
- Validation utilities for common actuarial data issues.

## Package structure

```text
actuarialpy/
├── metrics.py       # Core ratios, exposure-normalized metrics, and actuarial primitives
├── completion.py    # Completion factor and IBNR calculations
├── experience.py    # Grouped experience summaries
├── rolling.py       # Rolling-window experience summaries
├── trend.py         # Period-over-period trend summaries
├── components.py    # Component summaries and driver analysis
├── periods.py       # Period and duration helpers
├── compare.py       # Variance and change calculations
├── validation.py    # Validation utilities
└── reporting.py     # Basic Excel report export
```

## Basic metrics

```python
from actuarialpy.metrics import loss_ratio, pmpm, actual_to_expected

loss_ratio(850_000, 1_000_000)
# 0.85

pmpm(1_000_000, 2_000)
# 500.0

actual_to_expected(1_100_000, 1_000_000)
# 1.10
```

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

Rolling summaries are useful for reviewing longer-term experience patterns, such as rolling 12-month MLR.

```python
from actuarialpy.rolling import rolling_summary

rolling = rolling_summary(
    claims,
    date_col="incurred_date",
    window=12,
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

## Development status

ActuarialPy is in early development. The current focus is on reliable core experience-analysis workflows before adding more complex features such as forecasting, seasonality, credibility, or advanced reserving methods.