"""actuarialpy: shared actuarial primitives and general tooling.

Calculation building blocks on tidy tables: ratios and per-exposure metrics,
chain-ladder development and IBNR, credibility, trend, seasonality, financial
mathematics (time value of money), exposure and lifecycle bases, size banding,
pooling, margins, weighted rollups, and comparison/contribution helpers. Every
result is a DataFrame or Series, and the only dependencies are ``numpy`` and
``pandas``.
"""

from actuarialpy.metrics import (
    actual_to_expected,
    combined_ratio,
    expense_ratio,
    frequency,
    indicated_change,
    loss_ratio,
    per_exposure,
    permissible_loss_ratio,
    pure_premium,
    ratio,
    required_revenue,
    safe_divide,
    severity,
)
from actuarialpy.reserving import (
    ChainLadder,
    InsufficientDataWarning,
    apply_completion,
    chain_ladder_by,
    completion_factors,
    completion_factors_by,
    develop_ultimate,
    development_months,
    ibnr,
    lag_months,
    make_completion_triangle,
    validate_completion_factors,
)
from actuarialpy.credibility import (
    Buhlmann,
    BuhlmannStraub,
    credibility_weighted_estimate,
    full_credibility_claims,
    limited_fluctuation_z,
)
from actuarialpy.financial import (
    accumulated_due,
    accumulated_immediate,
    accumulation_factor,
    amortization_schedule,
    annuity_continuous,
    annuity_due,
    annuity_immediate,
    annuity_immediate_mthly,
    decreasing_annuity_immediate,
    deferred_annuity_immediate,
    discount_factor,
    discount_factors,
    effective_discount,
    force_of_interest,
    future_value,
    geometric_annuity_immediate,
    increasing_annuity_immediate,
    internal_rate_of_return,
    level_payment,
    net_present_value,
    nominal_discount,
    nominal_interest,
    outstanding_balance,
    perpetuity_due,
    perpetuity_immediate,
    present_value,
    present_value_curve,
    rate_from_force,
    rate_from_nominal_discount,
    rate_from_nominal_interest,
    year_fraction,
)
from actuarialpy.exposure import (
    add_exposure_column,
    age,
    exposure_years,
)
from actuarialpy.lifecycle import (
    STATUS_ACTIVE,
    STATUS_FIRST_YEAR,
    STATUS_TERMED,
    add_months_in_force,
    add_tenure,
    derive_status,
    earned_exposure,
    is_in_force,
)
from actuarialpy.banding import assign_band
from actuarialpy.adjustments import adjust
from actuarialpy.columns import factor_lookup
from actuarialpy.margins import add_margin, margin, margin_ratio
from actuarialpy.weighted import weighted_mean, weighted_summary
from actuarialpy.pooling import (
    excess_over_threshold,
    pool_losses,
    retained_cv,
    retention_for_target_cv,
)
from actuarialpy.trend import (
    TrendFit,
    annualized_trend,
    fit_trend,
    midpoint_trend_factor,
    period_change,
    project_forward,
    trend_factor,
    trend_summary,
)
from actuarialpy.seasonality import (
    add_business_days,
    apply_seasonality,
    business_days_in_period,
    deseasonalize,
    seasonality_factors,
    seasonality_factors_by,
)
from actuarialpy.periods import add_period_column, to_period
from actuarialpy.compare import (
    absolute_change,
    basis_point_change,
    percent_change,
    variance,
    variance_pct,
)
from actuarialpy.contribution import (
    component_contribution,
    contribution_to_change,
    share_of_total,
    top_contributors,
)

__all__ = [
    # ratios and per-exposure metrics
    "actual_to_expected",
    "combined_ratio",
    "expense_ratio",
    "frequency",
    "indicated_change",
    "loss_ratio",
    "per_exposure",
    "permissible_loss_ratio",
    "pure_premium",
    "ratio",
    "required_revenue",
    "safe_divide",
    "severity",
    # reserving / development
    "ChainLadder",
    "InsufficientDataWarning",
    "chain_ladder_by",
    "completion_factors",
    "completion_factors_by",
    "apply_completion",
    "develop_ultimate",
    "ibnr",
    "lag_months",
    "development_months",
    "make_completion_triangle",
    "validate_completion_factors",
    # credibility
    "Buhlmann",
    "BuhlmannStraub",
    "credibility_weighted_estimate",
    "limited_fluctuation_z",
    "full_credibility_claims",
    # lifecycle
    "STATUS_ACTIVE",
    "STATUS_FIRST_YEAR",
    "STATUS_TERMED",
    "add_months_in_force",
    "add_tenure",
    "derive_status",
    "earned_exposure",
    "is_in_force",
    # exposure and age bases
    "age",
    "exposure_years",
    "add_exposure_column",
    # banding (primitive)
    "assign_band",
    # adjustments / restatement
    "adjust",
    "factor_lookup",
    # margins
    "add_margin",
    "margin",
    "margin_ratio",
    # explicit-weight aggregation
    "weighted_mean",
    "weighted_summary",
    # large-loss pooling
    "excess_over_threshold",
    "pool_losses",
    "retained_cv",
    "retention_for_target_cv",
    # trend
    "annualized_trend",
    "midpoint_trend_factor",
    "period_change",
    "project_forward",
    "fit_trend",
    "TrendFit",
    "trend_factor",
    "trend_summary",
    # seasonality and working-day adjustment
    "business_days_in_period",
    "add_business_days",
    "seasonality_factors",
    "seasonality_factors_by",
    "deseasonalize",
    "apply_seasonality",
    # date / period helpers
    "to_period",
    "add_period_column",
    # comparison and variance
    "absolute_change",
    "percent_change",
    "basis_point_change",
    "variance",
    "variance_pct",
    # contribution / driver primitives
    "share_of_total",
    "contribution_to_change",
    "top_contributors",
    "component_contribution",
    # financial mathematics (time value of money)
    "discount_factor",
    "accumulation_factor",
    "effective_discount",
    "force_of_interest",
    "rate_from_force",
    "nominal_interest",
    "nominal_discount",
    "rate_from_nominal_interest",
    "rate_from_nominal_discount",
    "present_value",
    "future_value",
    "annuity_immediate",
    "annuity_due",
    "accumulated_immediate",
    "accumulated_due",
    "perpetuity_immediate",
    "perpetuity_due",
    "deferred_annuity_immediate",
    "annuity_continuous",
    "annuity_immediate_mthly",
    "increasing_annuity_immediate",
    "decreasing_annuity_immediate",
    "geometric_annuity_immediate",
    "net_present_value",
    "internal_rate_of_return",
    "level_payment",
    "outstanding_balance",
    "amortization_schedule",
    "discount_factors",
    "present_value_curve",
    "year_fraction",
]

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

try:
    __version__ = _version("actuarialpy")
except _PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

del _PackageNotFoundError, _version
