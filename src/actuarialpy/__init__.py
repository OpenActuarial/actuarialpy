"""ActuarialPy: tools for actuarial experience analysis."""

from actuarialpy.frame import Experience
from actuarialpy.metrics import (
    actual_to_expected,
    combined_ratio,
    expense_ratio,
    frequency,
    indicated_change,
    loss_ratio,
    medical_loss_ratio,
    pepm,
    per_exposure,
    permissible_loss_ratio,
    pmpm,
    pspm,
    pure_premium,
    ratio,
    required_revenue,
    safe_divide,
    severity,
    utilization_per_1000,
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
from actuarialpy.banding import assign_band, summarize_by_band
from actuarialpy.adjustments import adjust
from actuarialpy.columns import factor_lookup
from actuarialpy.margins import add_margin, margin, margin_ratio
from actuarialpy.pooling import excess_over_threshold, pool_losses
from actuarialpy.experience import status_summary, summarize_experience, summarize_views
from actuarialpy.expected import summarize_actual_vs_expected
from actuarialpy.claimants import summarize_claimants, top_claimants, large_claimant_flags, claim_concentration
from actuarialpy.rolling import rolling_summary
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
from actuarialpy.components import component_driver_analysis, component_trend, summarize_components
from actuarialpy.cohorts import cohort_summary, cohort_summary_by_period, duration_summary
from actuarialpy.decomposition import decompose_pmpm_trend, frequency_severity_summary
from actuarialpy.seasonality import (
    add_business_days,
    apply_seasonality,
    business_days_in_period,
    deseasonalize,
    seasonality_factors,
    seasonality_factors_by,
)

__all__ = [
    "Experience",
    "actual_to_expected",
    "adjust",
    "factor_lookup",
    "combined_ratio",
    "expense_ratio",
    "frequency",
    "indicated_change",
    "loss_ratio",
    "medical_loss_ratio",
    "pepm",
    "per_exposure",
    "permissible_loss_ratio",
    "pmpm",
    "pspm",
    "pure_premium",
    "utilization_per_1000",
    "ratio",
    "required_revenue",
    "safe_divide",
    "severity",
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
    "status_summary",
    "summarize_experience",
    "summarize_views",
    "summarize_actual_vs_expected",
    "summarize_claimants",
    "top_claimants",
    "large_claimant_flags",
    "claim_concentration",
    "cohort_summary",
    "cohort_summary_by_period",
    "frequency_severity_summary",
    "decompose_pmpm_trend",
    "duration_summary",
    "rolling_summary",
    "annualized_trend",
    "midpoint_trend_factor",
    "period_change",
    "project_forward",
    "fit_trend",
    "TrendFit",
    "trend_factor",
    "trend_summary",
    "component_driver_analysis",
    "component_trend",
    "summarize_components",
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
    # banding
    "assign_band",
    "summarize_by_band",
    # margins
    "add_margin",
    "margin",
    "margin_ratio",
    # large-loss pooling
    "excess_over_threshold",
    "pool_losses",
    # seasonality and working-day adjustment
    "business_days_in_period",
    "add_business_days",
    "seasonality_factors",
    "seasonality_factors_by",
    "deseasonalize",
    "apply_seasonality",
]

__version__ = "0.26.0"
