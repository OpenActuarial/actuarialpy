"""ActuarialPy: primitive-based tools for actuarial experience analysis.

The most commonly used workflows are re-exported here so they are importable
directly from the top level, e.g. ``from actuarialpy import summarize_experience``.
"""

from actuarialpy.banding import assign_band, summarize_by_band
from actuarialpy.cohorts import (
    cohort_summary,
    cohort_summary_by_period,
    duration_summary,
)
from actuarialpy.columns import (
    as_list,
    ensure_unique_keys,
    per_exposure_name,
    sum_columns,
    validate_columns,
)
from actuarialpy.compare import (
    absolute_change,
    basis_point_change,
    percent_change,
    relative_change,
    variance,
    variance_pct,
)
from actuarialpy.completion import (
    complete_claim_components,
    complete_claims,
    completed_experience,
    completed_from_factor,
    ibnr,
    lag_months,
    make_completion_triangle,
    validate_completion_factors,
)
from actuarialpy.components import (
    component_driver_analysis,
    component_trend,
    summarize_components,
)
from actuarialpy.concentration import (
    concentration_curve,
    concentration_summary,
    top_n_share,
)
from actuarialpy.contribution import (
    component_contribution,
    contribution_to_change,
    share_of_total,
    top_contributors,
)
from actuarialpy.credibility import (
    Buhlmann,
    BuhlmannStraub,
    credibility_weighted_estimate,
)
from actuarialpy.experience import (
    status_summary,
    summarize_experience,
    summarize_views,
)
from actuarialpy.forecast import (
    compare_actual_to_expected,
    expected_from_rate,
    forecast_experience,
    forecast_from_rate,
)
from actuarialpy.frame import Experience
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
from actuarialpy.margins import add_margin, margin, margin_ratio
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
)
from actuarialpy.periods import (
    add_duration_column,
    add_period_column,
    months_between,
    period_label,
    to_period,
)
from actuarialpy.pooling import (
    excess_over_threshold,
    flag_large_losses,
    large_loss_summary,
    pool_losses,
)
from actuarialpy.reporting import to_excel_report
from actuarialpy.rolling import rolling_summary
from actuarialpy.trend import (
    annualized_trend,
    midpoint_trend_factor,
    period_change,
    project_forward,
    trend_factor,
    trend_summary,
)

__all__ = [
    # metrics
    "actual_to_expected",
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
    "ratio",
    "required_revenue",
    "safe_divide",
    "severity",
    # comparison / change
    "absolute_change",
    "basis_point_change",
    "percent_change",
    "relative_change",
    "variance",
    "variance_pct",
    # completion / IBNR
    "complete_claim_components",
    "complete_claims",
    "completed_experience",
    "completed_from_factor",
    "ibnr",
    "lag_months",
    "make_completion_triangle",
    "validate_completion_factors",
    # experience summaries
    "status_summary",
    "summarize_experience",
    "summarize_views",
    "Experience",
    # components / contribution
    "component_contribution",
    "component_driver_analysis",
    "component_trend",
    "contribution_to_change",
    "share_of_total",
    "summarize_components",
    "top_contributors",
    # credibility
    "Buhlmann",
    "BuhlmannStraub",
    "credibility_weighted_estimate",
    # rolling
    "rolling_summary",
    # cohorts
    "cohort_summary",
    "cohort_summary_by_period",
    "duration_summary",
    # forecast
    "compare_actual_to_expected",
    "expected_from_rate",
    "forecast_experience",
    "forecast_from_rate",
    # trend
    "annualized_trend",
    "midpoint_trend_factor",
    "period_change",
    "project_forward",
    "trend_factor",
    "trend_summary",
    # periods
    "add_duration_column",
    "add_period_column",
    "months_between",
    "period_label",
    "to_period",
    # lifecycle (status / tenure / in-force / earned exposure)
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
    # concentration
    "concentration_curve",
    "concentration_summary",
    "top_n_share",
    # margins
    "add_margin",
    "margin",
    "margin_ratio",
    # large-loss / pooling
    "excess_over_threshold",
    "flag_large_losses",
    "large_loss_summary",
    "pool_losses",
    # column / validation helpers
    "as_list",
    "ensure_unique_keys",
    "per_exposure_name",
    "sum_columns",
    "validate_columns",
    # reporting
    "to_excel_report",
]

__version__ = "0.8.0"
