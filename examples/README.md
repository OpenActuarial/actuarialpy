# actuarialpy examples

Self-contained, runnable examples for the primitive surfaces of `actuarialpy`. Each script
generates its own small synthetic data (via `_sample_data.py`) and prints a short report, so
they run with nothing but the package installed:

```bash
pip install actuarialpy
python reserving_ibnr.py
```

Every script is standalone — run any one directly, in any order.

The walkthroughs are health-flavored (member-months, PMPM print labels) as a concrete book
to work through; the core they exercise is domain-agnostic, and the domain vocabulary lives
entirely in these callers.

| Script | Surface | What it shows |
|---|---|---|
| `reserving_ibnr.py` | reserving | `make_completion_triangle` → `completion_factors` → `apply_completion` → `ibnr`, per-line `completion_factors_by` + grouped `apply_completion(by=)`, and `develop_ultimate` (chain ladder vs Bornhuetter-Ferguson vs Benktander vs Cape Cod) |
| `development_uncertainty.py` | reserving | chain ladder with **Mack standard errors**: `ChainLadder.fit` → `project` → `mack_sigma_squared` → `mack_standard_errors`, reproducing the published Taylor-Ashe (1983)/Mack (1993) reserve and s.e. |
| `per_line_development.py` | reserving | grouped development via `chain_ladder_by` — each line fit by its own pattern with a CV per line, showing why pooling different volatilities hides the difference |
| `seasonality.py` | seasonality | `business_days_in_period`, `seasonality_factors` → `deseasonalize`, plus per-line `seasonality_factors_by` + grouped `deseasonalize(by=)` |
| `trend_and_forecast.py` | trend | `trend_summary`, `annualized_trend`, `project_forward`, `trend_factor`, and `fit_trend` (log-linear trend with diagnostics, fit on deseasonalized history) |
| `credibility.py` | credibility | limited-fluctuation (`full_credibility_claims`, `limited_fluctuation_z`, `credibility_weighted_estimate`) blended per group, and greatest-accuracy `BuhlmannStraub` (Z from EPV/VHM) |

## The sample data

`_sample_data.py` (not part of the library) provides deterministic generators shared by these
scripts — a long claim-payments frame per line of business for the reserving examples, a
monthly seasonal claims panel for the seasonality example, and a two-period trend panel. Each
is seeded so output is reproducible.

## Note

These mirror the worked examples shipped alongside the sibling packages (`lossmodels`,
`risksim`, `extremeloss`). For an end-to-end application that wires the packages together, see
the high-cost-claimant cost-model project.
