# actuarialpy

Purpose-neutral actuarial calculation primitives, plus the shared actuarial data contract — the foundation of the OpenActuarial ecosystem.

> `actuarialpy.Experience` is the ecosystem's canonical semantic wrapper for
> historical actuarial data: it binds column roles, grain metadata, and
> snapshot context. Its domain operations are immutable transformations;
> calculations and workflow outputs belong to consuming packages.

[![CI](https://github.com/OpenActuarial/actuarialpy/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/actuarialpy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/actuarialpy)](https://pypi.org/project/actuarialpy/)
[![Python](https://img.shields.io/pypi/pyversions/actuarialpy)](https://pypi.org/project/actuarialpy/)

## Overview

`actuarialpy` provides the atomic building blocks the rest of the ecosystem
is written against: ratios and per-exposure metrics, claim development and
completion, trend fitting and projection, credibility, large-claim pooling,
and financial mathematics. Everything operates on plain floats, NumPy arrays,
and pandas objects, with a consistent type-mirroring convention (scalar in,
`float` out; `Series` in, `Series` out with the index preserved).

The package deliberately contains no workflow orchestration and no
domain-specific vocabulary — those belong to the workflow packages built on
top of it. If a function here needs to know *why* you are calling it, it does
not belong here.

## Installation

```bash
pip install actuarialpy
```

Requires Python 3.10 or newer.

## Quick start

```python
import pandas as pd
import actuarialpy as ap

# ratios and per-exposure rates on any aggregate
print(ap.loss_ratio(1_240_000, 1_500_000))       # 0.8267
print(ap.per_exposure(1_240_000, 12_000))        # 103.33 per exposure unit

# trend claim severity to project future loss costs
monthly = pd.DataFrame({
    "month": pd.date_range("2024-01-01", periods=24, freq="MS"),
    "avg_severity": [5_000 * 1.004 ** i for i in range(24)],
    "claim_count": [20] * 24,
})
severity_trend = ap.fit_trend(monthly, date_col="month", value_col="avg_severity")
# if severity trends at +0.4%/month and claim count stays flat,
# projected losses next quarter will be:
projected_severity = ap.project_forward(monthly["avg_severity"].iloc[-1], 
                                        severity_trend.annual_trend, months=3)
projected_losses = projected_severity * monthly["claim_count"].iloc[-1]

# cap large claims at a pooling point; the excess moves to its own column
claims = pd.DataFrame({"member": ["a", "b", "c"],
                       "paid": [612_000.0, 340_000.0, 96_500.0]})
pooled = ap.pool_losses(claims, loss_col="paid", pooling_point=250_000)
print(pooled)
```

## What's inside

- **Metrics** — loss/expense ratios, per-exposure rates, weighted statistics,
  contribution and comparison helpers.
- **Reserving** — completion triangles, chain-ladder development factors,
  Mack standard errors, completion applied back to tidy data.
- **Trend and seasonality** — trend fitting, forward projection, seasonal
  adjustment.
- **Credibility** — Bühlmann, Bühlmann–Straub, and limited-fluctuation
  credibility.
- **Pooling** — large-claim capping and excess extraction.
- **Financial** — time-value-of-money primitives (present/future value,
  annuities, rate conversions).
- **Data utilities** — exposure handling, banding, period alignment, member
  lifecycle status, margins and adjustments.

The full API reference and end-to-end worked examples live at
**[openactuarial.org/actuarialpy.html](https://openactuarial.org/actuarialpy.html)**.

## The OpenActuarial ecosystem

`actuarialpy` is one of eight packages that share conventions — tidy tables,
explicit distribution parameterizations, reproducible random-number handling —
and compose across package seams:

| Package | Role |
|---|---|
| **[actuarialpy](https://github.com/OpenActuarial/actuarialpy)** | Calculation primitives the workflow packages build on |
| [experiencestudies](https://github.com/OpenActuarial/experiencestudies) | Experience reporting, actual-vs-expected, claimant and concentration analysis |
| [projectionmodels](https://github.com/OpenActuarial/projectionmodels) | Claim, premium, and expense projection over a renewal horizon |
| [ratingmodels](https://github.com/OpenActuarial/ratingmodels) | Manual and experience rating, credibility, indication, GLM relativities |
| [reservingmodels](https://github.com/OpenActuarial/reservingmodels) | Claims development and stochastic reserving: chain ladder, BF, Mack, ODP bootstrap |
| [lossmodels](https://github.com/OpenActuarial/lossmodels) | Severity and frequency fitting, aggregate loss distributions |
| [extremeloss](https://github.com/OpenActuarial/extremeloss) | Extreme-value tails: POT/GPD, GEV, return levels, splicing |
| [risksim](https://github.com/OpenActuarial/risksim) | Portfolio Monte Carlo, dependence, reinsurance contracts, risk measures |

Install everything at once with `pip install openactuarial`.

## Development

```bash
git clone https://github.com/OpenActuarial/actuarialpy
cd actuarialpy
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

CI runs the same gate on Python 3.10–3.14 across Linux and Windows.

## Versioning and stability

All ecosystem packages are pre-1.0: minor releases may change APIs, and every
release is documented in [CHANGELOG.md](CHANGELOG.md). Current per-package API
stability is tracked at
[openactuarial.org/stability.html](https://openactuarial.org/stability.html).

## License

MIT — see [LICENSE](LICENSE).
