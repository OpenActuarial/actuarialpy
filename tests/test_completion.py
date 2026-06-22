import pandas as pd
import pytest

from actuarialpy.completion import (
    complete_claim_components,
    complete_claims,
    completed_from_factor,
    ibnr,
    make_completion_triangle,
)


def test_completion_primitives():
    assert completed_from_factor(900_000, 0.9, method="divide") == 1_000_000
    assert ibnr(1_000_000, 900_000) == 100_000


def test_complete_claims():
    df = pd.DataFrame({"paid_claims": [900_000], "completion_factor": [0.9]})
    result = complete_claims(df)
    assert result.loc[0, "completed_claims"] == 1_000_000
    assert result.loc[0, "ibnr"] == 100_000


def test_complete_claim_components():
    df = pd.DataFrame({
        "inpatient_claims": [900],
        "outpatient_claims": [800],
        "inpatient_completion_factor": [0.9],
        "outpatient_completion_factor": [0.8],
    })
    out = complete_claim_components(
        df,
        {
            "inpatient_claims": "inpatient_completion_factor",
            "outpatient_claims": "outpatient_completion_factor",
        },
    )
    assert out.loc[0, "inpatient_claims_completed"] == 1000
    assert out.loc[0, "outpatient_claims_completed"] == 1000
    assert out.loc[0, "inpatient_claims_ibnr"] == 100


def test_completion_triangle():
    df = pd.DataFrame({"origin": ["2026-01-01", "2026-01-01"], "valuation": ["2026-01-31", "2026-02-28"], "paid": [100, 150]})
    tri = make_completion_triangle(df, origin_col="origin", valuation_col="valuation", amount_col="paid")
    assert tri.loc[pd.Period("2026-01", "M"), 0] == 100
    assert tri.loc[pd.Period("2026-01", "M"), 1] == 150
