"""Cross-library equivalence with chainladder-python on the GenIns data.

Both libraries cite Mack (1993) and ship the same canonical dataset
(GenIns == Taylor & Ashe), so overlapping outputs should agree -- and
where they do not, the difference should be a *documented methodological
choice*, not a mystery. Findings encoded below:

- development factors and ultimates agree to machine precision;
- the total Mack standard error agrees to ~0.2%: ours reproduces the
  paper's published 2,447,095 exactly, chainladder's default reports
  2,441,364 because it extrapolates the final-period sigma^2 by
  log-linear regression where Mack's paper (and this package) use his
  min-rule -- the residual per-origin differences (largest for the
  origins whose remaining development is dominated by that final
  period) trace to the same choice.
"""
import numpy as np
import pandas as pd
import pytest

from actuarialpy.reserving import ChainLadder

clp = pytest.importorskip("chainladder")


@pytest.fixture(scope="module")
def both_fits():
    genins = clp.load_sample("genins")
    tri = pd.DataFrame(
        genins.values[0, 0],
        index=pd.Index(range(1, 11), name="origin"),
        columns=range(1, 11),
    ).astype(float)
    ours = ChainLadder.fit(tri)
    return tri, ours, ours.mack_standard_errors(tri), clp.MackChainladder().fit(genins)


def test_development_factors_identical(both_fits):
    _, ours, _, theirs = both_fits
    ldf = np.asarray(theirs.ldf_.values[0, 0, 0], dtype=float)
    np.testing.assert_allclose(ours.age_to_age.to_numpy(), ldf[:9], rtol=1e-12)


def test_ultimates_identical(both_fits):
    _, _, mack_ours, theirs = both_fits
    ult = theirs.ultimate_.to_frame(origin_as_datetime=False).iloc[:, 0].to_numpy()
    np.testing.assert_allclose(
        mack_ours["ultimate"].iloc[:-1].to_numpy(), ult, rtol=1e-12
    )


def test_mack_errors_agree_within_documented_choice(both_fits):
    _, _, mack_ours, theirs = both_fits
    total_theirs = float(np.asarray(theirs.total_mack_std_err_).ravel()[0])
    # ours is the paper's published figure; chainladder's sigma
    # extrapolation default lands ~0.2% away
    assert mack_ours.loc["Total", "se"] == pytest.approx(2_447_095, rel=5e-3)
    assert mack_ours.loc["Total", "se"] == pytest.approx(total_theirs, rel=5e-3)
    se_theirs = theirs.summary_.to_frame(origin_as_datetime=False)[
        "Mack Std Err"].to_numpy()
    np.testing.assert_allclose(
        mack_ours["se"].iloc[1:-1].to_numpy(), se_theirs[1:], rtol=6e-2
    )
