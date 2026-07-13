"""Exactness tests for the Student-t quantile, incl. the dependency-free fallback.

actuarialpy does not depend on SciPy, so ``_student_t_ppf`` uses SciPy only when
importable and otherwise falls back to a from-scratch incomplete-beta inversion.
These tests exercise *both* paths and pin the specific small-df bug the exact
implementation fixes (the old Cornish-Fisher expansion was ~11% too narrow at
df=1).
"""

from __future__ import annotations

import sys

import pytest

from actuarialpy.trend import (
    _reg_incomplete_beta,
    _student_t_cdf,
    _student_t_ppf,
)

DFS = [1, 2, 3, 4, 5, 7, 8, 10, 15, 30, 60, 120, 500]
PS = [0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]


def test_scipy_path_matches_scipy():
    """When SciPy is importable, the ppf is SciPy's ppf exactly."""
    scit = pytest.importorskip("scipy.stats").t
    for df in DFS:
        for p in PS:
            assert _student_t_ppf(p, df) == pytest.approx(float(scit.ppf(p, df)), rel=1e-12, abs=1e-12)


def test_df1_regression_value():
    """df=1, p=0.975 must be the exact 12.7062, not the old approximation's 11.30."""
    assert _student_t_ppf(0.975, 1) == pytest.approx(12.706205, abs=1e-4)
    assert _student_t_ppf(0.975, 2) == pytest.approx(4.302653, abs=1e-4)


def test_fallback_matches_scipy_across_grid(monkeypatch):
    """Force the no-SciPy path and confirm it still matches SciPy to ~1e-8."""
    scit = pytest.importorskip("scipy.stats").t
    ref = {(df, p): float(scit.ppf(p, df)) for df in DFS for p in PS}

    monkeypatch.setitem(sys.modules, "scipy.stats", None)  # make `from scipy.stats import t` raise
    for (df, p), expected in ref.items():
        got = _student_t_ppf(p, df)
        assert got == pytest.approx(expected, rel=1e-7, abs=1e-8), f"df={df} p={p}"


def test_fallback_symmetry(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    for df in DFS:
        for p in [0.9, 0.975, 0.999]:
            assert _student_t_ppf(1 - p, df) == pytest.approx(-_student_t_ppf(p, df), rel=1e-10, abs=1e-10)
    assert _student_t_ppf(0.5, 3) == 0.0


def test_incomplete_beta_matches_scipy():
    """The hand-rolled regularized incomplete beta matches SciPy's betainc."""
    betainc = pytest.importorskip("scipy.special").betainc
    for a in [0.5, 1.0, 2.5, 5.0, 30.0]:
        for b in [0.5, 1.0, 3.0]:
            for x in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
                assert _reg_incomplete_beta(x, a, b) == pytest.approx(float(betainc(a, b, x)), abs=1e-12)


def test_cdf_ppf_roundtrip(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    for df in DFS:
        for p in PS:
            q = _student_t_ppf(p, df)
            assert _student_t_cdf(q, df) == pytest.approx(p, abs=1e-8)


def test_ppf_domain_errors():
    for bad_p in [0.0, 1.0, -0.1, 1.5]:
        with pytest.raises(ValueError):
            _student_t_ppf(bad_p, 5)
    with pytest.raises(ValueError):
        _student_t_ppf(0.9, 0)
