"""Credibility models and primitives.

The credibility tools a pricing or experience-rating actuary reaches for: the
credibility-weighting primitive plus greatest-accuracy (Bühlmann and
Bühlmann-Straub) credibility models.

The ``Buhlmann`` and ``BuhlmannStraub`` models were previously part of the
``lossmodels`` package and were moved here, where credibility sits next to the
experience and ratemaking workflows that consume it.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

_SCALAR_TYPES = (int, float, np.number)


def credibility_weighted_estimate(observed: Any, complement: Any, z: Any) -> Any:
    """Blend an observed estimate with its complement at credibility ``z``.

    Returns ``z * observed + (1 - z) * complement``. Scalar inputs return a
    native ``float``; ``pandas.Series`` inputs return a ``Series`` with the index
    preserved; other array-like inputs return a ``numpy.ndarray``. This is the
    atomic credibility operation; the ``z`` may come from a model below, a filed
    credibility formula, or any other source.
    """
    if isinstance(observed, pd.Series) or isinstance(complement, pd.Series) or isinstance(z, pd.Series):
        return z * observed + (1 - z) * complement
    if isinstance(observed, _SCALAR_TYPES) and isinstance(complement, _SCALAR_TYPES) and isinstance(z, _SCALAR_TYPES):
        return float(z * observed + (1 - z) * complement)
    observed_arr = np.asarray(observed, dtype=float)
    complement_arr = np.asarray(complement, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    return z_arr * observed_arr + (1 - z_arr) * complement_arr


class Buhlmann:
    """Bühlmann credibility model.

    This implementation assumes each risk has the same number of observations.

    Parameters
    ----------
    overall_mean : float
        Estimated collective mean.
    epv : float
        Estimated expected process variance (EPV).
    vhm : float
        Estimated variance of hypothetical means (VHM).
    n_obs : int
        Number of observations per risk.
    """

    def __init__(self, overall_mean: float, epv: float, vhm: float, n_obs: int):
        if n_obs <= 0:
            raise ValueError("n_obs must be positive.")
        if epv < 0:
            raise ValueError("epv must be nonnegative.")
        if vhm < 0:
            raise ValueError("vhm must be nonnegative.")

        self.overall_mean = float(overall_mean)
        self.epv = float(epv)
        self.vhm = float(vhm)
        self.n_obs = int(n_obs)

    @property
    def k(self) -> float:
        """K = EPV / VHM. Returns infinity when VHM = 0."""
        if self.vhm == 0:
            return float("inf")
        return self.epv / self.vhm

    @property
    def z(self) -> float:
        """Credibility factor ``Z = n / (n + K)``. Returns 0 when K is infinite."""
        k = self.k
        if not np.isfinite(k):
            return 0.0
        return self.n_obs / (self.n_obs + k)

    def premium(self, risk_mean: Any) -> Any:
        """Compute the Bühlmann credibility premium ``Z * risk_mean + (1 - Z) * overall_mean``.

        Parameters
        ----------
        risk_mean : float or array-like
            Risk-specific sample mean(s).

        Returns
        -------
        float or numpy.ndarray
            Credibility-weighted premium(s).
        """
        risk_mean = np.asarray(risk_mean, dtype=float)
        premium = self.z * risk_mean + (1.0 - self.z) * self.overall_mean
        return float(premium) if premium.ndim == 0 else premium

    @classmethod
    def fit(cls, data: Any) -> Buhlmann:
        """Fit a Bühlmann credibility model from data.

        Parameters
        ----------
        data : array-like, shape (m, n)
            Observations for m risks, each with n observations.

        Returns
        -------
        Buhlmann
            Fitted Bühlmann model.

        Notes
        -----
        Estimators used:

        - overall_mean = mean of all observations
        - EPV = average of within-risk sample variances
        - VHM = sample variance of risk means minus EPV / n, floored at 0
        """
        data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("data must be a 2D array with shape (n_risks, n_obs).")

        n_risks, n_obs = data.shape

        if n_risks < 2:
            raise ValueError("data must contain at least two risks.")
        if n_obs < 2:
            raise ValueError("each risk must have at least two observations.")

        risk_means = np.mean(data, axis=1)
        overall_mean = float(np.mean(data))

        within_vars = np.var(data, axis=1, ddof=1)
        epv = float(np.mean(within_vars))

        between_var = float(np.var(risk_means, ddof=1))
        vhm = max(between_var - epv / n_obs, 0.0)

        return cls(overall_mean=overall_mean, epv=epv, vhm=vhm, n_obs=n_obs)

    def __repr__(self) -> str:
        return (
            f"Buhlmann(overall_mean={self.overall_mean}, "
            f"epv={self.epv}, vhm={self.vhm}, n_obs={self.n_obs})"
        )


class BuhlmannStraub:
    """Bühlmann-Straub credibility model.

    This implementation allows different exposure weights by risk and period.

    Parameters
    ----------
    overall_mean : float
        Estimated collective mean.
    epv : float
        Estimated expected process variance (EPV).
    vhm : float
        Estimated variance of hypothetical means (VHM).
    weights : array-like
        Total weight (exposure) for each risk.
    """

    def __init__(self, overall_mean: float, epv: float, vhm: float, weights: Any):
        weights = np.asarray(weights, dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if weights.size == 0:
            raise ValueError("weights must not be empty.")
        if np.any(weights <= 0):
            raise ValueError("weights must be positive.")
        if epv < 0:
            raise ValueError("epv must be nonnegative.")
        if vhm < 0:
            raise ValueError("vhm must be nonnegative.")

        self.overall_mean = float(overall_mean)
        self.epv = float(epv)
        self.vhm = float(vhm)
        self.weights = weights

    @property
    def k(self) -> float:
        """K = EPV / VHM. Returns infinity when VHM = 0."""
        if self.vhm == 0:
            return float("inf")
        return self.epv / self.vhm

    def z(self, weight: Any) -> Any:
        """Credibility factor for a given total risk weight: ``Z_i = w_i / (w_i + K)``.

        Parameters
        ----------
        weight : float or array-like
            Total exposure weight(s).

        Returns
        -------
        float or numpy.ndarray
            Credibility factor(s).
        """
        weight = np.asarray(weight, dtype=float)

        if np.any(weight <= 0):
            raise ValueError("weight must be positive.")

        k = self.k
        if not np.isfinite(k):
            out = np.zeros_like(weight, dtype=float)
        else:
            out = weight / (weight + k)

        return float(out) if out.ndim == 0 else out

    def premium(self, risk_mean: Any, weight: Any) -> Any:
        """Compute the Bühlmann-Straub premium ``Z_i * risk_mean_i + (1 - Z_i) * overall_mean``.

        Parameters
        ----------
        risk_mean : float or array-like
            Risk-specific weighted mean(s).
        weight : float or array-like
            Total exposure weight(s).

        Returns
        -------
        float or numpy.ndarray
            Credibility-weighted premium(s).
        """
        risk_mean = np.asarray(risk_mean, dtype=float)
        z = self.z(weight)
        premium = z * risk_mean + (1.0 - z) * self.overall_mean
        return float(premium) if np.ndim(premium) == 0 else premium

    @staticmethod
    def _to_risk_lists(data: Any, weights: Any):
        """Normalize ``(data, weights)`` into lists of 1D arrays, one per risk.

        Accepts a 2D array (equal period counts) or a sequence of 1D arrays per
        risk (unequal period counts), validating shapes and positive weights.
        """
        arr: Any
        try:
            arr = np.asarray(data, dtype=float)
        except (ValueError, TypeError):
            arr = None
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            warr = np.asarray(weights, dtype=float)
            if warr.ndim != 2:
                raise ValueError("weights must be a 2D array when data is 2D.")
            if arr.shape != warr.shape:
                raise ValueError("data and weights must have the same shape.")
            if arr.shape[0] < 2:
                raise ValueError("data must contain at least two risks.")
            if arr.shape[1] < 2:
                raise ValueError("each risk must have at least two periods.")
            obs = [arr[i] for i in range(arr.shape[0])]
            wts = [warr[i] for i in range(warr.shape[0])]
        else:
            obs = [np.asarray(x, dtype=float) for x in data]
            wts = [np.asarray(w, dtype=float) for w in weights]
            if len(obs) != len(wts):
                raise ValueError("data and weights must have the same number of risks.")
            if len(obs) < 2:
                raise ValueError("data must contain at least two risks.")
            for x, w in zip(obs, wts, strict=True):
                if x.ndim != 1 or w.ndim != 1:
                    raise ValueError("each risk's data and weights must be 1D.")
                if x.shape != w.shape:
                    raise ValueError("each risk's data and weights must have the same length.")
                if x.size < 2:
                    raise ValueError("each risk must have at least two periods.")
        allw = np.concatenate([np.ravel(w) for w in wts])
        if np.any(allw <= 0):
            raise ValueError("weights must be positive.")
        return obs, wts

    @staticmethod
    def _estimate(risk_obs, risk_wts):
        r"""General Bühlmann-Straub estimators on per-risk observation/weight lists.

        For risks :math:`i` with observations :math:`X_{ij}` and weights
        :math:`w_{ij}`, with :math:`m_i = \sum_j w_{ij}`,
        :math:`\bar X_i = \sum_j w_{ij} X_{ij} / m_i`, :math:`m = \sum_i m_i`,
        and :math:`\bar X = \sum_i m_i \bar X_i / m`:

        .. math::
            \hat s^2 = \frac{\sum_i \sum_j w_{ij}(X_{ij}-\bar X_i)^2}{\sum_i (n_i-1)},
            \qquad
            \hat a = \frac{\sum_i m_i(\bar X_i-\bar X)^2 - (r-1)\hat s^2}
                          {m - \sum_i m_i^2 / m}.

        Handles unequal :math:`n_i`; :math:`\hat a` is floored at 0 (no credibility).
        Returns ``(overall_mean, epv, vhm, m_i, xbar_i)``.
        """
        r = len(risk_obs)
        if r < 2:
            raise ValueError("need at least two risks.")
        m_i = np.array([float(w.sum()) for w in risk_wts])
        xbar_i = np.array(
            [float(np.sum(w * x) / w.sum()) for x, w in zip(risk_obs, risk_wts, strict=True)]
        )
        m = float(m_i.sum())
        overall_mean = float(np.sum(m_i * xbar_i) / m)

        ss_within = float(
            sum(np.sum(w * (x - xb) ** 2) for x, w, xb in zip(risk_obs, risk_wts, xbar_i, strict=True))
        )
        dof = int(sum(len(x) - 1 for x in risk_obs))
        if dof <= 0:
            raise ValueError("need at least one risk with more than one observation.")
        epv = ss_within / dof

        between = float(np.sum(m_i * (xbar_i - overall_mean) ** 2))
        denom = m - float(np.sum(m_i**2)) / m
        vhm = max((between - (r - 1) * epv) / denom, 0.0) if denom > 0 else 0.0
        return overall_mean, epv, vhm, m_i, xbar_i

    @classmethod
    def fit(cls, data: Any, weights: Any) -> BuhlmannStraub:
        r"""Fit a Bühlmann-Straub model from observations and weights.

        Accepts either a 2D array (equal period counts) or a sequence of 1D
        arrays per risk (unequal period counts). The estimators are the general
        unbiased forms

        .. math::
            \hat s^2 = \frac{\sum_i\sum_j w_{ij}(X_{ij}-\bar X_i)^2}{\sum_i(n_i-1)},
            \quad
            \hat a = \frac{\sum_i m_i(\bar X_i-\bar X)^2 - (r-1)\hat s^2}
                          {m - \sum_i m_i^2/m},

        with :math:`k=\hat s^2/\hat a` and :math:`Z_i=m_i/(m_i+k)`; a negative
        :math:`\hat a` is floored at 0. For equal period counts this reduces to
        the usual estimator; unlike a divide-by-mean-weight approximation it stays
        unbiased when risks have different period counts or exposures.

        Parameters
        ----------
        data : array-like
            Either shape ``(r, n)`` (equal periods) or a sequence of ``r`` 1D
            arrays ``X_i`` whose lengths may differ.
        weights : array-like
            Exposure weights matching the shape/structure of ``data``.
        """
        risk_obs, risk_wts = cls._to_risk_lists(data, weights)
        overall_mean, epv, vhm, m_i, xbar_i = cls._estimate(risk_obs, risk_wts)
        model = cls(overall_mean=overall_mean, epv=epv, vhm=vhm, weights=m_i)
        model.groups_ = None
        model.risk_means_ = xbar_i
        return model

    @classmethod
    def from_frame(
        cls,
        df,
        *,
        group: str,
        value: str,
        weight: str,
        period: str | None = None,
    ) -> BuhlmannStraub:
        """Fit from long-format data: one row per (risk, period).

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format observations.
        group, value, weight : str
            Column names for the risk identifier, the per-unit observation (e.g.
            loss per member-month), and the exposure weight.
        period : str, optional
            Period column; used only to order observations within a risk. The
            number of observations per risk may differ.

        Returns
        -------
        BuhlmannStraub
            Fitted model with ``groups_`` (risk labels), ``risk_means_``, and
            ``weights`` (per-risk total exposure), all aligned to ``groups_``.
        """
        required = [group, value, weight] + ([period] if period else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"columns not found in df: {missing}")
        work = df[required].copy()
        work[value] = work[value].astype(float)
        work[weight] = work[weight].astype(float)
        if np.any(work[weight].to_numpy() <= 0):
            raise ValueError("weights must be positive.")

        risk_obs, risk_wts, labels = [], [], []
        for label, g in work.groupby(group, sort=True):
            if period is not None:
                g = g.sort_values(period)
            risk_obs.append(g[value].to_numpy(dtype=float))
            risk_wts.append(g[weight].to_numpy(dtype=float))
            labels.append(label)

        overall_mean, epv, vhm, m_i, xbar_i = cls._estimate(risk_obs, risk_wts)
        model = cls(overall_mean=overall_mean, epv=epv, vhm=vhm, weights=m_i)
        model.groups_ = pd.Index(labels, name=group)
        model.risk_means_ = xbar_i
        return model

    def __repr__(self) -> str:
        return (
            f"BuhlmannStraub(overall_mean={self.overall_mean}, "
            f"epv={self.epv}, vhm={self.vhm}, weights={self.weights})"
        )


def limited_fluctuation_z(exposure: Any, full_credibility_standard: float) -> Any:
    """Limited-fluctuation (classical) credibility factor -- the square-root rule.

    Returns ``Z = min(1, sqrt(exposure / full_credibility_standard))``. ``exposure``
    is the volume credibility is based on (claim counts, member months, life-years,
    ...) and ``full_credibility_standard`` is the amount of that volume required for
    full (``Z = 1``) credibility -- often a filed value. Scalars return a native
    ``float``; ``pandas.Series`` inputs return a ``Series`` (index preserved); other
    array-likes return a ``numpy.ndarray``, so credibility can be computed per group.
    Feed the result to :func:`credibility_weighted_estimate` to blend experience with
    its complement.
    """
    if full_credibility_standard <= 0:
        raise ValueError("full_credibility_standard must be positive.")
    if isinstance(exposure, _SCALAR_TYPES):
        return float(min(1.0, np.sqrt(max(float(exposure), 0.0) / full_credibility_standard)))
    ratio_arr = np.asarray(exposure, dtype=float) / full_credibility_standard
    z_arr = np.minimum(np.sqrt(np.clip(ratio_arr, 0.0, None)), 1.0)
    if isinstance(exposure, pd.Series):
        return pd.Series(z_arr, index=exposure.index, name=exposure.name)
    return z_arr


def full_credibility_claims(
    *, confidence: float = 0.90, tolerance: float = 0.05, severity_cv: float | None = None
) -> float:
    """Classical full-credibility standard, in expected number of claims.

    Returns the expected claim count for full credibility under the
    limited-fluctuation model: ``(z / k) ** 2`` for claim frequency, where ``z`` is
    the standard-normal quantile for two-sided ``confidence`` and ``k`` is the
    ``tolerance``. The classic 90% / 5% choice gives about 1082 claims. Supplying
    ``severity_cv`` (the coefficient of variation of individual claim severity)
    inflates it to ``(z / k) ** 2 * (1 + severity_cv ** 2)`` for aggregate losses
    rather than pure frequency.

    Many shops use a filed standard instead; pass that straight to
    :func:`limited_fluctuation_z`.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    standard = (z / tolerance) ** 2
    if severity_cv is not None:
        if severity_cv < 0.0:
            raise ValueError("severity_cv must be non-negative.")
        standard *= 1.0 + severity_cv**2
    return standard
