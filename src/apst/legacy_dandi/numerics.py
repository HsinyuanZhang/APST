"""Numerical primitives for the fair_v2 paired baseline protocol.

All fitting and exported maps use float64.  Inputs are observations by channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

_EPS = np.finfo(np.float64).eps


def _array2(x: np.ndarray, name: str = "x") -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional (observations, channels), got {a.shape}"
        )
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def smooth_raw(
    x: np.ndarray, bin_ms: float = 20, tau_ms: float = 240, extent: float = 1
) -> np.ndarray:
    """Causally convolve a complete raw stream with FALCON's normalized kernel.

    The output is float64 and has the same shape as ``x``.  There is no trial
    reset: callers must pass the whole recording before selecting support bins.
    ``extent=1, tau_ms=240, bin_ms=20`` produces its official 12 taps.
    """
    a = _array2(x)
    if bin_ms <= 0 or tau_ms <= 0 or extent <= 0:
        raise ValueError("bin_ms, tau_ms, and extent must be positive")
    n_taps = max(1, int(round(extent * tau_ms / bin_ms)))
    t = np.arange(n_taps, dtype=np.float64) * float(bin_ms)
    kernel = np.exp(-t / float(tau_ms))
    kernel /= kernel.sum()
    out = np.empty_like(a, dtype=np.float64)
    for channel in range(a.shape[1]):
        out[:, channel] = np.convolve(a[:, channel], kernel, mode="full")[: a.shape[0]]
    return out


@dataclass
class FactorModel:
    """Diagonal-noise FA fitted with covariance-based EM and KxK E steps."""

    components_: np.ndarray  # C, channels by latent dimensions
    mean_: np.ndarray
    noise_variance_: np.ndarray
    latent_dim: int
    diagnostics: dict

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        latent_dim: int,
        seed: int = 42,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        n_init: int = 3,
        noise_floor: float = 1e-6,
    ) -> "FactorModel":
        a = _array2(x)
        n, c = a.shape
        if not 0 < latent_dim < c:
            raise ValueError(
                "latent_dim must be positive and smaller than channel count"
            )
        if n < 2 or max_iter < 1 or n_init < 1 or tol < 0 or noise_floor <= 0:
            raise ValueError("invalid FactorModel fitting parameters")
        mean = a.mean(0)
        xc = a - mean
        # The entire sample dependence is compressed once into CxC sufficient
        # statistics.  EM iterations below never form an NxC latent matrix.
        sample_cov = xc.T @ xc / n
        var = np.maximum(np.diag(sample_cov), noise_floor)
        rng = np.random.RandomState(seed)
        runs = []
        best = None
        for restart in range(n_init):
            # A covariance eigendecomposition is used only for initialization,
            # never in the repeated EM E-step.
            if restart == 0:
                eigval, eigvec = np.linalg.eigh((sample_cov + sample_cov.T) * 0.5)
                order = np.argsort(eigval)[::-1][:latent_dim]
                base_noise = max(float(np.median(eigval)), noise_floor)
                load = eigvec[:, order] * np.sqrt(
                    np.maximum(eigval[order] - base_noise, noise_floor)
                )
                psi = np.maximum(
                    np.diag(sample_cov) - (load * load).sum(1), noise_floor
                )
            else:
                load = rng.normal(
                    scale=np.sqrt(var.mean() / latent_dim), size=(c, latent_dim)
                )
                psi = var.copy()
            previous_ll = -np.inf
            converged = False
            last_improvement = np.nan
            for iteration in range(1, max_iter + 1):
                try:
                    load, psi = _fa_em_step_sufficient(
                        sample_cov, load, psi, noise_floor
                    )
                except np.linalg.LinAlgError:
                    break
                ll = _fa_loglikelihood_sufficient(sample_cov, n, load, psi)
                last_improvement = (ll - previous_ll) / n
                if (
                    iteration > 1
                    and np.isfinite(last_improvement)
                    and abs(last_improvement) <= tol
                ):
                    converged = True
                    break
                previous_ll = ll
            final_ll = _fa_loglikelihood_sufficient(sample_cov, n, load, psi)
            record = {
                "restart": restart,
                "iterations": iteration,
                "converged": converged,
                "log_likelihood": float(final_ll),
                "per_sample_improvement": float(last_improvement),
                "noise_min": float(psi.min()),
                "noise_max": float(psi.max()),
            }
            runs.append(record)
            candidate = (final_ll, load.copy(), psi.copy(), record)
            if best is None or candidate[0] > best[0]:
                best = candidate
        best_ll, components, noise, best_record = best
        diagnostics = {
            "algorithm": "diagonal_psi_covariance_em_woodbury",
            "best_log_likelihood": float(best_ll),
            "best_restart": int(best_record["restart"]),
            "converged": bool(best_record["converged"]),
            "iterations": int(best_record["iterations"]),
            "per_sample_improvement": float(best_record["per_sample_improvement"]),
            "n_init": int(n_init),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "noise_floor": float(noise_floor),
            "runs": runs,
        }
        return cls(components, mean, noise, latent_dim, diagnostics)

    @property
    def C_(self) -> np.ndarray:
        return self.components_

    @property
    def psi_(self) -> np.ndarray:
        return self.noise_variance_

    def posterior_matrix(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Return beta (latent by selected-channel) for E[z|x]."""
        if rows is None:
            c, psi = self.components_, self.noise_variance_
        else:
            rows = np.asarray(rows, dtype=int)
            c, psi = self.components_[rows], self.noise_variance_[rows]
        invpsi = 1.0 / psi
        m = np.eye(self.latent_dim) + c.T @ (invpsi[:, None] * c)
        return np.linalg.solve(m, (c * invpsi[:, None]).T)

    def transform(self, x: np.ndarray) -> np.ndarray:
        a = _array2(x)
        if a.shape[1] != self.components_.shape[0]:
            raise ValueError("x channel count differs from fitted FactorModel")
        return (a - self.mean_) @ self.posterior_matrix().T


def _fa_em_step_sufficient(
    sample_cov: np.ndarray, load: np.ndarray, psi: np.ndarray, noise_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    """One diagonal-FA EM step from S=X'X/N, with no observation-axis arrays."""
    invpsi = 1.0 / psi
    m = np.eye(load.shape[1]) + load.T @ (invpsi[:, None] * load)
    minv = np.linalg.inv(m)
    beta = minv @ (load * invpsi[:, None]).T  # E[z x'] coefficient
    ezx = beta @ sample_cov  # E[z x'] averaged over samples
    ezz = minv + beta @ sample_cov @ beta.T  # E[z z'] averaged over samples
    new_load = (sample_cov @ beta.T) @ np.linalg.inv(ezz)
    new_psi = np.maximum(np.diag(sample_cov) - np.diag(new_load @ ezx), noise_floor)
    return new_load, new_psi


def _fa_loglikelihood_sufficient(
    sample_cov: np.ndarray, n: int, load: np.ndarray, psi: np.ndarray
) -> float:
    """Full-covariance Gaussian LL evaluated from CxC sample covariance."""
    invpsi = 1.0 / psi
    m = np.eye(load.shape[1]) + load.T @ (invpsi[:, None] * load)
    sign, logdet_m = np.linalg.slogdet(m)
    if sign <= 0:
        return -np.inf
    beta = np.linalg.solve(m, (load * invpsi[:, None]).T)
    # trace[(Psi + CC')^-1 S] by Woodbury, with no NxC intermediates.
    trace_precision_s = np.sum(np.diag(sample_cov) * invpsi) - np.trace(
        beta @ sample_cov @ (invpsi[:, None] * load)
    )
    per_sample = (
        load.shape[0] * np.log(2 * np.pi)
        + np.log(psi).sum()
        + logdet_m
        + trace_precision_s
    )
    return float(-0.5 * n * per_sample)


def causal_lags(
    features: np.ndarray, endpoints: np.ndarray, history_bins: int = 10
) -> np.ndarray:
    """Newest-to-oldest causal feature rows with literal-zero left padding."""
    a = _array2(features, "features")
    ep = np.asarray(endpoints)
    if ep.ndim != 1 or not np.issubdtype(ep.dtype, np.integer):
        raise ValueError("endpoints must be a one-dimensional integer array")
    if history_bins < 1:
        raise ValueError("history_bins is the total bin count and must be at least one")
    if np.any(ep < 0) or np.any(ep >= len(a)):
        raise ValueError("endpoints must lie within the feature stream")
    out = np.zeros((len(ep), a.shape[1] * history_bins), dtype=np.float64)
    for j in range(history_bins):
        indices = ep - j
        valid = indices >= 0
        out[valid, j * a.shape[1] : (j + 1) * a.shape[1]] = a[indices[valid]]
    return out


@dataclass
class RidgeReadout:
    coef_: np.ndarray
    intercept_: np.ndarray
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = _array2(x)
        if a.shape[1] != self.coef_.shape[0]:
            raise ValueError("x feature count differs from fitted ridge readout")
        return a @ self.coef_ + self.intercept_


def fit_ridge_grid(
    x: np.ndarray, y: np.ndarray, alphas: Iterable[float]
) -> dict[float, RidgeReadout]:
    """Fit sklearn-Ridge-equivalent intercept models, sharing one Gram eigensystem."""
    a, target = _array2(x, "x"), _array2(y, "y")
    if len(a) != len(target) or len(a) == 0:
        raise ValueError("x and y must have equal nonzero observation counts")
    alpha_values = [float(alpha) for alpha in alphas]
    if not alpha_values or any(alpha < 0 for alpha in alpha_values):
        raise ValueError("alphas must be a nonempty iterable of non-negative values")
    mx, my = a.mean(0), target.mean(0)
    xc, yc = a - mx, target - my
    gram = xc.T @ xc
    cross = xc.T @ yc
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    projected = vectors.T @ cross
    result = {}
    for alpha in alpha_values:
        coef = vectors @ (projected / (values[:, None] + alpha))
        result[alpha] = RidgeReadout(coef, my - mx @ coef, alpha)
    return result
