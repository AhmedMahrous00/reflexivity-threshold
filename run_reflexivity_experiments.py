#!/usr/bin/env python3
"""Shared helpers for the released synthetic simulation artifact.

The two figure scripts import only the constants and routines in this file.
They implement linear comparison dynamics of the form

    w_{t+1} = b + Gamma w_t,

where Gamma is a nonnegative reflexivity matrix rescaled to a requested
spectral radius.
"""

from __future__ import annotations

from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


GLOBAL_SEED = 20260428
K = 100
N_RUNS = 100
T = 100
EPS_LOG = 1e-12
MIN_RHO = 1e-12
TAU = 1e-4
COND_THRESHOLD = 1e10

FIGSIZE_SINGLE = (5.0, 3.35)
QUAL_COLORS = (
    "#2a788e",
    "#7ad151",
    "#440154",
    "#22a884",
    "#fde725",
    "#414487",
)
THRESHOLD_COLOR = "0.20"
REFERENCE_LINEWIDTH = 1.0
MAIN_LINEWIDTH = 1.8
NORM_LOG_LABEL = (
    r"$\log_{10}\left(\|w_t-w^\star\|_2/\|w_0-w^\star\|_2\right)$"
)


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, **kwargs)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "mathtext.fontset": "cm",
            "axes.prop_cycle": mpl.cycler(color=QUAL_COLORS),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def spectral_radius(matrix: np.ndarray) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    if np.allclose(matrix, np.triu(matrix)) or np.allclose(matrix, np.tril(matrix)):
        return float(np.max(np.abs(np.diag(matrix))))
    if matrix.shape[0] > 2:
        try:
            values, vectors = spla.eigs(
                matrix,
                k=1,
                which="LM",
                return_eigenvectors=True,
                tol=1e-10,
                maxiter=2000,
            )
            eigenvalue = values[0]
            eigenvector = vectors[:, 0]
            denom = np.linalg.norm(matrix, ord="fro") * np.linalg.norm(eigenvector) + 1e-300
            residual = np.linalg.norm(matrix @ eigenvector - eigenvalue * eigenvector) / denom
            rho = float(abs(eigenvalue))
            if np.isfinite(rho) and np.isfinite(residual) and residual < 1e-7:
                return rho
        except (
            spla.ArpackNoConvergence,
            spla.ArpackError,
            ValueError,
            FloatingPointError,
        ):
            pass
    values = la.eigvals(matrix, check_finite=False)
    return float(np.max(np.abs(values)))


def rescale_to_rho(matrix: np.ndarray, target_rho: float) -> tuple[np.ndarray, float]:
    rho = spectral_radius(matrix)
    if not np.isfinite(rho) or rho <= MIN_RHO:
        raise ValueError("base matrix has zero or numerically tiny spectral radius")
    gamma = target_rho * matrix / rho
    return np.maximum(gamma, 0.0), float(target_rho)


def generate_weighted_from_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    law: str,
) -> np.ndarray:
    n_edges = int(mask.sum())
    matrix = np.zeros(mask.shape, dtype=float)
    if n_edges == 0:
        return matrix
    if law == "equal":
        weights = np.ones(n_edges)
    elif law == "uniform":
        weights = rng.uniform(0.0, 1.0, size=n_edges)
    else:
        raise ValueError(f"unknown weight law: {law}")
    matrix[mask] = weights
    return matrix


def generate_sparse_random(
    k: int,
    p: float,
    rng: np.random.Generator,
    weight_law: str = "uniform",
) -> np.ndarray:
    mask = rng.random((k, k)) < p
    np.fill_diagonal(mask, False)
    return generate_weighted_from_mask(mask, rng, weight_law)


def sample_state(k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    scale = 1.0 / np.sqrt(k)
    b = rng.normal(0.0, scale, size=k)
    w0 = rng.normal(0.0, scale, size=k)
    return b, w0


def safe_solve_fixed_point(
    gamma: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    matrix = np.eye(gamma.shape[0]) - gamma
    try:
        cond = float(np.linalg.cond(matrix))
        if not np.isfinite(cond) or cond >= COND_THRESHOLD:
            return None, False
        w_star = la.solve(matrix, b, assume_a="gen", check_finite=False)
    except (la.LinAlgError, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None, False
    return (w_star, True) if np.all(np.isfinite(w_star)) else (None, False)


def run_dynamics(
    gamma: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    t_max: int,
    w_star: np.ndarray,
) -> np.ndarray:
    w = w0.copy()
    distances = np.empty(t_max + 1)
    for t in range(t_max + 1):
        distances[t] = float(np.linalg.norm(w - w_star, ord=2))
        if t < t_max:
            w = b + gamma @ w
    return distances


def normalized_distance(distances: np.ndarray) -> np.ndarray:
    denom = max(float(distances[0]), EPS_LOG)
    return distances / denom


def iterations_to_tolerance(norm_dist: np.ndarray, tau: float = TAU) -> float:
    hits = np.flatnonzero(norm_dist <= tau)
    return float(hits[0]) if hits.size else float(norm_dist.size)


def simulate_normalized_curve(
    matrix_generator: Callable[[np.random.Generator], np.ndarray],
    target_rho: float,
    rng: np.random.Generator,
    n_runs: int,
    t_max: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    logs = np.empty((n_runs, t_max + 1))
    norm_dists = np.empty((n_runs, t_max + 1))
    rhos = np.empty(n_runs)
    t_tau = np.empty(n_runs)
    regenerated = 0
    skipped = 0
    attempts = 0
    max_attempts = max(1000, 100 * n_runs)

    for run in progress(range(n_runs), desc=desc, leave=False):
        accepted = False
        while not accepted:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"too many failed attempts in {desc}")
            try:
                base = matrix_generator(rng)
                if np.min(base) < -1e-14:
                    raise ValueError("generated matrix has negative entries")
                gamma, realized = rescale_to_rho(base, target_rho)
                b, w0 = sample_state(gamma.shape[0], rng)
                w_star, ok = safe_solve_fixed_point(gamma, b)
                if not ok or w_star is None:
                    regenerated += 1
                    continue
                distances = run_dynamics(gamma, b, w0, t_max, w_star)
                nd = normalized_distance(distances)
            except (ValueError, la.LinAlgError, np.linalg.LinAlgError):
                regenerated += 1
                continue
            norm_dists[run] = nd
            logs[run] = np.log10(nd + EPS_LOG)
            rhos[run] = realized
            t_tau[run] = iterations_to_tolerance(nd)
            accepted = True
    return logs, norm_dists, rhos, t_tau, regenerated, skipped


def upper_triangular_nonnormal_matrix(
    k: int,
    target_rho: float,
    target_ratio: float,
    dense: bool,
) -> np.ndarray:
    if dense:
        upper = np.triu(np.ones((k, k), dtype=float), k=1)
    else:
        upper = np.zeros((k, k), dtype=float)
        upper[0, -1] = 1.0

    def ratio_for(scale: float) -> float:
        matrix = target_rho * np.eye(k) + scale * upper
        return float(la.norm(matrix, ord=2) / target_rho)

    lo = 0.0
    hi = 1.0
    while ratio_for(hi) < target_ratio:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ratio_for(mid) < target_ratio:
            lo = mid
        else:
            hi = mid
    return target_rho * np.eye(k) + 0.5 * (lo + hi) * upper


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
