#!/usr/bin/env python3
"""Iteration-complexity phase transition for best-response retraining."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from run_reflexivity_experiments import generate_sparse_random, rescale_to_rho


SEED = 0
K = 100
N_BASE_MATRICES = 20
N_INITIAL_STATES = 5
T_MAX = 10000
EPSILON = 1e-3
OUTPUT_DIR = Path("figures")
RHO_THRESHOLD = 1.0
RHO_HALF_WIDTH = 0.5
SPARSE_DENSITY = 0.15
DIVERGENCE_NORM = 1e10
PLOT_T_MAX = 1000
FIGSIZE_SINGLE = (5.0, 3.35)

MAIN_COLOR = "#2a8a8a"
SUBCRITICAL_SHADE = "#2a8a8a"
SUPERCRITICAL_SHADE = "#d95f02"
THRESHOLD_COLOR = "0.20"
BAND_ALPHA = 0.18
REFERENCE_LINEWIDTH = 1.0
MAIN_LINEWIDTH = 1.8


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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
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
    return float(np.max(np.abs(la.eigvals(matrix))))


def rho_grid() -> np.ndarray:
    low = np.linspace(0.50, 0.95, 60, endpoint=False)
    mid = np.linspace(0.95, 1.05, 80, endpoint=False)
    high = np.linspace(1.05, 1.50, 60)
    return np.unique(np.concatenate((low, mid, high)))


def perron_right_weights(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = la.eig(matrix, left=False, right=True, check_finite=False)
    idx = int(np.argmax(np.abs(eigvals)))
    weights = np.real(eigvecs[:, idx])
    if np.sum(weights) < 0:
        weights *= -1.0
    weights = np.maximum(weights, 0.0)
    if np.min(weights) <= 1e-10:
        raise ValueError("base matrix did not yield strictly positive Perron weights")
    return weights / np.mean(weights)


def weighted_norm(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.max(np.abs(states) / weights[:, None], axis=0)


def unit_weighted_samples(
    rng: np.random.Generator,
    n: int,
    weights: np.ndarray,
) -> np.ndarray:
    samples = rng.normal(size=(weights.size, n))
    samples /= weighted_norm(samples, weights)[None, :]
    return samples


def base_weighted_norms(
    base: np.ndarray,
    initial_states: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    norms = np.empty((T_MAX + 1, initial_states.shape[1]), dtype=float)
    states = initial_states.copy()
    norms[0] = weighted_norm(states, weights)

    for t in range(1, T_MAX + 1):
        states = base @ states
        norms[t] = np.minimum(weighted_norm(states, weights), DIVERGENCE_NORM)

    return norms


def compute_t_eps_from_base_norms(
    base_norms: np.ndarray,
    rhos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_grid = np.arange(base_norms.shape[0])
    hit_times = np.full((len(rhos), base_norms.shape[1]), T_MAX, dtype=float)
    capped = np.ones_like(hit_times, dtype=bool)

    for idx, rho in enumerate(rhos):
        if rho >= 1.0:
            continue
        scaled_norms = base_norms * np.power(float(rho), t_grid)[:, None]
        hits = scaled_norms <= EPSILON
        any_hit = np.any(hits, axis=0)
        hit_times[idx, any_hit] = np.argmax(hits[:, any_hit], axis=0)
        capped[idx, any_hit] = False
    return hit_times, capped


def run_experiment() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float, float]],
]:
    seed_sequence = np.random.SeedSequence(SEED)
    matrix_seed, trajectory_seed = seed_sequence.spawn(2)
    matrix_rng = np.random.default_rng(matrix_seed)
    trajectory_rng = np.random.default_rng(trajectory_seed)
    rhos = rho_grid()
    all_t_eps = np.empty((len(rhos), N_BASE_MATRICES * N_INITIAL_STATES), dtype=float)
    all_capped = np.empty_like(all_t_eps, dtype=bool)
    check_base: np.ndarray | None = None

    col = 0
    for _ in range(N_BASE_MATRICES):
        accepted = False
        while not accepted:
            candidate = generate_sparse_random(K, SPARSE_DENSITY, matrix_rng, weight_law="uniform")
            try:
                base, _ = rescale_to_rho(candidate, 1.0)
                weights = perron_right_weights(base)
            except ValueError:
                continue
            accepted = True

        if check_base is None:
            check_base = base
        initial_states = unit_weighted_samples(trajectory_rng, N_INITIAL_STATES, weights)
        norms = base_weighted_norms(base, initial_states, weights)
        t_eps, capped = compute_t_eps_from_base_norms(norms, rhos)
        all_t_eps[:, col : col + N_INITIAL_STATES] = t_eps
        all_capped[:, col : col + N_INITIAL_STATES] = capped
        col += N_INITIAL_STATES

    median = np.median(all_t_eps, axis=1)
    q25 = np.percentile(all_t_eps, 25, axis=1)
    q75 = np.percentile(all_t_eps, 75, axis=1)
    cap_fraction = np.mean(all_capped, axis=1)

    check_rhos = [0.5, 0.9, 1.0, 1.1, 1.5]
    checks = []
    assert check_base is not None
    for rho in check_rhos:
        gamma = rho * check_base
        checks.append((rho, spectral_radius(gamma), abs(spectral_radius(gamma) - rho)))

    return rhos, median, q25, q75, cap_fraction, checks


def save_csv(
    rhos: np.ndarray,
    median: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    cap_fraction: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "fig_iteration_complexity.csv"
    data = np.column_stack((rhos, median, q25, q75, cap_fraction))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="rho,t_eps_median,t_eps_q25,t_eps_q75,capped_fraction",
        comments="",
        fmt=["%.12g", "%.12g", "%.12g", "%.12g", "%.12g"],
    )


def plot_figure(rhos: np.ndarray, median: np.ndarray, q25: np.ndarray, q75: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)

    rho_min = RHO_THRESHOLD - RHO_HALF_WIDTH
    rho_max = RHO_THRESHOLD + RHO_HALF_WIDTH
    label_offset = 0.25
    subcritical = rhos < 1.0
    y_max = PLOT_T_MAX
    regime_label_y = 0.72 * y_max

    ax.axvspan(rho_min, RHO_THRESHOLD, color=SUBCRITICAL_SHADE, alpha=0.08, linewidth=0)
    ax.axvspan(RHO_THRESHOLD, rho_max, color=SUPERCRITICAL_SHADE, alpha=0.08, linewidth=0)

    ax.fill_between(
        rhos[subcritical],
        q25[subcritical],
        q75[subcritical],
        color=MAIN_COLOR,
        alpha=BAND_ALPHA,
        linewidth=0,
    )
    ax.plot(rhos[subcritical], median[subcritical], color=MAIN_COLOR, linewidth=MAIN_LINEWIDTH)

    ax.axvline(RHO_THRESHOLD, color=THRESHOLD_COLOR, linewidth=REFERENCE_LINEWIDTH, linestyle="--")
    ax.text(
        1.012,
        0.63 * y_max,
        r"$\rho(\Gamma)=1$",
        color=THRESHOLD_COLOR,
        fontsize=12,
        rotation=90,
        ha="left",
        va="center",
    )
    ax.text(
        RHO_THRESHOLD - label_offset,
        regime_label_y,
        "subcritical:\ncontraction",
        color=MAIN_COLOR,
        fontsize=12,
        ha="center",
        va="center",
    )
    ax.text(
        RHO_THRESHOLD + label_offset,
        regime_label_y,
        "supercritical:\ndivergence",
        color=SUPERCRITICAL_SHADE,
        fontsize=12,
        ha="center",
        va="center",
    )

    ax.set_xlim(rho_min, rho_max)
    ax.set_ylim(0, y_max)
    ax.set_xticks(np.linspace(rho_min, rho_max, 5))
    ax.set_xlabel(r"$\rho(\Gamma)$", fontsize=13)
    ax.set_ylabel(r"$T$", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    fig.savefig(OUTPUT_DIR / "fig_iteration_complexity.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig_iteration_complexity.pdf")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    rhos, median, q25, q75, cap_fraction, checks = run_experiment()
    save_csv(rhos, median, q25, q75, cap_fraction)
    plot_figure(rhos, median, q25, q75)

    print("Iteration-complexity phase-transition figure")
    print(f"  number of rho values: {len(rhos)}")
    print(f"  base matrices: {N_BASE_MATRICES}")
    print(f"  initial states per base: {N_INITIAL_STATES}")
    print(f"  T_max: {T_MAX}")
    print(f"  epsilon: {EPSILON:g}")
    print(f"  sparse density: {SPARSE_DENSITY:g}")
    print(f"  max subcritical capped fraction: {np.max(cap_fraction[rhos < 1.0]):.3g}")
    print("  spectral-radius sanity checks:")
    for rho, observed, error in checks:
        status = "OK" if error <= 1e-10 else "FAIL"
        print(
            f"    rho={rho:.3f}, spectral_radius(Gamma)={observed:.12g}, "
            f"abs_error={error:.3e} [{status}]"
        )
    print(f"  saved: {OUTPUT_DIR / 'fig_iteration_complexity.png'}")
    print(f"  saved: {OUTPUT_DIR / 'fig_iteration_complexity.pdf'}")
    print(f"  saved: {OUTPUT_DIR / 'fig_iteration_complexity.csv'}")


if __name__ == "__main__":
    main()
