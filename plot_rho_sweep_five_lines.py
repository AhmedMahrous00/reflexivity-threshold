#!/usr/bin/env python3
"""Five-line rho sweep with a matched strongly non-normal subcritical case."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from run_reflexivity_experiments import (
    EPS_LOG,
    FIGSIZE_SINGLE,
    GLOBAL_SEED,
    K,
    MAIN_LINEWIDTH,
    NORM_LOG_LABEL,
    N_RUNS,
    REFERENCE_LINEWIDTH,
    THRESHOLD_COLOR,
    configure_matplotlib,
    generate_sparse_random,
    save_figure,
    simulate_normalized_curve,
    upper_triangular_nonnormal_matrix,
)


OUTPUT_DIR = Path("figures")
OUTPUT_STEM = "fig_rho_sweep_five_lines"
NORMAL_RHOS = np.array([0.50, 0.99, 1.01, 1.50], dtype=float)
NONNORMAL_RHO = 0.50
NONNORMAL_RATIO = 10.0
T_MAX = 200


def run_five_line_sweep(
    k: int = K,
    n_runs: int = N_RUNS,
    t_max: int = T_MAX,
    seed: int = GLOBAL_SEED + 11,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = [rf"$\rho(\Gamma)={rho:.2f}$" for rho in NORMAL_RHOS]
    labels.append(rf"$\rho(\Gamma)={NONNORMAL_RHO:.2f}$, non-normal")

    logs = np.empty((len(labels), n_runs, t_max + 1))
    realized_rhos = np.empty((len(labels), n_runs))

    for idx, target_rho in enumerate(NORMAL_RHOS):
        out = simulate_normalized_curve(
            lambda r: generate_sparse_random(k, 0.15, r, weight_law="uniform"),
            float(target_rho),
            rng,
            n_runs,
            t_max,
            desc=f"five-line rho sweep {target_rho:.2f}",
        )
        logs[idx], _, realized_rhos[idx], _, _, _ = out

    nonnormal_base = upper_triangular_nonnormal_matrix(
        k=k,
        target_rho=NONNORMAL_RHO,
        target_ratio=NONNORMAL_RATIO,
        dense=True,
    )
    logs[-1] = random_nonnormal_curves(nonnormal_base, rng, n_runs, t_max)
    realized_rhos[-1] = NONNORMAL_RHO

    return labels, logs, realized_rhos, np.arange(t_max + 1)


def random_nonnormal_curves(
    gamma: np.ndarray,
    rng: np.random.Generator,
    n_runs: int,
    t_max: int,
) -> np.ndarray:
    states = rng.normal(size=(n_runs, gamma.shape[0]))
    initial_norms = np.linalg.norm(states, axis=1)
    states /= initial_norms[:, None]

    logs = np.empty((n_runs, t_max + 1), dtype=float)
    logs[:, 0] = 0.0
    for t in range(1, t_max + 1):
        states = states @ gamma.T
        normalized_dist = np.linalg.norm(states, axis=1)
        logs[:, t] = np.log10(normalized_dist + EPS_LOG)
    return logs


def plot_five_line_sweep(
    labels: list[str],
    logs: np.ndarray,
    t_grid: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = np.mean(logs, axis=1)
    q25 = np.percentile(logs, 25, axis=1)
    q75 = np.percentile(logs, 75, axis=1)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    normal_colors = ["#0072B2", "#009E73", "#CC79A7", "#000000"]
    for idx, label in enumerate(labels[:-1]):
        ax.plot(t_grid, mean[idx], color=normal_colors[idx], linewidth=MAIN_LINEWIDTH, label=label)
        ax.fill_between(t_grid, q25[idx], q75[idx], color=normal_colors[idx], alpha=0.08, linewidth=0)

    ax.plot(
        t_grid,
        mean[-1],
        color="#E69F00",
        linewidth=MAIN_LINEWIDTH,
        linestyle="--",
        marker="^",
        markevery=10,
        markersize=4.0,
        label=labels[-1],
    )
    ax.fill_between(t_grid, q25[-1], q75[-1], color="#E69F00", alpha=0.08, linewidth=0)

    ax.axhline(
        0.0,
        color=THRESHOLD_COLOR,
        linewidth=REFERENCE_LINEWIDTH,
        linestyle="--",
        alpha=0.75,
        label="initial distance",
    )
    ax.set_ylim(-10.0, 10.0)
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(NORM_LOG_LABEL)
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.04, 1.0),
        ncol=2,
        handlelength=1.8,
        columnspacing=0.9,
        fontsize=8.0,
    )
    ax.tick_params(axis="both")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    save_figure(fig, str(output_dir / OUTPUT_STEM))
    plt.close(fig)


def save_outputs(
    labels: list[str],
    logs: np.ndarray,
    realized_rhos: np.ndarray,
    t_grid: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = np.mean(logs, axis=1)
    csv_data = np.column_stack((t_grid, mean.T))
    np.savetxt(
        output_dir / f"{OUTPUT_STEM}.csv",
        csv_data,
        delimiter=",",
        header="t," + ",".join(label.replace(",", ";") for label in labels),
        comments="",
        fmt="%.12g",
    )
    np.savez_compressed(
        output_dir / f"{OUTPUT_STEM}.npz",
        labels=np.array(labels),
        logs=logs,
        realized_rhos=realized_rhos,
        t_grid=t_grid,
        epsilon_log=np.array(EPS_LOG),
    )


def main() -> None:
    configure_matplotlib()
    labels, logs, realized_rhos, t_grid = run_five_line_sweep()
    plot_five_line_sweep(labels, logs, t_grid)
    save_outputs(labels, logs, realized_rhos, t_grid)

    print("Five-line rho sweep figure")
    for label, rhos in zip(labels, realized_rhos, strict=True):
        print(f"  {label}: realized rho mean={np.mean(rhos):.6g}")
    print(f"  saved: {OUTPUT_DIR / f'{OUTPUT_STEM}.png'}")
    print(f"  saved: {OUTPUT_DIR / f'{OUTPUT_STEM}.pdf'}")
    print(f"  saved: {OUTPUT_DIR / f'{OUTPUT_STEM}.csv'}")
    print(f"  saved: {OUTPUT_DIR / f'{OUTPUT_STEM}.npz'}")


if __name__ == "__main__":
    main()
