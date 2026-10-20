#!/usr/bin/env python3
"""Regenerate the fixed presentation figures from their source simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

import plot_iteration_complexity
import plot_rho_sweep_five_lines


OUTPUTS = (
    Path("figures/fig_iteration_complexity.png"),
    Path("figures/fig_iteration_complexity.pdf"),
    Path("figures/fig_iteration_complexity.csv"),
    Path("figures/fig_rho_sweep_five_lines.png"),
    Path("figures/fig_rho_sweep_five_lines.pdf"),
    Path("figures/fig_rho_sweep_five_lines.csv"),
    Path("figures/fig_rho_sweep_five_lines.npz"),
)


def remove_existing_outputs() -> None:
    for path in OUTPUTS:
        path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate fig_iteration_complexity and fig_rho_sweep_five_lines "
            "under figures/."
        )
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete the existing generated files before rerunning the simulations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clean:
        remove_existing_outputs()

    plot_iteration_complexity.main()
    plot_rho_sweep_five_lines.main()


if __name__ == "__main__":
    main()
