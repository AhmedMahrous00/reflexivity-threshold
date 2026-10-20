#!/usr/bin/env python3
"""Convenience entry point for reproducing the released synthetic figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the released synthetic simulation figures."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete previous generated outputs before rerunning simulations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [sys.executable, "generate_figures.py"]
    if args.clean:
        command.append("--clean")
    subprocess.run(command, cwd=ARTIFACT_ROOT, check=True)


if __name__ == "__main__":
    main()
