#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finalize a training run: export metrics JSON, project model, and plots.

Plots are on by default. Prefer ``scripts/train_and_export.py`` so this runs
automatically after training.

Typical usage::

    python scripts/finalize_training_run.py --task move_cube --latest

Or point at a specific log directory::

    python scripts/finalize_training_run.py \\
        --task move_cube \\
        --log-dir logs/rsl_rl/move_cube_rgbd/<timestamp>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from training_results_lib import DEFAULT_TASK, default_log_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export + plot an RSL-RL training run")
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help=f"Task family folder under results/ and model/ (default: {DEFAULT_TASK})",
    )
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--latest", action="store_true", default=True)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate matplotlib plots (default: on)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots")
    args = parser.parse_args()

    export_script = REPO_ROOT / "scripts" / "export_training_results.py"
    cmd = [sys.executable, str(export_script), "--task", args.task]

    if args.log_dir is not None:
        cmd.extend(["--log-dir", str(args.log_dir)])
    else:
        cmd.append("--latest")

    log_root = args.log_root if args.log_root is not None else default_log_root(args.task)
    cmd.extend(["--log-root", str(log_root)])
    if args.results_root is not None:
        cmd.extend(["--results-root", str(args.results_root)])
    if not args.no_plot:
        cmd.append("--plot")

    print(f"[INFO] Running: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
