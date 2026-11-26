#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export an RSL-RL log directory into ``results/<task>/training_N/``.

Layout created::

    results/<task>/training_N/
      result.json          # per-iteration metrics from TensorBoard
      plots/               # matplotlib figures (when --plot is passed)

Also writes the final policy to::

    model/<task>/training_N.pt

Checkpoints are not copied into ``results/`` (they stay under ``logs/``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as ``python scripts/export_training_results.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_results_lib import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TASK,
    build_result_payload,
    default_log_root,
    find_existing_export_for_log,
    find_latest_log_dir,
    next_training_dir,
    save_final_project_model,
    task_results_dir,
    write_result_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RSL-RL training logs to results/<task>/training_N"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help=f"Task family folder under results/ and model/ (default: {DEFAULT_TASK})",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Path to a specific logs/rsl_rl/.../<timestamp> directory",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent run under the task's experiment log root",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Experiment log root (default: logs/rsl_rl/<task experiment>)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Results root (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also generate matplotlib plots under results/<task>/training_N/plots",
    )
    args = parser.parse_args()

    log_root = args.log_root if args.log_root is not None else default_log_root(args.task)

    if args.log_dir is not None:
        log_dir = args.log_dir.resolve()
    elif args.latest:
        log_dir = find_latest_log_dir(log_root)
    else:
        log_dir = find_latest_log_dir(log_root)

    if not log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {log_dir}")

    existing = find_existing_export_for_log(log_dir, args.task, args.results_root)
    if existing is not None:
        print(f"[INFO] Already exported: {existing}")
        print(f"[INFO] Source log dir : {log_dir}")
        if args.plot:
            result_path = existing / "result.json"
            if result_path.is_file():
                from plot_training_results import generate_all_plots

                plots_dir = generate_all_plots(result_path)
                print(f"[INFO] Wrote plots    : {plots_dir}")
        print(f"[INFO] Done: {existing}")
        return

    training_dir = next_training_dir(task_results_dir(args.task, args.results_root))

    print(f"[INFO] Task          : {args.task}")
    print(f"[INFO] Source log dir : {log_dir}")
    print(f"[INFO] Exporting to   : {training_dir}")

    model_paths = save_final_project_model(
        log_dir,
        task=args.task,
        training_id=training_dir.name,
    )
    if not model_paths:
        print(
            "[WARN] No model_*.pt checkpoints found in the log directory. "
            "Training may have stopped before the first save. "
            "result.json will still be written if TensorBoard events exist."
        )

    payload = build_result_payload(log_dir, training_dir, model_paths, task=args.task)
    result_path = training_dir / "result.json"
    write_result_json(payload, result_path)
    print(f"[INFO] Wrote metrics  : {result_path}")
    print(f"[INFO] Iterations     : {payload['meta']['num_iterations']}")
    print(f"[INFO] Scalar tags    : {payload['meta']['num_scalar_tags']}")
    project_key = f"project_model/{args.task}/{training_dir.name}.pt"
    if project_key in model_paths:
        print(f"[INFO] Project model  : {model_paths[project_key]}")

    if args.plot:
        from plot_training_results import generate_all_plots

        plots_dir = generate_all_plots(result_path)
        print(f"[INFO] Wrote plots    : {plots_dir}")

    print(f"[INFO] Done: {training_dir}")


if __name__ == "__main__":
    main()
