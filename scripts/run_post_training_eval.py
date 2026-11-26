#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Wait for a training process to exit, then run policy evaluation.

Spawned by ``thesis_rl.auto_export`` so Isaac Sim is fully shut down before a
second Kit/PhysX session starts. Stdlib-only until it execs ``evaluate_policy.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from training_results_lib import (  # noqa: E402
    DEFAULT_TASK,
    find_existing_export_for_log,
    find_project_checkpoint,
    gym_id_for_task,
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_pid(pid: int, extra_s: float = 8.0) -> None:
    print(f"[INFO] Waiting for training pid {pid} to exit...", flush=True)
    while _pid_alive(pid):
        time.sleep(2.0)
    if extra_s > 0:
        time.sleep(extra_s)
    print(f"[INFO] Training pid {pid} exited; starting evaluation.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training 100-attempt policy evaluation")
    parser.add_argument("--wait-pid", type=int, default=None, help="Wait until this PID exits")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--num-attempts", type=int, default=100)
    args = parser.parse_args()

    if args.wait_pid is not None:
        _wait_for_pid(args.wait_pid)

    export_dir = find_existing_export_for_log(args.log_dir.resolve(), args.task)
    if export_dir is None:
        raise SystemExit(f"No exported training run found for log dir: {args.log_dir}")
    training_id = export_dir.name
    checkpoint = find_project_checkpoint(args.task, training_id)
    if checkpoint is None:
        raise SystemExit(f"No project checkpoint for {args.task}/{training_id}")

    eval_script = REPO_ROOT / "scripts" / "evaluate_policy.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--task",
        gym_id_for_task(args.task),
        "--checkpoint",
        str(checkpoint),
        "--training-id",
        training_id,
        "--num_envs",
        "1",
        "--num-attempts",
        str(args.num_attempts),
        "physics=isaacsim_physx",
    ]
    # Current Isaac Lab: cameras auto-enable for Kit camera sensors; headless via env.
    env = {**os.environ, "HEADLESS": os.environ.get("HEADLESS", "1")}
    print(f"[INFO] Running: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT), env=env))


if __name__ == "__main__":
    main()
