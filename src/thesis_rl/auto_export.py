# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auto-export training metrics/plots when RSL-RL ``learn()`` finishes.

Installed when ``thesis_rl.tasks`` is imported (Isaac Lab task entry-point), so a
plain ``isaaclab train --task ThesisRl-...`` writes::

    results/<task>/training_N/result.json
    results/<task>/training_N/plots/
    model/<task>/training_N.pt

Set ``THESIS_RL_SKIP_AUTO_EXPORT=1`` to disable export.
Set ``THESIS_RL_SKIP_AUTO_EVAL=1`` to export metrics/model but skip the 100-attempt play eval.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# experiment_name (agent cfg) → results/model folder
EXPERIMENT_TO_TASK = {
    "move_cube_rgbd": "move_cube",
    "move_cube_detection": "move_cube_detection",
    "pick_cube": "pick_cube",
    "pick_cube_final_rb": "pick_cube_final_rb",
    "pick_place_cube": "pick_place_cube",
    "pick_cube_depth": "pick_cube_depth",
    "pick_cube_detection": "pick_cube_detection",
    "pick_cube_final": "pick_cube_final",
    "pick_cube_final_dual": "pick_cube_final_dual",
    "pick_cube_final_eye": "pick_cube_final_eye",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FINALIZE = _REPO_ROOT / "scripts" / "finalize_training_run.py"
_POST_EVAL = _REPO_ROOT / "scripts" / "run_post_training_eval.py"
_INSTALLED = False


def _resolve_script(name: str, fallback: Path) -> Path | None:
    """Locate a repo script even if the package is installed elsewhere."""
    if fallback.is_file():
        return fallback
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "scripts" / name
        if candidate.is_file() and (parent / "src" / "thesis_rl").is_dir():
            return candidate
    return None


def _resolve_finalize_script() -> Path | None:
    """Locate finalize_training_run.py even if the package is installed elsewhere."""
    return _resolve_script("finalize_training_run.py", _FINALIZE)


def _task_for_runner(runner) -> str | None:
    cfg = getattr(runner, "cfg", None) or {}
    experiment = cfg.get("experiment_name")
    if isinstance(experiment, str) and experiment in EXPERIMENT_TO_TASK:
        return EXPERIMENT_TO_TASK[experiment]

    log_dir = _log_dir_for_runner(runner)
    if log_dir is None:
        return None
    # logs/rsl_rl/<experiment>/<timestamp>
    try:
        experiment = Path(log_dir).resolve().parent.name
    except OSError:
        return None
    return EXPERIMENT_TO_TASK.get(experiment)


def _log_dir_for_runner(runner) -> str | None:
    logger = getattr(runner, "logger", None)
    log_dir = getattr(logger, "log_dir", None) if logger is not None else None
    if log_dir:
        return str(log_dir)
    return None


def _should_export(runner) -> bool:
    if os.environ.get("THESIS_RL_SKIP_AUTO_EXPORT", "").strip() in {"1", "true", "True", "yes"}:
        return False
    if os.environ.get("RANK", "0") not in {"0", ""}:
        return False
    if getattr(runner, "is_distributed", False) and getattr(runner, "gpu_global_rank", 0) != 0:
        return False
    return _task_for_runner(runner) is not None and _log_dir_for_runner(runner) is not None


def _should_eval() -> bool:
    return os.environ.get("THESIS_RL_SKIP_AUTO_EVAL", "").strip() not in {"1", "true", "True", "yes"}


def _schedule_eval(task: str, log_dir: str, repo_root: Path) -> None:
    """Start eval after this training process exits (second Isaac Sim session)."""
    if not _should_eval():
        print("[INFO] thesis_rl auto-eval skipped (THESIS_RL_SKIP_AUTO_EVAL)", flush=True)
        return
    waiter = _resolve_script("run_post_training_eval.py", _POST_EVAL)
    if waiter is None:
        print("[WARN] thesis_rl auto-eval: run_post_training_eval.py not found", flush=True)
        return
    results_dir = repo_root / "results" / task
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "auto_eval.log"
    cmd = [
        sys.executable,
        str(waiter),
        "--wait-pid",
        str(os.getpid()),
        "--task",
        task,
        "--log-dir",
        log_dir,
    ]
    print(f"[INFO] thesis_rl auto-eval scheduled: {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n# spawn {cmd}\n")
        log_file.flush()
        subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def export_after_training(runner) -> None:
    """Run finalize for this runner's log directory (best-effort)."""
    if not _should_export(runner):
        return

    task = _task_for_runner(runner)
    log_dir = _log_dir_for_runner(runner)
    assert task is not None and log_dir is not None

    finalize = _resolve_finalize_script()
    if finalize is None:
        print("[WARN] thesis_rl auto-export: finalize_training_run.py not found", flush=True)
        return

    repo_root = finalize.parents[1]
    cmd = [
        sys.executable,
        str(finalize),
        "--task",
        task,
        "--log-dir",
        log_dir,
    ]
    print(f"[INFO] thesis_rl auto-export: {' '.join(cmd)}", flush=True)
    code = subprocess.call(cmd, cwd=str(repo_root))
    if code != 0:
        print(f"[WARN] thesis_rl auto-export failed with exit code {code}", flush=True)
        return
    print(f"[INFO] thesis_rl auto-export finished for task={task}", flush=True)
    _schedule_eval(task, log_dir, repo_root)


def _wrap_learn(cls) -> None:
    if getattr(cls.learn, "_thesis_rl_auto_export", False):
        return

    original = cls.learn

    def learn(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            try:
                export_after_training(self)
            except Exception as exc:  # noqa: BLE001 — never break training teardown
                print(f"[WARN] thesis_rl auto-export error: {exc}", flush=True)

    learn._thesis_rl_auto_export = True  # type: ignore[attr-defined]
    cls.learn = learn  # type: ignore[method-assign]


def install_auto_export_hook() -> None:
    """Monkey-patch RSL-RL runners so export runs after ``learn()``."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        return

    _wrap_learn(OnPolicyRunner)
    try:
        from rsl_rl.runners import DistillationRunner

        _wrap_learn(DistillationRunner)
    except ImportError:
        pass
    _INSTALLED = True
