#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for exporting and plotting RSL-RL training runs."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "move_cube_rgbd"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
# Task-family folder name under ``results/`` and ``model/`` (matches ``src/thesis_rl/tasks/<name>``).
DEFAULT_TASK = "move_cube"

# Optional map from task family → default RSL-RL experiment log folder name.
TASK_LOG_EXPERIMENT = {
    "move_cube": "move_cube_rgbd",
    "move_cube_detection": "move_cube_detection",
    "pick_cube": "pick_cube",
    "pick_cube_final_rb": "pick_cube_final_rb",
    "pick_place_cube": "pick_place_cube",
    "pick_cube_depth": "pick_cube_depth",
    "pick_cube_final": "pick_cube_final",
    "pick_cube_final_dual": "pick_cube_final_dual",
    "pick_cube_final_eye": "pick_cube_final_eye",
}

# Gymnasium env id used by ``isaaclab train`` / ``isaaclab play``.
TASK_GYM_ID = {
    "move_cube": "ThesisRl-Move-Cube-UR5E",
    "move_cube_detection": "ThesisRl-Move-Cube-Detection-UR5E",
    "pick_cube": "ThesisRl-Pick-Cube-UR5E",
    "pick_cube_final_rb": "ThesisRl-Pick-Cube-Final-Rb-UR5E",
    "pick_place_cube": "ThesisRl-Pick-Place-Cube-UR5E",
    "pick_cube_depth": "ThesisRl-Pick-Cube-Depth-UR5E",
    "pick_cube_final": "ThesisRl-Pick-Cube-Final-UR5E",
    "pick_cube_final_dual": "ThesisRl-Pick-Cube-Final-Dual-UR5E",
    "pick_cube_final_eye": "ThesisRl-Pick-Cube-Final-Eye-UR5E",
}


def default_log_root(task: str = DEFAULT_TASK) -> Path:
    """Return ``logs/rsl_rl/<experiment>/`` for a task family."""
    experiment = TASK_LOG_EXPERIMENT.get(task, task)
    return REPO_ROOT / "logs" / "rsl_rl" / experiment


def task_results_dir(
    task: str = DEFAULT_TASK,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Return ``results/<task>/`` for a task family."""
    safe = task.strip().replace(" ", "_")
    if not safe or "/" in safe or "\\" in safe or safe in {".", ".."}:
        raise ValueError(f"Invalid task name for results path: {task!r}")
    return results_root / safe


def gym_id_for_task(task: str = DEFAULT_TASK) -> str:
    """Return the Gymnasium env id for a task family folder name."""
    if task in TASK_GYM_ID:
        return TASK_GYM_ID[task]
    raise ValueError(f"No Gymnasium id registered for task family {task!r}")


def task_from_gym_id(gym_id: str) -> str:
    """Return the task-family folder for a Gymnasium env id."""
    name = gym_id.split(":")[-1]
    for task, registered in TASK_GYM_ID.items():
        if registered == name:
            return task
    raise ValueError(f"No task family registered for Gymnasium id {gym_id!r}")


def testing_dir_for_training_id(
    training_id: str,
    task: str = DEFAULT_TASK,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Return ``results/<task>/testing_N`` matching ``training_N``."""
    match = re.fullmatch(r"training_(\d+)", training_id)
    if match:
        name = f"testing_{match.group(1)}"
    elif re.fullmatch(r"testing_(\d+)", training_id):
        name = training_id
    else:
        raise ValueError(f"Expected training_N id, got {training_id!r}")
    out = task_results_dir(task, results_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_project_checkpoint(task: str, training_id: str) -> Path | None:
    """Return ``model/<task>/training_N.pt`` if it exists."""
    path = REPO_ROOT / "model" / task / f"{training_id}.pt"
    return path if path.is_file() else None


def find_latest_log_dir(log_root: Path = DEFAULT_LOG_ROOT) -> Path:
    """Return the most recently modified run directory under ``log_root``."""
    if not log_root.is_dir():
        raise FileNotFoundError(f"Log root not found: {log_root}")

    candidates = [p for p in log_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories under: {log_root}")

    # Prefer dirs that contain TensorBoard events.
    with_events = []
    for path in candidates:
        if any(path.glob("events.out.tfevents.*")):
            with_events.append(path)
    pool = with_events or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def next_training_dir(task_dir: Path) -> Path:
    """Allocate ``results/<task>/training_N`` with the next free integer N.

    Reuses an existing empty ``training_N`` directory if present.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    existing: list[int] = []
    empty: list[Path] = []
    for path in task_dir.iterdir():
        match = re.fullmatch(r"training_(\d+)", path.name)
        if not (path.is_dir() and match):
            continue
        n = int(match.group(1))
        existing.append(n)
        if not any(path.iterdir()):
            empty.append(path)
    if empty:
        return min(empty, key=lambda p: int(re.fullmatch(r"training_(\d+)", p.name).group(1)))  # type: ignore[union-attr]
    next_id = (max(existing) + 1) if existing else 1
    out = task_dir / f"training_{next_id}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def find_existing_export_for_log(
    log_dir: Path,
    task: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Path | None:
    """Return ``results/<task>/training_N`` if this log dir was already exported."""
    task_dir = task_results_dir(task, results_root)
    if not task_dir.is_dir():
        return None
    target = str(log_dir.resolve())
    for result_path in task_dir.glob("training_*/result.json"):
        try:
            with result_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            source = payload.get("meta", {}).get("source_log_dir")
            if source and Path(source).resolve() == Path(target).resolve():
                return result_path.parent
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return None


def find_latest_result_json(
    task: str | None = DEFAULT_TASK,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Return the newest ``result.json`` under a task (or all tasks if ``task`` is None)."""
    if task is None:
        pattern = "*/training_*/result.json"
        search_root = results_root
    else:
        pattern = "training_*/result.json"
        search_root = task_results_dir(task, results_root)

    runs = sorted(search_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No exported runs under {search_root}")
    return runs[-1]


def load_tensorboard_scalars(log_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Load all scalar series from a TensorBoard event directory.

    Returns
    -------
    dict
        ``{tag: [{"iteration": int, "wall_time": float, "value": float}, ...]}``
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise ImportError(
            "tensorboard is required to read training logs. "
            "Install it in the training environment (usually already present with Isaac Lab)."
        ) from exc

    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    series: dict[str, list[dict[str, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        points = []
        for event in ea.Scalars(tag):
            points.append(
                {
                    "iteration": int(event.step),
                    "wall_time": float(event.wall_time),
                    "value": float(event.value),
                }
            )
        series[tag] = points
    return series


def series_to_iterations(
    series: dict[str, list[dict[str, float]]],
) -> list[dict[str, Any]]:
    """Merge per-tag series into a list of per-iteration metric dicts."""
    by_iter: dict[int, dict[str, Any]] = {}
    for tag, points in series.items():
        for point in points:
            it = int(point["iteration"])
            row = by_iter.setdefault(it, {"iteration": it})
            row[tag] = point["value"]
            # Keep one wall_time reference if present.
            if "wall_time" not in row:
                row["wall_time"] = point["wall_time"]
    return [by_iter[k] for k in sorted(by_iter)]


def compute_summary(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute compact summary statistics for the run."""
    if not iterations:
        return {}

    def values(key: str) -> list[tuple[int, float]]:
        out = []
        for row in iterations:
            if key in row:
                out.append((int(row["iteration"]), float(row[key])))
        return out

    summary: dict[str, Any] = {
        "num_iterations": len(iterations),
        "first_iteration": iterations[0]["iteration"],
        "last_iteration": iterations[-1]["iteration"],
    }

    reward_key = "Train/mean_reward"
    rewards = values(reward_key)
    if rewards:
        best_it, best_val = max(rewards, key=lambda x: x[1])
        summary["final_mean_reward"] = rewards[-1][1]
        summary["final_mean_reward_iteration"] = rewards[-1][0]
        summary["best_mean_reward"] = best_val
        summary["best_mean_reward_iteration"] = best_it

    success_key = "Episode_Termination/success"
    succ = values(success_key)
    if succ:
        summary["final_success_rate"] = succ[-1][1]
        summary["max_success_rate"] = max(v for _, v in succ)
        # Early/late termination success (not Episode_Reward/success — that is a sparse bonus).
        early_n = max(1, len(succ) // 10)
        late_n = max(1, len(succ) // 10)
        e_vals = [v for _, v in succ[:early_n]]
        l_vals = [v for _, v in succ[-late_n:]]
        summary["success_rate_progress"] = {
            "early_mean": sum(e_vals) / len(e_vals),
            "late_mean": sum(l_vals) / len(l_vals),
            "delta": (sum(l_vals) / len(l_vals)) - (sum(e_vals) / len(e_vals)),
        }

    # Prefer Train/mean_reward coverage as the "learning" iteration count when available.
    if rewards:
        summary["num_learning_iterations"] = len(rewards)
        summary["last_learning_iteration"] = rewards[-1][0]

    # Late-vs-early reward-term means. Skip */success reward tags — use termination success above.
    reward_rows = [row for row in iterations if any(k.startswith("Episode_Reward/") for k in row)]
    stage_progress = {}
    if reward_rows:
        n = len(reward_rows)
        early_n = max(1, n // 10)
        late_n = max(1, n // 10)
        early = reward_rows[:early_n]
        late = reward_rows[-late_n:]

        stage_tags = sorted(
            {
                k
                for row in reward_rows
                for k in row
                if k.startswith("Episode_Reward/")
                and not k.endswith("/success")
                and not any(
                    p in k.rsplit("/", 1)[-1].lower()
                    for p in (
                        "penalty",
                        "action_rate",
                        "drop",
                        "failed",
                        "wrong",
                        "unsafe",
                        "collision",
                        "timeout",
                        "off_table",
                        "detection_lost",
                        "out_of_workspace",
                    )
                )
            }
        )
        for tag in stage_tags:
            e_vals = [float(r[tag]) for r in early if tag in r]
            l_vals = [float(r[tag]) for r in late if tag in r]
            if e_vals and l_vals:
                stage_progress[tag] = {
                    "early_mean": sum(e_vals) / len(e_vals),
                    "late_mean": sum(l_vals) / len(l_vals),
                    "delta": (sum(l_vals) / len(l_vals)) - (sum(e_vals) / len(e_vals)),
                }
    if stage_progress:
        summary["stage_reward_progress"] = stage_progress

    return summary


def find_checkpoints(log_dir: Path) -> list[Path]:
    """Return sorted ``model_*.pt`` checkpoints in a log directory."""
    models = list(log_dir.glob("model_*.pt"))

    def sort_key(path: Path) -> int:
        match = re.search(r"model_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    return sorted(models, key=sort_key)


def save_final_project_model(
    log_dir: Path,
    task: str,
    training_id: str,
) -> dict[str, str]:
    """Copy the final checkpoint to ``model/<task>/training_N.pt``.

    Checkpoints remain in the RSL-RL log directory; they are not duplicated
    under ``results/``.
    """
    checkpoints = find_checkpoints(log_dir)
    if not checkpoints:
        return {}

    project_task_dir = REPO_ROOT / "model" / task
    project_task_dir.mkdir(parents=True, exist_ok=True)
    project_dst = project_task_dir / f"{training_id}.pt"
    shutil.copy2(checkpoints[-1], project_dst)
    return {f"project_model/{task}/{training_id}.pt": str(project_dst)}


def build_result_payload(
    log_dir: Path,
    training_dir: Path,
    model_paths: dict[str, str],
    task: str | None = None,
) -> dict[str, Any]:
    """Build the full ``result.json`` payload for a run."""
    series = load_tensorboard_scalars(log_dir)
    iterations = series_to_iterations(series)
    summary = compute_summary(iterations)

    # Prefer explicit task; otherwise infer from ``results/<task>/training_N``.
    resolved_task = task
    if resolved_task is None and training_dir.parent.name not in {"results", ""}:
        resolved_task = training_dir.parent.name

    return {
        "meta": {
            "task": resolved_task,
            "training_id": training_dir.name,
            "source_log_dir": str(log_dir.resolve()),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "num_scalar_tags": len(series),
            "num_iterations": len(iterations),
            "model_files": model_paths,
        },
        "summary": summary,
        "tags": sorted(series.keys()),
        "iterations": iterations,
    }


def write_result_json(payload: dict[str, Any], path: Path) -> None:
    """Write ``result.json`` with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def load_result_json(path: Path) -> dict[str, Any]:
    """Load a previously exported ``result.json``."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
