#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate reward-ablation eval runs into shared comparison plots.

Reads ``results/<task>/testing_*/{result.json,rewards.txt}`` and writes figures
under ``results/<task>/ablation/``. Every figure includes all configs together
(accuracy is the primary metric; reward weights label which config produced it).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from training_results_lib import DEFAULT_RESULTS_ROOT, DEFAULT_TASK, task_results_dir

# Canonical reward-weight order for heatmaps / tables (move_cube + pick_cube).
_REWARD_KEYS = (
    # move_cube
    "APPROACH",
    "CONTACT",
    "PUSH",
    "SUCCESS",
    "STEP_PENALTY",
    "ACTION_RATE",
    # pick_cube / pick_cube_final
    "REACHING",
    "GRASPING",
    "TO_GOAL",
    "JOINT_VELOCITY",
    "DROPPING",
)


@dataclass(frozen=True)
class AblationRun:
    testing_id: str
    training_id: str | None
    checkpoint: str | None
    accuracy: float
    num_success: int
    num_attempts: int
    mean_completion_time_s: float | None
    mean_episode_time_s: float | None
    rewards: dict[str, float]
    label: str


def _parse_rewards_txt(path: Path) -> dict[str, float]:
    """Parse reward weights from a rewards.txt snapshot.

    Supports both move_cube style::

        APPROACH = 1.0

    and pick_cube ``RewardsCfg`` / RewTerm snapshots::

        reaching = RewTerm(func=mdp.reach_cube, weight=1.0)
    """
    text = path.read_text(encoding="utf-8")
    rewards: dict[str, float] = {}

    for match in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE):
        name = match.group(1)
        raw = match.group(2).split("#", 1)[0].strip()
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, (int, float)):
            rewards[name] = float(value)

    # RewTerm lines (indent-tolerant):  reaching = RewTerm(..., weight=1.0)
    for match in re.finditer(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*RewTerm\([^)]*weight\s*=\s*([^,\)]+)",
        text,
        flags=re.MULTILINE,
    ):
        name = match.group(1).upper()
        raw = match.group(2).strip()
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, (int, float)):
            rewards[name] = float(value)

    return rewards


def _primary_reward_keys(rewards: dict[str, float]) -> list[str]:
    """Prefer canonical stage terms; fall back to positive weights present."""
    primary_candidates = (
        "APPROACH",
        "CONTACT",
        "PUSH",
        "REACHING",
        "GRASPING",
        "TO_GOAL",
        "SUCCESS",
    )
    keys = [k for k in primary_candidates if k in rewards]
    if keys:
        return keys
    return [k for k, v in rewards.items() if v > 0]


def _config_label(testing_id: str, rewards: dict[str, float]) -> str:
    """Short multi-line label: testing id + key reward weights."""
    parts = [testing_id]
    for key in _REWARD_KEYS:
        if key in rewards:
            parts.append(f"{key}={rewards[key]:g}")
    for key in sorted(k for k in rewards if k not in _REWARD_KEYS):
        parts.append(f"{key}={rewards[key]:g}")
    return "\n".join(parts)


def _short_config_label(testing_id: str, rewards: dict[str, float]) -> str:
    """Compact axis label: testing id + key weights."""
    bits = [f"{k[:3]}={rewards[k]:g}" for k in _primary_reward_keys(rewards)[:4]]
    reg = []
    for key, tag in (
        ("STEP_PENALTY", "SP"),
        ("ACTION_RATE", "AR"),
        ("JOINT_VELOCITY", "JV"),
        ("DROPPING", "DR"),
    ):
        if key in rewards:
            reg.append(f"{tag}={rewards[key]:g}")
    tag = ", ".join(bits + reg)
    return f"{testing_id}\n({tag})" if tag else testing_id


def discover_testing_dirs(task_dir: Path, testing_ids: list[str] | None) -> list[Path]:
    if testing_ids:
        dirs = [task_dir / tid for tid in testing_ids]
        missing = [str(p) for p in dirs if not p.is_dir()]
        if missing:
            raise SystemExit(f"Missing testing folders:\n  " + "\n  ".join(missing))
        return dirs
    dirs = sorted(
        (p for p in task_dir.glob("testing_*") if p.is_dir() and re.fullmatch(r"testing_\d+", p.name)),
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    if not dirs:
        raise SystemExit(f"No testing_* folders under {task_dir}")
    return dirs


def load_ablation_runs(testing_dirs: list[Path]) -> list[AblationRun]:
    runs: list[AblationRun] = []
    for testing_dir in testing_dirs:
        result_path = testing_dir / "result.json"
        rewards_path = testing_dir / "rewards.txt"
        if not result_path.is_file():
            raise SystemExit(f"Missing {result_path}")
        if not rewards_path.is_file():
            raise SystemExit(f"Missing {rewards_path} (needed to label reward configs)")

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        meta = payload.get("meta", {})
        rewards = _parse_rewards_txt(rewards_path)
        if not rewards:
            raise SystemExit(f"Could not parse reward weights from {rewards_path}")

        testing_id = str(meta.get("testing_id") or testing_dir.name)
        runs.append(
            AblationRun(
                testing_id=testing_id,
                training_id=meta.get("training_id"),
                checkpoint=meta.get("checkpoint"),
                accuracy=float(summary["accuracy"]),
                num_success=int(summary.get("num_success", 0)),
                num_attempts=int(summary.get("num_attempts", 0)),
                mean_completion_time_s=(
                    None
                    if summary.get("mean_completion_time_s") is None
                    else float(summary["mean_completion_time_s"])
                ),
                mean_episode_time_s=(
                    None
                    if summary.get("mean_episode_time_s") is None
                    else float(summary["mean_episode_time_s"])
                ),
                rewards=rewards,
                label=_short_config_label(testing_id, rewards),
            )
        )
    return runs


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {path}", flush=True)


def plot_accuracy_bars(runs: list[AblationRun], out_dir: Path) -> None:
    """Grouped accuracy bar chart; xticklabels carry the reward configuration."""
    labels = [r.label for r in runs]
    accuracies = np.array([100.0 * r.accuracy for r in runs], dtype=np.float64)
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(runs)))

    fig, ax = plt.subplots(figsize=(max(9.0, 2.8 * len(runs)), 6.0))
    x = np.arange(len(runs))
    bars = ax.bar(x, accuracies, color=colors, edgecolor="#222222", linewidth=0.6, width=0.72)
    for bar, run, acc in zip(bars, runs, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            f"{acc:.1f}%\n({run.num_success}/{run.num_attempts})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0.0, min(115.0, max(100.0, float(accuracies.max()) + 18.0)))
    ax.set_title("Reward ablation — accuracy by configuration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(float(accuracies.max()), color="#2e7d32", linestyle="--", linewidth=1.2, alpha=0.8)
    _save_fig(fig, out_dir / "01_accuracy_by_config.png")


def plot_reward_heatmap(runs: list[AblationRun], out_dir: Path) -> None:
    """Heatmap of reward weights (rows) × configs (cols), with accuracy on the top."""
    keys = [k for k in _REWARD_KEYS if any(k in r.rewards for r in runs)]
    # Include any unexpected keys from files so nothing is dropped.
    extra = sorted({k for r in runs for k in r.rewards if k not in keys})
    keys = keys + extra
    mat = np.full((len(keys), len(runs)), np.nan, dtype=np.float64)
    for j, run in enumerate(runs):
        for i, key in enumerate(keys):
            if key in run.rewards:
                mat[i, j] = run.rewards[key]

    accuracies = np.array([100.0 * r.accuracy for r in runs], dtype=np.float64)
    col_labels = [
        f"{r.testing_id}\nacc={100.0 * r.accuracy:.1f}%\n({r.num_success}/{r.num_attempts})" for r in runs
    ]

    fig, (ax_acc, ax) = plt.subplots(
        2,
        1,
        figsize=(max(8.5, 2.6 * len(runs) + 2.0), 1.2 + 0.55 * len(keys) + 1.4),
        gridspec_kw={"height_ratios": [0.9, max(2.8, 0.45 * len(keys))], "hspace": 0.18},
    )

    # Accuracy strip (same columns as the weight heatmap).
    acc_row = accuracies.reshape(1, -1)
    im_acc = ax_acc.imshow(acc_row, aspect="auto", cmap="YlGn", vmin=0.0, vmax=100.0)
    ax_acc.set_yticks([0])
    ax_acc.set_yticklabels(["accuracy (%)"])
    ax_acc.set_xticks(range(len(runs)))
    ax_acc.set_xticklabels(col_labels, fontsize=8)
    ax_acc.xaxis.tick_top()
    ax_acc.xaxis.set_label_position("top")
    for j, acc in enumerate(accuracies):
        ax_acc.text(j, 0, f"{acc:.1f}", ha="center", va="center", fontsize=10, fontweight="bold")
    fig.colorbar(im_acc, ax=ax_acc, fraction=0.05, pad=0.02, label="Accuracy (%)")

    # Diverging norm around 0 so penalties (negative) and bonuses (positive) read clearly.
    finite = mat[np.isfinite(mat)]
    if finite.size:
        vmax = float(np.max(np.abs(finite)))
        vmax = max(vmax, 1.0e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm = None
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r.testing_id for r in runs])
    ax.set_xlabel("Configuration")
    ax.set_title("Reward weights per ablation config (accuracy shown above)")
    for i in range(len(keys)):
        for j in range(len(runs)):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:g}", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="Reward weight")
    _save_fig(fig, out_dir / "02_reward_weights_heatmap.png")


def plot_accuracy_vs_weights(runs: list[AblationRun], out_dir: Path) -> None:
    """Scatter matrix: each reward weight vs achieved accuracy (all configs on one figure)."""
    keys = [k for k in _REWARD_KEYS if any(k in r.rewards for r in runs)]
    if not keys:
        return
    n = len(keys)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    accuracies = np.array([100.0 * r.accuracy for r in runs], dtype=np.float64)
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(runs)))

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        xs = np.array([r.rewards.get(key, np.nan) for r in runs], dtype=np.float64)
        ax.scatter(xs, accuracies, c=colors, s=90, edgecolors="#222222", zorder=3)
        for run, x, y in zip(runs, xs, accuracies):
            if np.isfinite(x):
                ax.annotate(run.testing_id, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel(key)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0.0, 105.0)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Accuracy vs {key}")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Ablation: accuracy vs each reward weight (all configs)", fontsize=12, y=1.01)
    fig.tight_layout()
    _save_fig(fig, out_dir / "03_accuracy_vs_reward_weights.png")


def plot_metrics_heatmap(runs: list[AblationRun], out_dir: Path) -> None:
    """Compact metrics heatmap (accuracy + times) with config column labels."""
    metric_names = ["accuracy (%)", "mean success time (s)", "mean episode time (s)"]
    mat = np.full((len(metric_names), len(runs)), np.nan, dtype=np.float64)
    for j, run in enumerate(runs):
        mat[0, j] = 100.0 * run.accuracy
        if run.mean_completion_time_s is not None:
            mat[1, j] = run.mean_completion_time_s
        if run.mean_episode_time_s is not None:
            mat[2, j] = run.mean_episode_time_s

    # Row-wise normalize for coloring only; annotate raw values.
    color_mat = np.full_like(mat, np.nan)
    for i in range(mat.shape[0]):
        row = mat[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1.0e-12:
            color_mat[i] = 0.5
        else:
            color_mat[i] = (row - lo) / (hi - lo)

    col_labels = []
    for run in runs:
        reward_bits = " | ".join(f"{k}={run.rewards[k]:g}" for k in _REWARD_KEYS if k in run.rewards)
        col_labels.append(f"{run.testing_id}\n{reward_bits}")

    fig, ax = plt.subplots(figsize=(max(10.0, 3.2 * len(runs)), 4.8))
    im = ax.imshow(color_mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_title("Ablation metrics (columns = reward configurations)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                text = "n/a"
            elif i == 0:
                text = f"{val:.1f}"
            else:
                text = f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Row-normalized (for color only)")
    _save_fig(fig, out_dir / "04_metrics_heatmap.png")


def plot_completion_time_bars(runs: list[AblationRun], out_dir: Path) -> None:
    """Secondary comparison: mean success / episode time with same config labels."""
    labels = [r.label for r in runs]
    success_t = np.array(
        [np.nan if r.mean_completion_time_s is None else r.mean_completion_time_s for r in runs],
        dtype=np.float64,
    )
    episode_t = np.array(
        [np.nan if r.mean_episode_time_s is None else r.mean_episode_time_s for r in runs],
        dtype=np.float64,
    )
    x = np.arange(len(runs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(9.0, 2.8 * len(runs)), 5.5))
    ax.bar(x - width / 2, success_t, width, label="mean success time", color="#1f4e79")
    ax.bar(x + width / 2, episode_t, width, label="mean episode time", color="#c0392b")
    for i, run in enumerate(runs):
        ax.text(i, max(success_t[i] if np.isfinite(success_t[i]) else 0.0, episode_t[i] if np.isfinite(episode_t[i]) else 0.0) + 0.05,
                f"acc={100.0 * run.accuracy:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time (s)")
    ax.set_title("Reward ablation — timing by configuration (accuracy annotated)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    _save_fig(fig, out_dir / "05_timing_by_config.png")


def write_tables(runs: list[AblationRun], out_dir: Path) -> None:
    """CSV + human-readable summary of accuracy and reward configs."""
    keys = [k for k in _REWARD_KEYS if any(k in r.rewards for r in runs)]
    extra = sorted({k for r in runs for k in r.rewards if k not in keys})
    keys = keys + extra

    csv_lines = [
        "testing_id,training_id,accuracy_pct,num_success,num_attempts,"
        "mean_completion_time_s,mean_episode_time_s," + ",".join(keys)
    ]
    for run in runs:
        row = [
            run.testing_id,
            run.training_id or "",
            f"{100.0 * run.accuracy:.4f}",
            str(run.num_success),
            str(run.num_attempts),
            "" if run.mean_completion_time_s is None else f"{run.mean_completion_time_s:.6f}",
            "" if run.mean_episode_time_s is None else f"{run.mean_episode_time_s:.6f}",
        ]
        for key in keys:
            row.append("" if key not in run.rewards else f"{run.rewards[key]:g}")
        csv_lines.append(",".join(row))

    csv_path = out_dir / "ablation_summary.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {csv_path}", flush=True)

    best = max(runs, key=lambda r: r.accuracy)
    lines = [
        "Reward ablation summary",
        "=======================",
        f"configs: {len(runs)}",
        f"best accuracy: {100.0 * best.accuracy:.1f}%  ({best.testing_id})",
        "",
    ]
    for run in runs:
        lines.append(f"[{run.testing_id}]  accuracy={100.0 * run.accuracy:.1f}%  ({run.num_success}/{run.num_attempts})")
        if run.training_id:
            lines.append(f"  training_id: {run.training_id}")
        if run.checkpoint:
            lines.append(f"  checkpoint: {run.checkpoint}")
        if run.mean_completion_time_s is not None:
            lines.append(f"  mean_completion_time_s: {run.mean_completion_time_s:.4f}")
        if run.mean_episode_time_s is not None:
            lines.append(f"  mean_episode_time_s: {run.mean_episode_time_s:.4f}")
        lines.append("  rewards:")
        for key in keys:
            if key in run.rewards:
                lines.append(f"    {key} = {run.rewards[key]:g}")
        lines.append("")

    txt_path = out_dir / "summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote {txt_path}", flush=True)

    # Machine-readable aggregate for later reuse.
    payload = {
        "task": out_dir.parent.name if out_dir.name == "ablation" else None,
        "runs": [
            {
                "testing_id": r.testing_id,
                "training_id": r.training_id,
                "checkpoint": r.checkpoint,
                "accuracy": r.accuracy,
                "num_success": r.num_success,
                "num_attempts": r.num_attempts,
                "mean_completion_time_s": r.mean_completion_time_s,
                "mean_episode_time_s": r.mean_episode_time_s,
                "rewards": r.rewards,
            }
            for r in runs
        ],
    }
    json_path = out_dir / "ablation_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {json_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward-ablation accuracy comparisons")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task family under results/ (default: move_cube)")
    parser.add_argument(
        "--testing-ids",
        nargs="+",
        default=["testing_1", "testing_2", "testing_3"],
        help="Testing folders to include (default: testing_1 testing_2 testing_3)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results root (default: <repo>/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/<task>/ablation)",
    )
    args = parser.parse_args()

    task_dir = task_results_dir(args.task, results_root=args.results_root)
    out_dir = args.output_dir if args.output_dir is not None else task_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    testing_dirs = discover_testing_dirs(task_dir, args.testing_ids)
    runs = load_ablation_runs(testing_dirs)
    print(f"[INFO] Loaded {len(runs)} ablation configs from {task_dir}", flush=True)

    plot_accuracy_bars(runs, out_dir)
    plot_reward_heatmap(runs, out_dir)
    plot_accuracy_vs_weights(runs, out_dir)
    plot_metrics_heatmap(runs, out_dir)
    plot_completion_time_bars(runs, out_dir)
    write_tables(runs, out_dir)
    print(f"[INFO] Ablation plots written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
