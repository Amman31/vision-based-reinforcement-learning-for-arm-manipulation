#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate matplotlib figures from an exported ``result.json``.

Creates line charts, heatmaps, and a correlation matrix under
``results/<task>/training_N/plots/``.

Success rate always comes from ``Episode_Termination/success`` (fraction of
episodes that ended successfully), never from ``Episode_Reward/success``
(sparse one-shot bonus, typically near zero when averaged).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training_results_lib import (
    DEFAULT_TASK,
    compute_summary,
    find_latest_result_json,
    load_result_json,
)

# Reward terms that are regularization / failure penalties (not stage shaping).
_PENALTY_KEYS = (
    "penalty",
    "action_rate",
    "joint_velocity",
    "joint_vel",
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


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(values, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _rolling_mean_nan(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean that ignores NaNs (used after outlier masking)."""
    y = np.asarray(values, dtype=np.float64)
    if window <= 1 or y.size == 0:
        return y.copy()
    window = min(window, len(y))
    out = np.full_like(y, np.nan, dtype=np.float64)
    # Cumulative sums for O(n) nan-aware window means.
    valid = np.isfinite(y).astype(np.float64)
    filled = np.where(np.isfinite(y), y, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    ccnt = np.concatenate([[0.0], np.cumsum(valid)])
    half = window // 2
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), lo + window)
        lo = max(0, hi - window)
        cnt = ccnt[hi] - ccnt[lo]
        if cnt > 0:
            out[i] = (csum[hi] - csum[lo]) / cnt
    return out


def _mask_reward_outliers(y: np.ndarray) -> np.ndarray:
    """Mask catastrophic logging spikes so the learning curve is visible.

    TensorBoard / RSL-RL occasionally emits absurd ``Train/mean_reward`` values
    (e.g. -1e15). Those blow up the y-axis and make a normal 0→50 reward curve
    look flat. Returns a boolean mask (True = keep).
    """
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    if finite.sum() < 5:
        return finite

    # Absolute garbage (NaNs already excluded): physics / log explosions.
    keep = finite & (np.abs(y) < 1.0e4)

    if keep.sum() < 5:
        return finite

    med = float(np.median(y[keep]))
    mad = float(np.median(np.abs(y[keep] - med)))
    scale = max(1.4826 * mad, 1.0)
    # ~25 robust-sigma, but never tighter than ±100 around the median band floor.
    band = max(25.0 * scale, 100.0)
    keep = keep & (np.abs(y - med) <= band)

    # If almost everything was discarded, fall back to central percentiles.
    if keep.sum() < 0.5 * finite.sum():
        lo, hi = np.percentile(y[finite], [2, 98])
        keep = finite & (y >= lo) & (y <= hi)
    return keep


def _series(iterations: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in iterations:
        if key in row:
            xs.append(row["iteration"])
            ys.append(float(row[key]))
    return np.asarray(xs), np.asarray(ys, dtype=np.float64)


def _metric_matrix(
    iterations: list[dict],
    tags: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (iterations, values[num_tags, num_iters]) with NaN for missing."""
    its = np.array([row["iteration"] for row in iterations], dtype=np.int64)
    mat = np.full((len(tags), len(iterations)), np.nan, dtype=np.float64)
    for j, row in enumerate(iterations):
        for i, tag in enumerate(tags):
            if tag in row:
                mat[i, j] = float(row[tag])
    return its, mat


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _all_reward_tags(iterations: list[dict]) -> list[str]:
    return sorted({k for row in iterations for k in row if k.startswith("Episode_Reward/")})


def _all_termination_tags(iterations: list[dict]) -> list[str]:
    return sorted({k for row in iterations for k in row if k.startswith("Episode_Termination/")})


def _is_penalty_reward(tag: str) -> bool:
    name = tag.rsplit("/", 1)[-1].lower()
    return any(p in name for p in _PENALTY_KEYS)


def _is_success_reward(tag: str) -> bool:
    """Sparse success *bonus* — not the episode success rate."""
    return tag.endswith("/success")


def _split_reward_tags(reward_tags: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (stage_shaping_tags, success_bonus_tags, penalty_tags)."""
    success = [t for t in reward_tags if _is_success_reward(t)]
    rest = [t for t in reward_tags if t not in success]
    penalties = [t for t in rest if _is_penalty_reward(t)]
    stages = [t for t in rest if t not in penalties]
    return stages, success, penalties


def plot_mean_reward(iterations: list[dict], out_dir: Path, smooth: int = 25) -> None:
    x, y = _series(iterations, "Train/mean_reward")
    if x.size == 0:
        return

    keep = _mask_reward_outliers(y)
    y_plot = y.copy()
    y_plot[~keep] = np.nan
    n_masked = int((~keep).sum())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, y_plot, color="#8aa0b4", alpha=0.45, linewidth=1.0, label="raw")
    ax.plot(
        x,
        _rolling_mean_nan(y_plot, smooth),
        color="#1f4e79",
        linewidth=2.0,
        label=f"rolling mean ({smooth})",
    )
    ax.set_xlabel("Learning iteration")
    ax.set_ylabel("Mean episode reward")
    title = "Training reward"
    if n_masked:
        title += f"  ({n_masked} outlier point(s) hidden)"
    ax.set_title(title)
    # Robust y-limits from the cleaned series so early learning is visible.
    valid = y_plot[np.isfinite(y_plot)]
    if valid.size:
        lo, hi = np.percentile(valid, [1, 99])
        pad = max(0.05 * (hi - lo), 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "01_mean_reward.png")


def plot_losses(iterations: list[dict], out_dir: Path) -> None:
    tags = ["Loss/value", "Loss/surrogate", "Loss/entropy"]
    present = [t for t in tags if any(t in row for row in iterations)]
    if not present:
        return

    fig, axes = plt.subplots(len(present), 1, figsize=(10, 3.2 * len(present)), sharex=True)
    if len(present) == 1:
        axes = [axes]
    for ax, tag in zip(axes, present):
        x, y = _series(iterations, tag)
        ax.plot(x, y, linewidth=1.5)
        ax.set_ylabel(tag.split("/")[-1])
        ax.set_title(tag)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Learning iteration")
    _save(fig, out_dir / "02_losses.png")


def plot_policy_and_lr(iterations: list[dict], out_dir: Path) -> None:
    pairs = [
        ("Policy/mean_std", "Mean action std"),
        ("Loss/learning_rate", "Learning rate"),
    ]
    present = [(k, lab) for k, lab in pairs if any(k in row for row in iterations)]
    if not present:
        return

    fig, axes = plt.subplots(len(present), 1, figsize=(10, 3.0 * len(present)), sharex=True)
    if len(present) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, present):
        x, y = _series(iterations, key)
        ax.plot(x, y, color="#b35c1e", linewidth=1.5)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Learning iteration")
    _save(fig, out_dir / "03_policy_std_learning_rate.png")


def plot_success_rate(iterations: list[dict], out_dir: Path, smooth: int = 25) -> None:
    """Episode success rate from terminations (not the sparse success reward)."""
    key = "Episode_Termination/success"
    x, y = _series(iterations, key)
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, y, color="#8aa0b4", alpha=0.4, linewidth=1.0, label="raw")
    ax.plot(
        x,
        _rolling_mean(y, smooth),
        color="#1f7a4c",
        linewidth=2.2,
        label=f"rolling mean ({smooth})",
    )
    ax.set_xlabel("Learning iteration")
    ax.set_ylabel("Success rate")
    ax.set_title("Episode success rate (termination)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "04_success_rate.png")


def _series_for_plot(iterations: list[dict], key: str, smooth: int = 25) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (x, smoothed_y_with_outliers_as_nan, n_masked)."""
    x, y = _series(iterations, key)
    if x.size == 0:
        return x, y, 0
    keep = _mask_reward_outliers(y)
    y_plot = y.copy()
    y_plot[~keep] = np.nan
    return x, _rolling_mean_nan(y_plot, smooth), int((~keep).sum())


def plot_episode_rewards(iterations: list[dict], out_dir: Path, smooth: int = 25) -> None:
    """Plot reward terms only (no Episode_Termination/*).

    ``05_stage_rewards.png``: stage shaping + success *bonus* + penalties.
    Termination success rate stays on ``04_success_rate.png`` / ``07_terminations.png``.
    """
    reward_tags = _all_reward_tags(iterations)
    stages, success_bonuses, penalties = _split_reward_tags(reward_tags)
    positive_tags = stages + success_bonuses

    if positive_tags or penalties:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        left_vals: list[np.ndarray] = []
        n_masked_total = 0
        for tag in positive_tags:
            x, y_s, n_masked = _series_for_plot(iterations, tag, smooth=smooth)
            if x.size == 0:
                continue
            n_masked_total += n_masked
            label = "success (reward)" if _is_success_reward(tag) else tag.replace("Episode_Reward/", "")
            ax.plot(x, y_s, linewidth=2.0, label=label)
            left_vals.append(y_s[np.isfinite(y_s)])

        ax.set_xlabel("Learning iteration")
        ax.set_ylabel("Episode reward term")
        title = "Stage rewards, success bonus, and penalties"
        if n_masked_total:
            title += f"  ({n_masked_total} outlier point(s) hidden)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if left_vals:
            cat = np.concatenate(left_vals) if left_vals else np.array([])
            if cat.size:
                lo, hi = np.percentile(cat, [1, 99])
                pad = max(0.05 * (hi - lo), 0.02)
                ax.set_ylim(lo - pad, hi + pad)

        if penalties:
            ax2 = ax.twinx()
            right_vals: list[np.ndarray] = []
            for tag in penalties:
                x, y_s, n_masked = _series_for_plot(iterations, tag, smooth=smooth)
                if x.size == 0:
                    continue
                n_masked_total += n_masked
                label = tag.replace("Episode_Reward/", "")
                ax2.plot(x, y_s, linewidth=1.8, linestyle="--", label=label)
                right_vals.append(y_s[np.isfinite(y_s)])
            ax2.set_ylabel("Penalty reward term")
            if right_vals:
                cat = np.concatenate(right_vals)
                if cat.size:
                    lo, hi = np.percentile(cat, [1, 99])
                    pad = max(0.05 * (hi - lo), 1e-5)
                    ax2.set_ylim(lo - pad, hi + pad)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc="best")
        else:
            ax.legend(fontsize=7, ncol=2, loc="best")

        _save(fig, out_dir / "05_stage_rewards.png")

    if penalties:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        vals: list[np.ndarray] = []
        n_masked_total = 0
        for tag in penalties:
            x, y = _series(iterations, tag)
            if x.size == 0:
                continue
            keep = _mask_reward_outliers(y)
            y_plot = y.copy()
            y_plot[~keep] = np.nan
            n_masked_total += int((~keep).sum())
            ax.plot(x, y_plot, alpha=0.35, linewidth=1.0)
            y_s = _rolling_mean_nan(y_plot, smooth)
            ax.plot(x, y_s, linewidth=2.0, label=tag.replace("Episode_Reward/", ""))
            vals.append(y_s[np.isfinite(y_s)])
        ax.set_xlabel("Learning iteration")
        ax.set_ylabel("Episode reward term")
        title = "Penalty / regularization terms"
        if n_masked_total:
            title += f"  ({n_masked_total} outlier point(s) hidden)"
        ax.set_title(title)
        if vals:
            cat = np.concatenate(vals)
            if cat.size:
                lo, hi = np.percentile(cat, [1, 99])
                pad = max(0.05 * (hi - lo), 1e-5)
                ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
        _save(fig, out_dir / "06_penalty_terms.png")


def plot_terminations(iterations: list[dict], out_dir: Path, smooth: int = 25) -> None:
    tags = _all_termination_tags(iterations)
    if not tags:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for tag in tags:
        x, y = _series(iterations, tag)
        label = tag.replace("Episode_Termination/", "")
        ax.plot(x, y, alpha=0.35, linewidth=1.0)
        ax.plot(x, _rolling_mean(y, smooth), linewidth=2.0, label=label)
    ax.set_xlabel("Learning iteration")
    ax.set_ylabel("Fraction of episode ends")
    ax.set_title("Episode terminations")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "07_terminations.png")


def plot_reward_heatmap(iterations: list[dict], out_dir: Path, max_cols: int = 200) -> None:
    # Exclude sparse success bonus — it dwarfs nothing but confuses the scale story.
    tags = [t for t in _all_reward_tags(iterations) if not _is_success_reward(t)]
    if not tags or not iterations:
        return

    its, mat = _metric_matrix(iterations, tags)

    if mat.shape[1] > max_cols:
        idx = np.linspace(0, mat.shape[1] - 1, max_cols).astype(int)
        mat = mat[:, idx]
        its = its[idx]

    row_mean = np.nanmean(mat, axis=1, keepdims=True)
    row_std = np.nanstd(mat, axis=1, keepdims=True)
    row_std = np.where(row_std < 1e-8, 1.0, row_std)
    z = (mat - row_mean) / row_std

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.35 * len(tags))))
    im = ax.imshow(z, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_yticks(np.arange(len(tags)))
    ax.set_yticklabels([t.replace("Episode_Reward/", "") for t in tags], fontsize=8)
    tick_count = min(8, len(its))
    tick_pos = np.linspace(0, len(its) - 1, tick_count).astype(int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(its[i]) for i in tick_pos])
    ax.set_xlabel("Learning iteration")
    ax.set_title("Episode reward terms (row-wise z-score; success bonus excluded)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="z-score")
    _save(fig, out_dir / "08_reward_terms_heatmap.png")


def plot_correlation_heatmap(iterations: list[dict], out_dir: Path) -> None:
    """Pearson correlation across selected training metrics."""
    candidates = []
    for key in (
        "Train/mean_reward",
        "Loss/value",
        "Loss/surrogate",
        "Loss/entropy",
        "Policy/mean_std",
        "Episode_Termination/success",
        "Episode_Termination/time_out",
    ):
        if any(key in row for row in iterations):
            candidates.append(key)

    # Prefer shaping/penalty rewards; never include Episode_Reward/success.
    reward_tags = [t for t in _all_reward_tags(iterations) if not _is_success_reward(t)]
    candidates.extend(reward_tags[:12])
    if len(candidates) < 2:
        return

    _, mat = _metric_matrix(iterations, candidates)
    valid_rows = ~np.all(np.isnan(mat), axis=1)
    mat = mat[valid_rows]
    labels = [candidates[i] for i, ok in enumerate(valid_rows) if ok]
    if mat.shape[0] < 2 or mat.shape[1] == 0:
        return

    valid_cols = ~np.all(np.isnan(mat), axis=0)
    mat = mat[:, valid_cols]
    if mat.shape[1] < 2:
        return

    filled = mat.copy()
    for i in range(filled.shape[0]):
        row = filled[i]
        if np.all(np.isnan(row)):
            continue
        mean = np.nanmean(row)
        row[np.isnan(row)] = mean
        filled[i] = row

    if np.any(np.isnan(filled)):
        return

    std = np.std(filled, axis=1)
    keep = std > 1e-12
    filled = filled[keep]
    labels = [lab for lab, ok in zip(labels, keep) if ok]
    if filled.shape[0] < 2:
        return

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(filled)
    if np.any(np.isnan(corr)):
        corr = np.nan_to_num(corr, nan=0.0)
    short = []
    for lab in labels:
        short.append(
            lab.replace("Episode_Reward/", "R/")
            .replace("Episode_Termination/", "T/")
            .replace("Loss/", "L/")
            .replace("Train/", "")
            .replace("Policy/", "P/")
        )

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(short)), max(6, 0.55 * len(short))))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(short)))
    ax.set_yticks(np.arange(len(short)))
    ax.set_xticklabels(short, rotation=75, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title("Metric correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir / "09_metric_correlation.png")


def plot_stage_progress_bars(summary: dict, out_dir: Path) -> None:
    """Bar chart: late−early for reward stages + termination success rate."""
    progress = dict(summary.get("stage_reward_progress") or {})
    # Inject termination success so "success" means episode success rate.
    succ = summary.get("success_rate_progress")
    if succ:
        progress = {
            "Episode_Termination/success": succ,
            **{k: v for k, v in progress.items() if not str(k).endswith("/success")},
        }
    if not progress:
        return

    tags = sorted(progress.keys(), key=lambda t: progress[t]["delta"])
    deltas = [progress[t]["delta"] for t in tags]
    labels = [
        t.replace("Episode_Reward/", "R/")
        .replace("Episode_Termination/", "T/")
        for t in tags
    ]

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(labels))))
    colors = ["#2a9d8f" if d >= 0 else "#e76f51" for d in deltas]
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Late mean − early mean (first/last 10% of iterations)")
    ax.set_title("Training progress (T/success = termination success rate)")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, out_dir / "10_stage_progress.png")


def plot_performance(iterations: list[dict], out_dir: Path) -> None:
    tags = [
        ("Perf/total_fps", "Steps / second"),
        ("Perf/collection_time", "Collection time (s)"),
        ("Perf/learning_time", "Learning time (s)"),
    ]
    present = [(k, lab) for k, lab in tags if any(k in row for row in iterations)]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(10, 2.8 * len(present)), sharex=True)
    if len(present) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, present):
        x, y = _series(iterations, key)
        ax.plot(x, y, linewidth=1.2)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Learning iteration")
    _save(fig, out_dir / "11_performance.png")


def generate_all_plots(result_json: Path, plots_dir: Path | None = None) -> Path:
    """Generate all analysis figures for one exported run."""
    payload = load_result_json(result_json)
    iterations = payload.get("iterations", [])
    # Always recompute summary so older exports pick up termination-success metrics.
    summary = compute_summary(iterations)
    payload["summary"] = summary
    if plots_dir is None:
        plots_dir = result_json.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not iterations:
        raise ValueError(f"No iteration data in {result_json}")

    # Drop stale filenames from older plot layouts.
    for stale in (
        "04_stage_rewards.png",
        "05_penalty_terms.png",
        "06_terminations.png",
        "07_reward_terms_heatmap.png",
        "08_metric_correlation.png",
        "09_stage_progress.png",
        "10_performance.png",
    ):
        old = plots_dir / stale
        if old.is_file():
            old.unlink()

    plot_mean_reward(iterations, plots_dir)
    plot_losses(iterations, plots_dir)
    plot_policy_and_lr(iterations, plots_dir)
    plot_success_rate(iterations, plots_dir)
    plot_episode_rewards(iterations, plots_dir)
    plot_terminations(iterations, plots_dir)
    plot_reward_heatmap(iterations, plots_dir)
    plot_correlation_heatmap(iterations, plots_dir)
    plot_stage_progress_bars(summary, plots_dir)
    plot_performance(iterations, plots_dir)

    summary_path = plots_dir / "summary.txt"
    lines = [
        f"task: {payload.get('meta', {}).get('task')}",
        f"training_id: {payload.get('meta', {}).get('training_id')}",
        f"source: {payload.get('meta', {}).get('source_log_dir')}",
        f"iterations: {summary.get('num_iterations')}",
        f"learning_iterations: {summary.get('num_learning_iterations', summary.get('num_iterations'))}",
        f"final_mean_reward: {summary.get('final_mean_reward')}",
        f"best_mean_reward: {summary.get('best_mean_reward')} @ iter {summary.get('best_mean_reward_iteration')}",
        f"final_success_rate: {summary.get('final_success_rate')}  (Episode_Termination/success)",
        f"max_success_rate: {summary.get('max_success_rate')}  (Episode_Termination/success)",
    ]
    succ_prog = summary.get("success_rate_progress")
    if succ_prog:
        lines.append(
            f"success_rate_progress: early={succ_prog['early_mean']:.4f} "
            f"late={succ_prog['late_mean']:.4f} delta={succ_prog['delta']:.4f}"
        )
    summary_path.write_text("\n".join(str(x) for x in lines) + "\n", encoding="utf-8")
    return plots_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an exported training result.json")
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help=f"Task family under results/ when --result-json is omitted (default: {DEFAULT_TASK})",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Path to result.json (default: latest results/<task>/training_*/result.json)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: <run>/plots)",
    )
    args = parser.parse_args()

    result_json = args.result_json
    if result_json is None:
        try:
            result_json = find_latest_result_json(task=args.task)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    plots_dir = generate_all_plots(result_json, args.plots_dir)
    print(f"[INFO] Wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()
