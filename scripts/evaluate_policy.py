#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy for a fixed number of attempts and plot accuracy / time.

Example (same flags as ``isaaclab play``)::

    python scripts/evaluate_policy.py \\
        --task ThesisRl-Move-Cube-UR5E \\
        --checkpoint model/move_cube/training_1.pt \\
        --num_envs 1 --viz kit \\
        physics=isaacsim_physx

Writes ``results/<task>/testing_N/`` next to the matching ``training_N`` export.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env, request_determinism
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import thesis_rl.tasks  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_results_lib import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    gym_id_for_task,
    task_from_gym_id,
    testing_dir_for_training_id,
)

_DEFAULT_GYM = gym_id_for_task("move_cube_detection")

parser = argparse.ArgumentParser(description="Evaluate a trained RSL-RL policy for N attempts.")
parser.add_argument("--task", type=str, default=_DEFAULT_GYM, help="Gymnasium task id.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num-attempts", type=int, default=100, dest="num_attempts")
parser.add_argument("--training-id", type=str, default=None, help="e.g. training_1 → results/<task>/testing_1")
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument(
    "--train_env_cfg",
    action="store_true",
    default=False,
    help="Use the training env config as-is (skip play-mode overrides).",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(attempts: list[dict], out_dir: Path, accuracy: float) -> None:
    xs = [row["attempt"] for row in attempts]
    ys = [1.0 if row["success"] else 0.0 for row in attempts]
    running = []
    ok = 0
    for i, y in enumerate(ys, start=1):
        ok += int(y)
        running.append(ok / i)
    n_ok = int(sum(ys))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(xs, ys, c="#1f4e79", s=18, zorder=3, label="attempt (1 = success)")
    ax.plot(xs, running, color="#c0392b", linewidth=2.0, label="running accuracy")
    ax.axhline(accuracy, color="#2e7d32", linestyle="--", linewidth=1.4, label="mean accuracy")
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Success")
    ax.set_title(f"Average accuracy: {100.0 * accuracy:.1f}%  ({n_ok}/{len(attempts)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    _save_fig(fig, out_dir / "01_accuracy.png")


def plot_completion_time(attempts: list[dict], out_dir: Path, mean_success_s: float | None) -> None:
    xs = [row["attempt"] for row in attempts]
    success_x = [row["attempt"] for row in attempts if row["success"]]
    success_y = [row["time_s"] for row in attempts if row["success"]]
    fail_x = [row["attempt"] for row in attempts if not row["success"]]
    fail_y = [row["time_s"] for row in attempts if not row["success"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if success_x:
        ax.scatter(success_x, success_y, c="#1f4e79", s=22, zorder=3, label="success")
    if fail_x:
        ax.scatter(fail_x, fail_y, c="#c0392b", s=22, zorder=3, marker="x", label="timeout / fail")
    if mean_success_s is not None:
        ax.axhline(mean_success_s, color="#2e7d32", linestyle="--", linewidth=1.4, label="mean success time")
        title = f"Average completion time: {mean_success_s:.2f} s  ({len(success_x)} successful)"
    else:
        title = "Average completion time: n/a  (no successful attempts)"
    if xs:
        ax.plot(xs, [row["time_s"] for row in attempts], color="#8aa0b4", alpha=0.35, linewidth=1.0)
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Episode time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    _save_fig(fig, out_dir / "02_completion_time.png")


def write_eval_outputs(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    plots_dir = out_dir / "plots"
    summary = payload["summary"]
    plot_accuracy(payload["attempts"], plots_dir, float(summary["accuracy"]))
    mean_s = summary.get("mean_completion_time_s")
    plot_completion_time(payload["attempts"], plots_dir, None if mean_s is None else float(mean_s))
    lines = [
        f"task: {payload['meta']['task']}",
        f"testing_id: {payload['meta']['testing_id']}",
        f"checkpoint: {payload['meta']['checkpoint']}",
        f"attempts: {summary['num_attempts']}",
        f"accuracy: {100.0 * summary['accuracy']:.1f}% ({summary['num_success']}/{summary['num_attempts']})",
        (
            f"mean_completion_time_s: {summary['mean_completion_time_s']:.4f}"
            if summary.get("mean_completion_time_s") is not None
            else "mean_completion_time_s: n/a"
        ),
        f"mean_episode_time_s: {summary['mean_episode_time_s']:.4f}",
    ]
    (plots_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote eval results: {result_path}", flush=True)
    print(f"[INFO] Wrote eval plots  : {plots_dir}", flush=True)


def _resolve_output_dir(gym_id: str, checkpoint: str, training_id: str | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    task = task_from_gym_id(gym_id)
    if training_id is None:
        stem = Path(checkpoint).stem
        training_id = stem if stem.startswith("training_") else None
    if training_id is None:
        raise SystemExit("Pass --training-id training_N or a checkpoint named training_N.pt")
    return testing_dir_for_training_id(training_id, task=task, results_root=DEFAULT_RESULTS_ROOT)


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run N sequential attempts with a loaded policy and write accuracy / time plots."""
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
        request_determinism(args_cli, env_cfg)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if not args_cli.checkpoint:
            raise SystemExit("Pass --checkpoint /path/to/training_N.pt")
        resume_path = retrieve_file_path(args_cli.checkpoint)
        out_dir = _resolve_output_dir(args_cli.task, resume_path, args_cli.training_id, args_cli.output_dir)

        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
        )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading checkpoint: {resume_path}", flush=True)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, torch_deterministic=True)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        raw = env.unwrapped
        dt = float(raw.step_dt)
        max_steps = int(raw.max_episode_length) + 8
        num_envs = int(raw.num_envs)
        obs = env.get_observations()
        attempts: list[dict] = []
        steps = 0
        print(f"[INFO] Evaluating {args_cli.num_attempts} attempts (num_envs={num_envs})...", flush=True)

        while len(attempts) < args_cli.num_attempts:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
            steps += 1
            done = bool(dones[0].item()) if torch.is_tensor(dones) else bool(dones[0])
            if not done and steps < max_steps:
                continue
            # Prefer the named success term. reset_terminated is True for any
            # failure termination too (e.g. cube_dropped), so it cannot be used
            # as the accuracy signal when those are present.
            term_mgr = getattr(raw, "termination_manager", None)
            if done and term_mgr is not None and "success" in term_mgr.active_terms:
                success = bool(term_mgr.get_term("success")[0].item())
                timed_out = bool(raw.reset_time_outs[0].item())
            else:
                success = bool(raw.reset_terminated[0].item()) if done else False
                timed_out = not success
            time_s = steps * dt
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "success": success,
                    "timed_out": timed_out,
                    "time_s": float(time_s),
                    "steps": int(steps),
                }
            )
            print(
                f"[INFO] attempt {attempts[-1]['attempt']:3d}/{args_cli.num_attempts}  "
                f"{'SUCCESS' if success else 'FAIL   '}  {time_s:.3f}s",
                flush=True,
            )
            if not done:
                raw.reset()
                obs = env.get_observations()
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(torch.ones(num_envs, dtype=torch.long, device=raw.device))
            steps = 0

        n = len(attempts)
        n_ok = sum(1 for row in attempts if row["success"])
        success_times = [row["time_s"] for row in attempts if row["success"]]
        payload = {
            "meta": {
                "task": task_from_gym_id(args_cli.task),
                "gym_id": args_cli.task,
                "testing_id": out_dir.name,
                "training_id": args_cli.training_id or Path(resume_path).stem,
                "checkpoint": str(Path(resume_path).resolve()),
                "evaluated_at": datetime.now(UTC).isoformat(),
                "num_envs": num_envs,
            },
            "summary": {
                "num_attempts": n,
                "num_success": n_ok,
                "num_fail": n - n_ok,
                "accuracy": (n_ok / n) if n else 0.0,
                "mean_completion_time_s": (sum(success_times) / len(success_times)) if success_times else None,
                "mean_episode_time_s": (sum(row["time_s"] for row in attempts) / n) if n else 0.0,
            },
            "attempts": attempts,
        }
        env.close()
        write_eval_outputs(out_dir, payload)


if __name__ == "__main__":
    main()
