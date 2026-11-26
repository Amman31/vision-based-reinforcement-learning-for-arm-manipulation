#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the policy camera view (CameraLeft) as native RGB + inferno depth.

Launches the given Gymnasium task the same way training does, runs the env for
``--wait-s`` wall-clock seconds, then saves the raw 128x128 RGB frame and an
inferno colormap of ``distance_to_image_plane`` from the same camera. Images are
written at the sensor resolution with no resize.

Example::

    python scripts/capture_rgbd_view.py \\
        --task ThesisRl-Move-Cube-UR5E \\
        --wait-s 10 \\
        physics=isaacsim_physx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks  # noqa: F401
import thesis_rl.tasks  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "ThesisRl-Move-Cube-UR5E"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "images"

# Short task-family names → Gymnasium ids (same map as training_results_lib).
_TASK_ALIASES = {
    "move_cube": "ThesisRl-Move-Cube-UR5E",
    "move_cube_detection": "ThesisRl-Move-Cube-Detection-UR5E",
    "pick_cube": "ThesisRl-Pick-Cube-UR5E",
    "pick_cube_final_rb": "ThesisRl-Pick-Cube-Final-Rb-UR5E",
    "pick_cube_final": "ThesisRl-Pick-Cube-Final-UR5E",
    "pick_cube_final_dual": "ThesisRl-Pick-Cube-Final-Dual-UR5E",
    "pick_cube_final_eye": "ThesisRl-Pick-Cube-Final-Eye-UR5E",
    "pick_place_cube": "ThesisRl-Pick-Place-Cube-UR5E",
    "pick_and_place": "ThesisRl-Pick-And-Place-UR5E",
    "orange_pick_place": "ThesisRl-Orange-Pick-Place-UR5E",
}


def _resolve_task_id(task: str) -> str:
    key = task.strip()
    if key in _TASK_ALIASES:
        return _TASK_ALIASES[key]
    return key


def _to_numpy(x) -> np.ndarray:
    # Isaac Lab camera buffers are ProxyArray; prefer the explicit .torch view.
    if hasattr(x, "torch"):
        x = x.torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rgb_uint8(rgb) -> np.ndarray:
    """Return HxWx3 uint8 RGB with no spatial resize."""
    arr = _to_numpy(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        # Camera sensors usually emit 0–255 floats; obs code may also normalize to 0–1.
        max_v = float(np.nanmax(arr)) if arr.size else 0.0
        if max_v <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _depth_inferno(depth, max_depth: float) -> np.ndarray:
    """Map meters → inferno RGB at the same HxW (no resize)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    arr = _to_numpy(depth).astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    finite = np.isfinite(arr)
    clipped = np.full_like(arr, max_depth, dtype=np.float32)
    clipped[finite] = np.clip(arr[finite], 0.0, max_depth)
    # Near → bright (1), far → dark (0) so close geometry pops in inferno.
    norm = 1.0 - (clipped / max_depth)
    rgba = cm.inferno(norm)
    out = (rgba[..., :3] * 255.0).astype(np.uint8)
    # Invalid / missing depth → black
    out[~finite] = 0
    return np.ascontiguousarray(out)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Pillow is required to save PNGs (pip install pillow).") from exc
    Image.fromarray(image).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture native CameraLeft RGB + inferno depth after a short wait."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Gymnasium task id or short name (move_cube, move_cube_detection, ...).",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--wait-s",
        type=float,
        default=10.0,
        help="Wall-clock seconds to run the env before capturing (after reset).",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="camera",
        help="Scene sensor name (default: camera → ZED CameraLeft).",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=3.0,
        help="Far clip in meters for the inferno map (same default as training obs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rgb / depth PNGs.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Filename stem (default: <task>_cameraleft).",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    task_id = _resolve_task_id(args_cli.task)
    stem = args_cli.stem or f"{task_id.split(':')[-1].lower()}_cameraleft"
    out_dir = Path(args_cli.output_dir)

    env_cfg, _ = resolve_task_config(task_id, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    print(f"[INFO] Task        : {task_id}", flush=True)
    print(f"[INFO] Sensor      : {args_cli.sensor}", flush=True)
    print(f"[INFO] Wait        : {args_cli.wait_s:.1f}s after reset", flush=True)
    print(f"[INFO] Output dir  : {out_dir.resolve()}", flush=True)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(task_id, cfg=env_cfg)
        try:
            env.reset()
            unwrapped = env.unwrapped
            if args_cli.sensor not in unwrapped.scene.keys():
                raise SystemExit(
                    f"Sensor {args_cli.sensor!r} not in scene. "
                    f"Available: {list(unwrapped.scene.keys())}"
                )

            action = torch.zeros(env.action_space.shape, device=unwrapped.device)
            t0 = time.perf_counter()
            steps = 0
            while (time.perf_counter() - t0) < float(args_cli.wait_s):
                env.step(action)
                steps += 1

            camera = unwrapped.scene[args_cli.sensor]
            # Ensure latest render is available.
            if hasattr(camera, "update"):
                camera.update(0.0)

            output = camera.data.output
            if "rgb" not in output:
                raise SystemExit("Camera output missing 'rgb'. Check CameraCfg.data_types.")
            if "distance_to_image_plane" not in output:
                raise SystemExit(
                    "Camera output missing 'distance_to_image_plane'. "
                    "Check CameraCfg.data_types."
                )

            rgb = _rgb_uint8(output["rgb"])
            depth = _depth_inferno(output["distance_to_image_plane"], float(args_cli.max_depth))

            if rgb.shape[:2] != depth.shape[:2]:
                raise SystemExit(
                    f"RGB shape {rgb.shape} and depth shape {depth.shape} mismatch."
                )

            rgb_path = out_dir / f"{stem}_rgb.png"
            depth_path = out_dir / f"{stem}_depth_inferno.png"
            _save_png(rgb_path, rgb)
            _save_png(depth_path, depth)

            h, w = rgb.shape[:2]
            print("\n==========================================")
            print(" CAMERALEFT CAPTURE")
            print("==========================================")
            print(f"Steps run   : {steps}")
            print(f"Resolution  : {w} x {h} (native, no resize)")
            print(f"RGB         : {rgb_path.resolve()}")
            print(f"Depth       : {depth_path.resolve()}")
            print(f"Max depth   : {args_cli.max_depth:.2f} m → inferno")
            print("==========================================\n")
        finally:
            env.close()


if __name__ == "__main__":
    main()
