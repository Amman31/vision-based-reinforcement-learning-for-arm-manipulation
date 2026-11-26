#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize red-cube bounding-box detection on pick_cube_detection.

Opens ``ThesisRl-Pick-Cube-Detection-UR5E``, captures several CameraLeft RGB
frames, draws a yellow bbox around the detected cube, and overlays camera-
estimated XYZ vs simulator ground-truth XYZ. Images are saved under ``images/``.

Example::

    python scripts/detection.py
    python scripts/detection.py --num-images 8 physics=isaacsim_physx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks  # noqa: F401
import thesis_rl.tasks  # noqa: F401

from thesis_rl.tasks.pick_cube_detection.mdp.observations import (
    _axis_aligned_bbox,
    _clean_detection_mask,
    cube_position_from_camera,
)
from thesis_rl.tasks.pick_cube_detection.mdp.rewards import cube_position

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "ThesisRl-Pick-Cube-Detection-UR5E"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "images"
YELLOW = (255, 220, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "torch"):
        x = x.torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rgb_uint8(rgb) -> np.ndarray:
    """HxWx3 uint8 RGB from camera buffer (env 0)."""
    arr = _to_numpy(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        max_v = float(np.nanmax(arr)) if arr.size else 0.0
        if max_v <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _load_font(size: int):
    from PIL import ImageFont

    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text_outlined(draw, xy, text, font, fill=YELLOW, outline=BLACK, width: int = 3) -> None:
    """Yellow text with a dark outline so it stays readable on any background."""
    x, y = xy
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def _draw_bbox_and_labels(
    rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int] | None,
    cam_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    err_m: float,
    scale: int,
) -> np.ndarray:
    """Upscale RGB, draw yellow bbox, put large XYZ labels in a bottom caption bar."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Pillow is required (pip install pillow).") from exc

    h, w = rgb.shape[:2]
    view = Image.fromarray(rgb).resize((w * scale, h * scale), Image.NEAREST)

    # Same caption proportion as the old 128×10 frames (~48 px on a 1280 px view).
    font_size = max(12, int(0.038 * view.height))
    font = _load_font(font_size)
    probe = ImageDraw.Draw(view)
    lines = [
        f"camera:  [{cam_xyz[0]:+.3f}, {cam_xyz[1]:+.3f}, {cam_xyz[2]:+.3f}]",
        f"ground_truth:  [{gt_xyz[0]:+.3f}, {gt_xyz[1]:+.3f}, {gt_xyz[2]:+.3f}]",
        f"error:  {err_m:.3f}",
    ]
    line_gap = max(2, font_size // 4)
    heights = []
    for line in lines:
        box = probe.textbbox((0, 0), line, font=font)
        heights.append(box[3] - box[1])
    pad_y = max(4, font_size // 2)
    pad_x = max(6, font_size // 2)
    bar_h = pad_y * 2 + sum(heights) + line_gap * (len(lines) - 1)

    # Canvas = camera view + dark bottom caption bar.
    canvas = Image.new("RGB", (view.width, view.height + bar_h), BLACK)
    canvas.paste(view, (0, 0))
    draw = ImageDraw.Draw(canvas)

    if bbox_xyxy is not None:
        u0, v0, u1, v1 = bbox_xyxy
        pad = 1
        x0 = max(0, (u0 - pad) * scale)
        y0 = max(0, (v0 - pad) * scale)
        x1 = min(view.width - 1, (u1 + pad) * scale)
        y1 = min(view.height - 1, (v1 + pad) * scale)
        for t in range(max(1, int(round(view.height * 5 / 1280)))):
            draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=YELLOW)
    else:
        _draw_text_outlined(draw, (pad_x, pad_y), "NO DETECTION", font, fill=(255, 80, 80))

    # Thin yellow separator above the caption.
    draw.line([(0, view.height), (canvas.width, view.height)], fill=YELLOW, width=max(2, scale // 4))

    y = view.height + pad_y
    for line, lh in zip(lines, heights):
        _draw_text_outlined(draw, (pad_x, y), line, font, width=max(1, font_size // 24))
        y += lh + line_gap

    return np.asarray(canvas)


def _detect_bbox_env0(env) -> tuple[tuple[int, int, int, int] | None, np.ndarray, np.ndarray, float]:
    """Run training detection on env 0; return bbox, cam XYZ, GT XYZ, L2 error."""
    camera = env.scene["depth_camera"]
    rgb_t = camera.data.output["rgb"].torch[..., :3].float()
    if rgb_t.max() > 1.5:
        rgb_t = rgb_t / 255.0
    depth = camera.data.output["distance_to_image_plane"].torch.float()
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # Same cleaned mask as training (rejects red-lag noise).
    mask = _clean_detection_mask(rgb_t, depth)
    u_min, u_max, v_min, v_max, has_det = _axis_aligned_bbox(mask)

    cam_xyz = cube_position_from_camera(env)[0].detach().cpu().numpy()
    gt_xyz = cube_position(env)[0].detach().cpu().numpy()
    err = float(np.linalg.norm(cam_xyz - gt_xyz))

    if bool(has_det[0].item()):
        bbox = (
            int(u_min[0].item()),
            int(v_min[0].item()),
            int(u_max[0].item()),
            int(v_max[0].item()),
        )
    else:
        bbox = None
    return bbox, cam_xyz, gt_xyz, err


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save CameraLeft RGB frames with yellow cube bbox + cam/GT XYZ."
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--num-images",
        type=int,
        default=6,
        help="Number of annotated frames to save (reset between captures).",
    )
    parser.add_argument(
        "--steps-after-reset",
        type=int,
        default=4,
        help="Sim steps after each reset before capturing (lets render settle).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Nearest-neighbor upscale applied after the 400x400 capture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for annotated PNGs (default: images/).",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    task_id = args_cli.task.strip()
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg, _ = resolve_task_config(task_id, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    # Visualization capture. Training still uses the 128x128 camera in the task cfg.
    env_cfg.scene.depth_camera.width = 400
    env_cfg.scene.depth_camera.height = 400
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    print(f"[INFO] Task        : {task_id}", flush=True)
    print(f"[INFO] Num images  : {args_cli.num_images}", flush=True)
    print(f"[INFO] Output dir  : {out_dir.resolve()}", flush=True)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(task_id, cfg=env_cfg)
        try:
            unwrapped = env.unwrapped
            action = torch.zeros(env.action_space.shape, device=unwrapped.device)
            saved: list[Path] = []

            for i in range(int(args_cli.num_images)):
                env.reset()
                for _ in range(int(args_cli.steps_after_reset)):
                    env.step(action)

                camera = unwrapped.scene["depth_camera"]
                if hasattr(camera, "update"):
                    camera.update(0.0)

                rgb = _rgb_uint8(camera.data.output["rgb"])
                bbox, cam_xyz, gt_xyz, err = _detect_bbox_env0(unwrapped)
                annotated = _draw_bbox_and_labels(
                    rgb, bbox, cam_xyz, gt_xyz, err, scale=max(1, int(args_cli.scale))
                )

                path = out_dir / f"pick_cube_detection_bbox_{i:02d}.png"
                try:
                    from PIL import Image
                except ImportError as exc:  # pragma: no cover
                    raise SystemExit("Pillow is required (pip install pillow).") from exc
                Image.fromarray(annotated).save(path)
                saved.append(path)

                bbox_s = "none" if bbox is None else f"[{bbox[0]},{bbox[1]}]–[{bbox[2]},{bbox[3]}]"
                print(
                    f"[INFO] Saved {path.name}  bbox={bbox_s}  "
                    f"err={err:.3f} m  "
                    f"cam=[{cam_xyz[0]:+.3f},{cam_xyz[1]:+.3f},{cam_xyz[2]:+.3f}]  "
                    f"gt=[{gt_xyz[0]:+.3f},{gt_xyz[1]:+.3f},{gt_xyz[2]:+.3f}]",
                    flush=True,
                )

            print("\n==========================================")
            print(" DETECTION VISUALIZATION")
            print("==========================================")
            print(f"Frames saved : {len(saved)}")
            print(f"Directory    : {out_dir.resolve()}")
            for p in saved:
                print(f"  - {p.name}")
            print("==========================================\n")
        finally:
            env.close()


if __name__ == "__main__":
    main()
