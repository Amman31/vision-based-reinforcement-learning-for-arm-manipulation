#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture CameraLeft frames from move-cube detection with the policy vision labels.

Opens ``ThesisRl-Move-Cube-Detection-UR5E``, runs the same OpenCV red-cube
detector as training, and saves the RGB frame with a yellow bounding box.
Beside the box: pixel center ``(x, y)``, median depth in meters, and the
normalized ``[xn, yn, depth]`` triple that is fed to the policy.

Example::

    python scripts/detection_move_cube.py physics=isaacsim_physx
    python scripts/detection_move_cube.py --num-images 4 physics=isaacsim_physx
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

from thesis_rl.tasks.move_cube_detection.mdp.observations import (
    _detect_red_bbox_xyxy,
    _median_depth_at,
    _rgb_to_u8,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.move_cube_detection import variables as rv  # noqa: E402

DEFAULT_TASK = "ThesisRl-Move-Cube-Detection-UR5E"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "images"
YELLOW = (255, 220, 0)
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


def _detect_env0(env) -> tuple[tuple[int, int, int, int] | None, float, float, float | None]:
    """Same detector as ``cube_center_depth`` on env 0.

    Returns bbox ``(x0, y0, x1, y1)``, pixel center ``(x, y)``, and median
    depth in meters (``None`` when the window has no valid depth).
    """
    camera = env.scene["camera"]
    rgb = camera.data.output["rgb"]
    rgb_t = rgb.torch if hasattr(rgb, "torch") else rgb
    depth = camera.data.output["distance_to_image_plane"]
    depth_t = depth.torch if hasattr(depth, "torch") else depth
    depth_t = depth_t.to(dtype=torch.float32)
    if depth_t.ndim == 4:
        depth_t = depth_t[..., 0]

    rgb_u8 = _rgb_to_u8(rgb_t[..., :3])
    depth_np = depth_t.detach().to(device="cpu", dtype=torch.float32).numpy()

    xyxy = _detect_red_bbox_xyxy(
        rgb_u8[0],
        hsv_lower_1=rv.RED_HSV_LOWER_1,
        hsv_upper_1=rv.RED_HSV_UPPER_1,
        hsv_lower_2=rv.RED_HSV_LOWER_2,
        hsv_upper_2=rv.RED_HSV_UPPER_2,
        min_contour_area=rv.RED_MIN_CONTOUR_AREA,
    )
    if xyxy is None:
        return None, 0.0, 0.0, None

    x0, y0, x1, y1 = xyxy
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    depth_m = _median_depth_at(
        depth_np[0],
        cx,
        cy,
        half_window=rv.DEPTH_MEDIAN_HALF_WINDOW,
        max_depth=rv.DEPTH_MAX_METERS,
    )
    return (x0, y0, x1, y1), cx, cy, depth_m


def _label_lines(cx: float, cy: float, depth_m: float, width: int, height: int) -> list[str]:
    """Pixel center, meters, and the normalized triple fed to the policy."""
    xn = cx / float(width)
    yn = cy / float(height)
    depth_norm = float(np.clip(depth_m / rv.DEPTH_MAX_METERS, 0.0, 1.0))
    return [
        f"x: {cx:.1f} px",
        f"y: {cy:.1f} px",
        f"depth: {depth_m:.3f} m",
    ]


def _block_metrics(draw, lines: list[str], font, line_gap: int) -> tuple[int, int, list[int]]:
    widths: list[int] = []
    heights: list[int] = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    block_w = max(widths) if widths else 0
    block_h = sum(heights) + line_gap * max(0, len(lines) - 1)
    return block_w, block_h, heights


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + margin < bx0 or bx1 + margin < ax0 or ay1 + margin < by0 or by1 + margin < ay0)


def _place_beside_box(
    bbox: tuple[int, int, int, int],
    block_w: int,
    block_h: int,
    img_w: int,
    img_h: int,
    gap: int,
    pad: int,
) -> tuple[int, int]:
    """Place the label off the cube: beside the box, otherwise in the farthest clear corner."""
    x0, y0, x1, y1 = bbox
    label_w = block_w + 2 * pad
    label_h = block_h + 2 * pad
    margin = max(gap, 12)
    corners = (
        (margin, margin),
        (img_w - label_w - margin, margin),
        (margin, img_h - label_h - margin),
        (img_w - label_w - margin, img_h - label_h - margin),
    )
    candidates = (
        (x1 + gap, y0),
        (x0 - gap - label_w, y0),
        (x0, y0 - gap - label_h),
        (x0, y1 + gap),
        *corners,
    )
    clear: list[tuple[int, int]] = []
    for tx, ty in candidates:
        if tx < 0 or ty < 0 or tx + label_w > img_w or ty + label_h > img_h:
            continue
        rect = (tx, ty, tx + label_w, ty + label_h)
        if _rects_overlap(rect, bbox, margin):
            continue
        clear.append((tx, ty))
    if clear:
        return clear[0]
    # Every on-image spot hits the box. Use the corner farthest from its center.
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    return max(corners, key=lambda p: (p[0] + label_w / 2 - cx) ** 2 + (p[1] + label_h / 2 - cy) ** 2)


def _draw_frame(
    rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int] | None,
    cx: float,
    cy: float,
    depth_m: float | None,
    scale: int,
) -> np.ndarray:
    """Upscale RGB, draw a yellow bbox, and place center/depth text beside it."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Pillow is required (pip install pillow).") from exc

    h, w = rgb.shape[:2]
    view = Image.fromarray(rgb).resize((w * scale, h * scale), Image.NEAREST)
    draw = ImageDraw.Draw(view)
    font_size = max(22, int(0.045 * view.height))
    font = _load_font(font_size)
    line_gap = max(6, font_size // 5)
    pad = max(6, font_size // 6)

    if bbox_xyxy is None or depth_m is None:
        msg = "NO DETECTION" if bbox_xyxy is None else "NO VALID DEPTH"
        draw.rectangle([8, 8, 8 + font_size * 12, 8 + font_size + 2 * pad], fill=BLACK)
        draw.text((8 + pad, 8 + pad), msg, font=font, fill=(255, 80, 80))
        return np.asarray(view)

    lines = _label_lines(cx, cy, depth_m, w, h)
    block_w, block_h, heights = _block_metrics(draw, lines, font, line_gap)

    u0, v0, u1, v1 = bbox_xyxy
    box_pad = 1
    x0 = max(0, (u0 - box_pad) * scale)
    y0 = max(0, (v0 - box_pad) * scale)
    x1 = min(view.width - 1, (u1 + box_pad) * scale)
    y1 = min(view.height - 1, (v1 + box_pad) * scale)
    for t in range(max(3, scale // 2)):
        draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=YELLOW)

    # Mark the center the depth window is sampled around.
    ccx = int(round(cx * scale))
    ccy = int(round(cy * scale))
    arm = max(4, scale)
    draw.line([(ccx - arm, ccy), (ccx + arm, ccy)], fill=YELLOW, width=max(2, scale // 5))
    draw.line([(ccx, ccy - arm), (ccx, ccy + arm)], fill=YELLOW, width=max(2, scale // 5))

    gap = max(8, scale * 2)
    tx, ty = _place_beside_box((x0, y0, x1, y1), block_w, block_h, view.width, view.height, gap, pad)
    draw.rectangle([tx, ty, tx + block_w + 2 * pad, ty + block_h + 2 * pad], fill=BLACK)
    y = ty + pad
    for line, lh in zip(lines, heights):
        draw.text((tx + pad, y), line, font=font, fill=YELLOW)
        y += lh + line_gap

    return np.asarray(view)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save move-cube CameraLeft frames with yellow bbox, center xy, and depth."
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of annotated frames to save (scene is reset between captures).",
    )
    parser.add_argument(
        "--steps-after-reset",
        type=int,
        default=4,
        help="Sim steps after each reset before capturing (lets the camera render).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Nearest-neighbor upscale so the bbox and labels stay readable (camera is 128x128).",
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

                camera = unwrapped.scene["camera"]
                if hasattr(camera, "update"):
                    camera.update(0.0)

                rgb = _rgb_uint8(camera.data.output["rgb"])
                bbox, cx, cy, depth_m = _detect_env0(unwrapped)
                annotated = _draw_frame(
                    rgb,
                    bbox,
                    cx,
                    cy,
                    depth_m,
                    scale=max(1, int(args_cli.scale)),
                )

                path = out_dir / f"move_cube_detection_bbox_{i:02d}.png"
                try:
                    from PIL import Image
                except ImportError as exc:  # pragma: no cover
                    raise SystemExit("Pillow is required (pip install pillow).") from exc
                Image.fromarray(annotated).save(path)
                saved.append(path)

                if bbox is None or depth_m is None:
                    print(f"[INFO] Saved {path.name}  detection=none", flush=True)
                    continue
                h, w = rgb.shape[:2]
                xn = cx / float(w)
                yn = cy / float(h)
                depth_norm = float(np.clip(depth_m / rv.DEPTH_MAX_METERS, 0.0, 1.0))
                print(
                    f"[INFO] Saved {path.name}  "
                    f"bbox=[{bbox[0]},{bbox[1]}]–[{bbox[2]},{bbox[3]}]  "
                    f"center=({cx:.1f}, {cy:.1f}) px  "
                    f"depth={depth_m:.3f} m  "
                    f"policy=[{xn:.3f}, {yn:.3f}, {depth_norm:.3f}]",
                    flush=True,
                )

            print("\n==========================================")
            print(" MOVE-CUBE DETECTION VISUALIZATION")
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
