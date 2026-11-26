# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for color-detection move-cube: [xn, yn, depth]."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .common import _as_torch, _cube_pos, _cube_vel, _min_finger_cube_dist, _spawn_xy, _torch, _xy_disp

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def _cv2():
    import cv2

    return cv2


def _rgb_to_u8(rgb: "torch.Tensor") -> np.ndarray:
    torch = _torch()
    rgb_cpu = rgb.detach().to(device="cpu")
    if rgb_cpu.dtype != torch.uint8:
        max_val = float(rgb_cpu.max().item()) if rgb_cpu.numel() else 0.0
        if max_val <= 1.0 + 1e-3:
            rgb_cpu = (rgb_cpu.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
        else:
            rgb_cpu = rgb_cpu.clamp(0.0, 255.0).to(dtype=torch.uint8)
    return rgb_cpu.numpy()


def _detect_red_bbox_xyxy(
    rgb_u8: np.ndarray,
    *,
    hsv_lower_1: tuple[int, int, int],
    hsv_upper_1: tuple[int, int, int],
    hsv_lower_2: tuple[int, int, int],
    hsv_upper_2: tuple[int, int, int],
    min_contour_area: float,
) -> tuple[int, int, int, int] | None:
    """Return (x_min, y_min, x_max, y_max) for the largest red contour, or None."""
    cv2 = _cv2()
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(hsv_lower_1, dtype=np.uint8), np.array(hsv_upper_1, dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array(hsv_lower_2, dtype=np.uint8), np.array(hsv_upper_2, dtype=np.uint8))
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_contour_area:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def _median_depth_at(
    depth: np.ndarray,
    u: float,
    v: float,
    *,
    half_window: int,
    max_depth: float,
) -> float | None:
    """Median of valid depth samples in a small window around (u, v)."""
    h, w = depth.shape[:2]
    ui = int(round(u))
    vi = int(round(v))
    x0 = max(0, ui - half_window)
    x1 = min(w, ui + half_window + 1)
    y0 = max(0, vi - half_window)
    y1 = min(h, vi + half_window + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth[y0:y1, x0:x1].astype(np.float32).reshape(-1)
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return None
    patch = patch[(patch > 1.0e-4) & (patch < max_depth)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def cube_center_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_depth: float = 3.0,
    half_window: int = 2,
    hsv_lower_1: tuple[int, int, int] = (0, 80, 80),
    hsv_upper_1: tuple[int, int, int] = (10, 255, 255),
    hsv_lower_2: tuple[int, int, int] = (170, 80, 80),
    hsv_upper_2: tuple[int, int, int] = (179, 255, 255),
    min_contour_area: float = 8.0,
) -> torch.Tensor:
    """Vision observation ``[xn, yn, depth_norm]``, shape ``(N, 3)``.

    - ``xn, yn``: bbox center normalized by image width/height (typically 128).
    - ``depth_norm``: median depth near the center, divided by ``max_depth`` → [0, 1].
    - Missed detection → zeros.
    """
    torch = _torch()
    camera = env.scene[sensor_cfg.name]
    rgb = _as_torch(camera.data.output["rgb"])[..., :3]
    depth = _as_torch(camera.data.output["distance_to_image_plane"]).to(dtype=torch.float32)
    if depth.ndim == 4:
        depth = depth[..., 0]

    rgb_u8 = _rgb_to_u8(rgb)
    depth_np = depth.detach().to(device="cpu", dtype=torch.float32).numpy()
    n, h, w, _ = rgb_u8.shape
    out = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        xyxy = _detect_red_bbox_xyxy(
            rgb_u8[i],
            hsv_lower_1=hsv_lower_1,
            hsv_upper_1=hsv_upper_1,
            hsv_lower_2=hsv_lower_2,
            hsv_upper_2=hsv_upper_2,
            min_contour_area=min_contour_area,
        )
        if xyxy is None:
            continue
        x0, y0, x1, y1 = xyxy
        x = 0.5 * (x0 + x1)
        y = 0.5 * (y0 + y1)
        z = _median_depth_at(depth_np[i], x, y, half_window=half_window, max_depth=max_depth)
        if z is None:
            continue
        out[i, 0] = x / float(w)
        out[i, 1] = y / float(h)
        out[i, 2] = float(np.clip(z / max_depth, 0.0, 1.0))

    return torch.as_tensor(out, device=env.device, dtype=torch.float32)


def privileged_state(env: ManagerBasedRLEnv, body_names: tuple[str, ...]) -> torch.Tensor:
    """Ground-truth critic state: cube vs grasp, spawn XY error, velocities, fingers."""
    torch = _torch()
    cube_pos = _cube_pos(env)
    grasp_pos = _as_torch(env.scene["grasp_frame"].data.target_pos_w)[:, 0, :]
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("finger_joint")
    finger_pos = _as_torch(robot.data.joint_pos)[:, joint_ids[0]].unsqueeze(-1)
    cube_to_grasp = cube_pos - grasp_pos
    disp_xy = cube_pos[:, 0:2] - _spawn_xy(env)
    return torch.cat(
        (
            cube_to_grasp,
            disp_xy,
            _cube_vel(env),
            finger_pos,
            _xy_disp(env).unsqueeze(-1),
            _min_finger_cube_dist(env, body_names).unsqueeze(-1),
        ),
        dim=1,
    )
