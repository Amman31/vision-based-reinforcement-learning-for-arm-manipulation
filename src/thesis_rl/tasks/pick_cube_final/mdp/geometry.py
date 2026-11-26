"""Pixel+depth → camera XYZ → training-frame XYZ (env origin, same as pick_cube)."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

# 5 cm cube: depth hits the visible face; training GT is the cube root/center.
CUBE_HALF_SIZE_M = 0.025


def unproject_uvz(u, v, z, intrinsics: torch.Tensor) -> torch.Tensor:
    """``(u,v,Z)`` + K → camera-frame points ``(N, 3)`` (ROS / OpenCV axes)."""
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.stack((x, y, z), dim=-1)


def surface_to_center_camera(p_cam: torch.Tensor, half_size: float = CUBE_HALF_SIZE_M) -> torch.Tensor:
    """Approx. cube center from frontal-surface hit: push +half_size along optical +Z."""
    out = p_cam.clone()
    out[:, 2] = out[:, 2] + half_size
    return out


def camera_to_training_frame(env, p_cam: torch.Tensor, sensor_name: str = "depth_camera") -> torch.Tensor:
    """Camera XYZ → world → env-origin frame (same as ``gripper_xyz`` / pick_cube ``cube_xyz``).

    pick_cube training used ``root_pos_w - env_origins``, not a body-fixed robot
    base frame. Matching that frame is required for the trained .pt to work.
    """
    camera = env.scene[sensor_name]
    p_world = quat_apply(camera.data.quat_w_ros.torch, p_cam) + camera.data.pos_w.torch
    return p_world - env.scene.env_origins


def estimate_cube_xyz(
    env,
    u,
    v,
    z,
    valid: torch.Tensor,
    sensor_name: str = "depth_camera",
    apply_surface_offset: bool = True,
) -> torch.Tensor:
    """Full geometry chain → training-frame cube XYZ ``(N, 3)`` (zeros where invalid)."""
    camera = env.scene[sensor_name]
    p_cam = unproject_uvz(u, v, z, camera.data.intrinsic_matrices.torch)
    if apply_surface_offset:
        p_cam = surface_to_center_camera(p_cam)
    p_env = camera_to_training_frame(env, p_cam, sensor_name=sensor_name)
    return torch.where(valid.unsqueeze(-1), p_env, torch.zeros_like(p_env))
