"""Pixel+depth → camera XYZ → training-frame XYZ, plus two-camera fusion."""

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


def optical_axis_world(env, sensor_name: str) -> torch.Tensor:
    """Unit camera +Z (optical axis) in world axes, shape ``(N, 3)``."""
    camera = env.scene[sensor_name]
    quat = camera.data.quat_w_ros.torch
    axis = quat.new_tensor([0.0, 0.0, 1.0]).expand(quat.shape[0], 3)
    axis = quat_apply(quat, axis)
    return axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


# Image-plane error is small; depth error lies along the optical axis and grows with range.
_SIGMA_XY_M = 0.006
_SIGMA_Z_M = 0.02
_DEPTH_REF_M = 0.8


def _axis_information(axis: torch.Tensor, sigma_z: torch.Tensor) -> torch.Tensor:
    """Inverse covariance ``(N, 3, 3)``: tight across the ray, loose along it."""
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(axis.shape[0], 3, 3)
    aa = axis.unsqueeze(-1) @ axis.unsqueeze(-2)
    inv_xy = 1.0 / (_SIGMA_XY_M ** 2)
    inv_z = 1.0 / sigma_z.square().clamp_min(1.0e-8)
    return inv_xy * eye - (inv_xy - inv_z).view(-1, 1, 1) * aa


def fuse_cube_xyz(
    p0: torch.Tensor,
    valid0: torch.Tensor,
    axis0: torch.Tensor,
    z0: torch.Tensor,
    p1: torch.Tensor,
    valid1: torch.Tensor,
    axis1: torch.Tensor,
    z1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse two env-frame cube estimates. One camera is used when the other misses.

    Each camera is trusted in its image plane and down-weighted along its optical
    axis, so a side view corrects the depth error of the front view (and the reverse).
    """
    scale0 = (z0 / _DEPTH_REF_M).clamp(0.5, 3.0)
    scale1 = (z1 / _DEPTH_REF_M).clamp(0.5, 3.0)
    info0 = _axis_information(axis0, _SIGMA_Z_M * scale0)
    info1 = _axis_information(axis1, _SIGMA_Z_M * scale1)
    rhs = info0 @ p0.unsqueeze(-1) + info1 @ p1.unsqueeze(-1)
    fused = torch.linalg.solve(info0 + info1, rhs).squeeze(-1)

    both = valid0 & valid1
    out = torch.zeros_like(p0)
    out = torch.where(both.unsqueeze(-1), fused, out)
    out = torch.where((valid0 & ~valid1).unsqueeze(-1), p0, out)
    out = torch.where((valid1 & ~valid0).unsqueeze(-1), p1, out)
    return out, valid0 | valid1
