# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for RGB-D move-cube."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import _as_torch, _cube_pos, _cube_vel, _min_finger_cube_dist, _spawn_xy, _torch, _xy_disp

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def rgbd_image_chw(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_depth: float = 3.0) -> torch.Tensor:
    """Normalized RGB-D observation with shape (N, 4, H, W)."""
    torch = _torch()
    camera = env.scene[sensor_cfg.name]
    rgb = _as_torch(camera.data.output["rgb"]).to(dtype=torch.float32)[..., :3] / 255.0
    depth = _as_torch(camera.data.output["distance_to_image_plane"]).to(dtype=torch.float32)
    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
    depth = torch.clamp(depth, min=0.0, max=max_depth) / max_depth
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    elif depth.shape[-1] != 1:
        depth = depth[..., :1]
    return torch.cat((rgb, depth), dim=-1).permute(0, 3, 1, 2).contiguous()


def rgb_image_chw(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    torch = _torch()
    rgb = _as_torch(env.scene[sensor_cfg.name].data.output["rgb"]).to(dtype=torch.float32)[..., :3] / 255.0
    return rgb.permute(0, 3, 1, 2).contiguous()


def depth_image_chw(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_depth: float = 3.0) -> torch.Tensor:
    torch = _torch()
    depth = _as_torch(env.scene[sensor_cfg.name].data.output["distance_to_image_plane"]).to(dtype=torch.float32)
    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
    depth = torch.clamp(depth, min=0.0, max=max_depth) / max_depth
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    elif depth.shape[-1] != 1:
        depth = depth[..., :1]
    return depth.permute(0, 3, 1, 2).contiguous()


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
