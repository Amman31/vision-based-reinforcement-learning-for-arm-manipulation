# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for move-cube domain randomization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .common import _as_torch, _torch

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv


class reset_cube_xy_radius(ManagerTermBase):
    """Respawn the cube uniformly in a disk of ``radius`` around its default XY."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        torch = _torch()
        # EventManager is constructed before physics starts. Do not read GPU pose buffers here.
        env._cube_spawn_xy = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)

    def __call__(
        self, env: ManagerBasedEnv, env_ids: torch.Tensor, radius: float, asset_cfg: SceneEntityCfg | None = None
    ):
        torch = _torch()
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg("cube")
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        cube = env.scene[asset_cfg.name]
        default_pose = _as_torch(cube.data.default_root_pose)
        if default_pose.ndim == 1:
            default_pose = default_pose.unsqueeze(0)
        default_pose = default_pose[env_ids]
        n = env_ids.shape[0]
        u = torch.rand(n, 2, device=env.device)
        r, theta = radius * torch.sqrt(u[:, 1]), u[:, 0] * (2.0 * math.pi)
        offset = torch.stack((r * torch.cos(theta), r * torch.sin(theta), torch.zeros(n, device=env.device)), dim=1)
        pos = default_pose[:, 0:3] + env.scene.env_origins[env_ids] + offset
        cube.write_root_pose_to_sim_index(root_pose=torch.cat((pos, default_pose[:, 3:7]), dim=-1), env_ids=env_ids)
        cube.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(n, 6, device=env.device, dtype=torch.float32), env_ids=env_ids
        )
        env._cube_spawn_xy[env_ids] = pos[:, 0:2]
