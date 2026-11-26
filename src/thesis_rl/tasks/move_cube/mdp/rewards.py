# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for pushing a cube with any gripper finger."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .common import _any_finger_contact, _mean_finger_cube_dist, _torch, _xy_disp

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv


def approach_reward(env, body_names: tuple[str, ...], distance_scale: float):
    torch = _torch()
    return 1.0 - torch.tanh(distance_scale * _mean_finger_cube_dist(env, body_names))


def contact_reward(
    env, sensor_names: tuple[str, ...], force_threshold: float, body_names: tuple[str, ...], contact_distance: float
):
    return _any_finger_contact(env, sensor_names, force_threshold, body_names, contact_distance).float()


def push_reward(
    env,
    sensor_names: tuple[str, ...],
    force_threshold: float,
    success_distance: float,
    body_names: tuple[str, ...],
    contact_distance: float,
):
    torch = _torch()
    contacted = _any_finger_contact(env, sensor_names, force_threshold, body_names, contact_distance)
    return contacted.float() * torch.clamp(_xy_disp(env) / max(success_distance, 1.0e-6), min=0.0, max=1.0)


class success_reward(ManagerTermBase):
    """One-shot bonus once any finger has touched the cube and it has moved in XY."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        torch = _torch()
        self._contacted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._dwell = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        self._contacted[env_ids] = False
        self._dwell[env_ids] = 0

    def __call__(
        self,
        env,
        sensor_names: tuple[str, ...],
        force_threshold: float,
        success_distance: float,
        dwell_steps: int,
        body_names: tuple[str, ...],
        contact_distance: float,
    ):
        torch = _torch()
        self._contacted |= _any_finger_contact(env, sensor_names, force_threshold, body_names, contact_distance)
        valid = self._contacted & (_xy_disp(env) >= success_distance)
        self._dwell = torch.where(valid, self._dwell + 1, torch.zeros_like(self._dwell))
        return (self._dwell == dwell_steps).float()


def step_penalty(env):
    torch = _torch()
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)


def action_rate_l2(env):
    torch = _torch()
    if env.action_manager.action is None or env.action_manager.prev_action is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
