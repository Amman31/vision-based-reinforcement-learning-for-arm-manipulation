# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations for move-cube."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .common import _any_finger_contact, _torch, _xy_disp

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv


class cube_moved(ManagerTermBase):
    """Success when any finger has touched the cube and XY displacement exceeds the threshold."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
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
        return self._dwell >= dwell_steps
