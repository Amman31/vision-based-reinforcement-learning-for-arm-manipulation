"""RL observation integration: camera cube XYZ for the trained pick_cube actor.

Actor obs order (same as pick_cube training)::

    joint_pos, joint_vel*0.1, gripper_xyz, cube_xyz, goal_xyz, last_action
    history_length = 3
"""

from __future__ import annotations

import torch

from thesis_rl.tasks.pick_cube.mdp.rewards import goal_position

from .camera import read_rgb_depth
from .detection import detect_red_cube
from .geometry import estimate_cube_xyz

_MAX_JUMP_M = 0.12


def _ensure_cache(env, n_envs: int, device, dtype) -> None:
    if getattr(env, "_pick_final_cube_cache", None) is None or env._pick_final_cube_cache.shape[0] != n_envs:
        env._pick_final_cube_cache = torch.zeros(n_envs, 3, device=device, dtype=dtype)
        env._pick_final_cube_has = torch.zeros(n_envs, dtype=torch.bool, device=device)


def goal_xyz(env):
    """Fixed goal XYZ in the env frame (same as pick_cube ``GOAL_POS``)."""
    return goal_position(env)


def cube_position_from_camera(env):
    """Estimate cube XYZ in the training frame (env origin), shape (N, 3)."""
    rgb, depth = read_rgb_depth(env)
    u, v, z, valid, _mask = detect_red_cube(rgb, depth)
    estimate = estimate_cube_xyz(env, u, v, z, valid)

    n_envs = estimate.shape[0]
    _ensure_cache(env, n_envs, estimate.device, estimate.dtype)
    cache = env._pick_final_cube_cache
    has = env._pick_final_cube_has

    if hasattr(env, "episode_length_buf"):
        has = has & (env.episode_length_buf > 0)

    jump = (estimate - cache).norm(dim=-1)
    stable = valid & ((~has) | (jump < _MAX_JUMP_M))

    cache = torch.where(stable.unsqueeze(-1), estimate, cache)
    has = has | stable
    env._pick_final_cube_cache = cache
    env._pick_final_cube_has = has

    return torch.where(has.unsqueeze(-1), cache, estimate)
