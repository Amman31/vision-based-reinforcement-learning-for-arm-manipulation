"""RL observation integration: fused two-camera cube XYZ for the trained pick_cube actor.

Actor obs order (same as pick_cube training)::

    joint_pos, joint_vel*0.1, gripper_xyz, cube_xyz, goal_xyz, last_action
    history_length = 3

``cube_xyz`` is one 3-vector. Front and side ZED detections are fused; the
policy layout does not change.
"""

from __future__ import annotations

import torch

from thesis_rl.tasks.pick_cube.mdp.rewards import goal_position

from .camera import read_rgb_depth
from .detection import detect_red_cube
from .geometry import estimate_cube_xyz, fuse_cube_xyz, optical_axis_world

_MAX_JUMP_M = 0.12
_SENSORS = ("depth_camera", "depth_camera_side")


def _ensure_cache(env, n_envs: int, device, dtype) -> None:
    if getattr(env, "_pick_dual_cube_cache", None) is None or env._pick_dual_cube_cache.shape[0] != n_envs:
        env._pick_dual_cube_cache = torch.zeros(n_envs, 3, device=device, dtype=dtype)
        env._pick_dual_cube_has = torch.zeros(n_envs, dtype=torch.bool, device=device)


def goal_xyz(env):
    """Fixed goal XYZ in the env frame (same as pick_cube ``GOAL_POS``)."""
    return goal_position(env)


def _detect(env, sensor_name: str):
    rgb, depth = read_rgb_depth(env, sensor_name=sensor_name)
    u, v, z, valid, _mask = detect_red_cube(rgb, depth)
    estimate = estimate_cube_xyz(env, u, v, z, valid, sensor_name=sensor_name)
    axis = optical_axis_world(env, sensor_name)
    return estimate, valid, axis, z


def cube_position_from_cameras(env):
    """Fused cube XYZ in the training frame (env origin), shape (N, 3)."""
    p0, valid0, axis0, z0 = _detect(env, _SENSORS[0])
    p1, valid1, axis1, z1 = _detect(env, _SENSORS[1])
    estimate, valid = fuse_cube_xyz(p0, valid0, axis0, z0, p1, valid1, axis1, z1)

    n_envs = estimate.shape[0]
    _ensure_cache(env, n_envs, estimate.device, estimate.dtype)
    cache = env._pick_dual_cube_cache
    has = env._pick_dual_cube_has

    if hasattr(env, "episode_length_buf"):
        has = has & (env.episode_length_buf > 0)

    jump = (estimate - cache).norm(dim=-1)
    stable = valid & ((~has) | (jump < _MAX_JUMP_M))

    cache = torch.where(stable.unsqueeze(-1), estimate, cache)
    has = has | stable
    env._pick_dual_cube_cache = cache
    env._pick_dual_cube_has = has

    return torch.where(has.unsqueeze(-1), cache, estimate)
