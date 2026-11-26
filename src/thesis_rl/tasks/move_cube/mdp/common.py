# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the move-cube MDP."""

from __future__ import annotations


def _torch():
    import torch

    return torch


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def _cube_pos(env):
    return _as_torch(env.scene["cube"].data.root_link_pose_w)[:, 0:3]


def _cube_vel(env):
    return _as_torch(env.scene["cube"].data.root_link_vel_w)[:, 0:3]


def _spawn_xy(env):
    if hasattr(env, "_cube_spawn_xy"):
        return env._cube_spawn_xy
    return _cube_pos(env)[:, 0:2]


def _xy_disp(env):
    torch = _torch()
    return torch.linalg.vector_norm(_cube_pos(env)[:, 0:2] - _spawn_xy(env), dim=1)


def _body_pos(env, body_name: str):
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    return _as_torch(robot.data.body_link_pos_w)[:, body_ids[0], :]


def _min_finger_cube_dist(env, body_names: tuple[str, ...]):
    torch = _torch()
    cube = _cube_pos(env)
    dist = torch.full((env.num_envs,), 1.0e6, device=env.device, dtype=torch.float32)
    for name in body_names:
        dist = torch.minimum(dist, torch.linalg.vector_norm(_body_pos(env, name) - cube, dim=1))
    return dist


def _mean_finger_cube_dist(env, body_names: tuple[str, ...]):
    torch = _torch()
    cube = _cube_pos(env)
    dists = [torch.linalg.vector_norm(_body_pos(env, name) - cube, dim=1) for name in body_names]
    return torch.stack(dists).mean(dim=0)


def _finger_force(env, sensor_name: str):
    torch = _torch()
    data = env.scene[sensor_name].data
    for attr in ("normal_force_matrix_w", "force_matrix_w", "net_forces_w"):
        forces = getattr(data, attr, None)
        if forces is None:
            continue
        forces = _as_torch(forces)
        if forces.numel() == 0:
            continue
        mag = torch.linalg.vector_norm(forces, dim=-1)
        return mag.reshape(mag.shape[0], -1).amax(dim=1) if mag.ndim > 1 else mag
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def _any_finger_contact(
    env, sensor_names: tuple[str, ...], force_threshold: float, body_names: tuple[str, ...], contact_distance: float
):
    torch = _torch()
    touching = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in sensor_names:
        touching |= _finger_force(env, name) > force_threshold
    cube = _cube_pos(env)
    for name in body_names:
        touching |= torch.linalg.vector_norm(_body_pos(env, name) - cube, dim=1) < contact_distance
    return touching
