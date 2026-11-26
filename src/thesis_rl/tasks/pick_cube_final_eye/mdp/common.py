"""Minimal helpers for play: gripper/cube frames + episode terminations."""

from __future__ import annotations

import torch

from thesis_rl.tasks.pick_cube.mdp.rewards import GOAL_TOLERANCE, MAX_SUCCESS_SPEED, goal_position

CLOSED_FINGER_ANGLE = 0.15
GRASP_DISTANCE = 0.065


def gripper_position(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_bodies(["left_inner_finger", "right_inner_finger"], preserve_order=True)
    from isaaclab.utils.math import quat_apply
    positions = robot.data.body_pos_w.torch[:, ids]
    offsets = positions.new_tensor([[0.0, -0.045, 0.14], [0.0, 0.045, 0.14]])
    pads = positions + quat_apply(robot.data.body_quat_w.torch[:, ids], offsets.expand_as(positions))
    return pads.mean(dim=1)


def gripper_position_local(env):
    """Same training frame as pick_cube (env origin)."""
    return gripper_position(env) - env.scene.env_origins


def cube_position(env):
    """Simulator GT cube XYZ in env frame (debug / terminations only)."""
    return env.scene["cube"].data.root_pos_w.torch - env.scene.env_origins


def _fingers_closed(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints("finger_joint")
    return robot.data.joint_pos.torch[:, ids[0]] > CLOSED_FINGER_ANGLE


def _grasp_gate(env):
    distance = torch.linalg.vector_norm(
        gripper_position(env) - env.scene["cube"].data.root_pos_w.torch, dim=-1
    )
    return _fingers_closed(env) & (distance < GRASP_DISTANCE)


def success(env):
    """Success: cube grasped, within GOAL_TOLERANCE of GOAL_POS, and nearly still."""
    speed = torch.linalg.vector_norm(env.scene["cube"].data.root_lin_vel_w.torch, dim=-1)
    distance = torch.linalg.vector_norm(cube_position(env) - goal_position(env), dim=-1)
    return _grasp_gate(env) & (distance < GOAL_TOLERANCE) & (speed < MAX_SUCCESS_SPEED)


def cube_dropped(env):
    pos = cube_position(env)
    return (pos[:, 2] < 0.65) | (pos[:, 0].abs() > 0.65) | (pos[:, 1].abs() > 0.50)


def zero_reward(env):
    """Placeholder: ManagerBasedRLEnvCfg requires a rewards field; unused at play."""
    return torch.zeros(env.num_envs, device=env.device)
