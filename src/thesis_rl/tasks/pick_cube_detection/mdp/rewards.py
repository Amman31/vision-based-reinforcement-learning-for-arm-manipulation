"""Pick-and-lift rewards: reach, grasp, raise the cube. Success = stable lift."""
from __future__ import annotations

import torch

from thesis_rl.assets.ur5e import TABLE_HEIGHT

CUBE_SPAWN_HEIGHT = TABLE_HEIGHT + 0.027
# Dense lift guidance saturates here.
LIFT_HEIGHT = TABLE_HEIGHT + 0.12
# Success: grasped cube clearly off the table and nearly still.
SUCCESS_HEIGHT = TABLE_HEIGHT + 0.08
MAX_SUCCESS_SPEED = 0.2
CLOSED_FINGER_ANGLE = 0.15
GRASP_DISTANCE = 0.065

REACH_COARSE_SCALE = 0.30
REACH_FINE_SCALE = 0.05


def gripper_position(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_bodies(["left_inner_finger", "right_inner_finger"], preserve_order=True)
    from isaaclab.utils.math import quat_apply
    positions = robot.data.body_pos_w.torch[:, ids]
    offsets = positions.new_tensor([[0.0, -0.045, 0.14], [0.0, 0.045, 0.14]])
    pads = positions + quat_apply(robot.data.body_quat_w.torch[:, ids], offsets.expand_as(positions))
    return pads.mean(dim=1)


def gripper_position_local(env):
    return gripper_position(env) - env.scene.env_origins


def cube_position(env):
    return env.scene["cube"].data.root_pos_w.torch - env.scene.env_origins


def _proximity(distance, coarse: float, fine: float):
    near = 1.0 - torch.tanh(distance / coarse)
    close = 1.0 - torch.tanh(distance / fine)
    return 0.5 * (near + close)


def reach_cube(env):
    distance = torch.linalg.vector_norm(
        gripper_position(env) - env.scene["cube"].data.root_pos_w.torch, dim=-1
    )
    return _proximity(distance, REACH_COARSE_SCALE, REACH_FINE_SCALE)


def fingers_closed(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints("finger_joint")
    return robot.data.joint_pos.torch[:, ids[0]] > CLOSED_FINGER_ANGLE


def grasp_gate(env):
    distance = torch.linalg.vector_norm(
        gripper_position(env) - env.scene["cube"].data.root_pos_w.torch, dim=-1
    )
    return fingers_closed(env) & (distance < GRASP_DISTANCE)


def lift_cube(env):
    """Dense guidance up to LIFT_HEIGHT while grasping."""
    height = cube_position(env)[:, 2]
    progress = ((height - CUBE_SPAWN_HEIGHT) / (LIFT_HEIGHT - CUBE_SPAWN_HEIGHT)).clamp(0.0, 1.0)
    return grasp_gate(env).float() * progress


def success(env):
    """Success: cube is grasped, above SUCCESS_HEIGHT, and nearly still."""
    speed = torch.linalg.vector_norm(env.scene["cube"].data.root_lin_vel_w.torch, dim=-1)
    height = cube_position(env)[:, 2]
    return grasp_gate(env) & (height > SUCCESS_HEIGHT) & (speed < MAX_SUCCESS_SPEED)


def cube_dropped(env):
    """Cube on the floor or off the table."""
    pos = cube_position(env)
    return (pos[:, 2] < 0.65) | (pos[:, 0].abs() > 0.65) | (pos[:, 1].abs() > 0.50)
