"""Pick-and-hold-at-goal rewards: reach, grasp, move cube to GOAL_POS while grasped."""
from __future__ import annotations

import torch

from thesis_rl.assets.ur5e import TABLE_HEIGHT

CUBE_SPAWN_HEIGHT = TABLE_HEIGHT + 0.027

# Fixed env-frame target: hold grasped cube near this point for success.
GOAL_POS = (-0.04, 0.32, 1.15)
GOAL_TOLERANCE = 0.05  # meters
# Dense move-to-goal uses the same proximity scales as reach.
GOAL_COARSE_SCALE = 0.30
GOAL_FINE_SCALE = 0.05

MAX_SUCCESS_SPEED = 0.1
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


def goal_position(env):
    """Broadcast fixed GOAL_POS to (num_envs, 3) in the env/training frame."""
    n = env.scene.num_envs
    device = env.scene["cube"].data.root_pos_w.torch.device
    dtype = env.scene["cube"].data.root_pos_w.torch.dtype
    return torch.tensor(GOAL_POS, device=device, dtype=dtype).unsqueeze(0).expand(n, -1)


def cube_to_goal_distance(env):
    return torch.linalg.vector_norm(cube_position(env) - goal_position(env), dim=-1)


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


def move_to_goal(env):
    """Dense guidance of cube toward GOAL_POS while grasping."""
    return grasp_gate(env).float() * _proximity(
        cube_to_goal_distance(env), GOAL_COARSE_SCALE, GOAL_FINE_SCALE
    )


def success(env):
    """Success: cube grasped, within GOAL_TOLERANCE of GOAL_POS, and nearly still."""
    speed = torch.linalg.vector_norm(env.scene["cube"].data.root_lin_vel_w.torch, dim=-1)
    return (
        grasp_gate(env)
        & (cube_to_goal_distance(env) < GOAL_TOLERANCE)
        & (speed < MAX_SUCCESS_SPEED)
    )


def cube_dropped(env):
    """Cube on the floor or off the table."""
    pos = cube_position(env)
    return (pos[:, 2] < 0.65) | (pos[:, 0].abs() > 0.65) | (pos[:, 1].abs() > 0.50)
