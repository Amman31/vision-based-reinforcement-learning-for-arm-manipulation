"""Policy observations for XYZ sim-to-real (noisy cube pose as if from a camera)."""
from __future__ import annotations

import torch

from .rewards import cube_position, goal_position


def cube_position_observed(env, noise_std: float = 0.005):
    """Cube XYZ in the env frame, with Gaussian noise to mimic perception error.

    Same frame as ``gripper_position_local`` / ``cube_position`` (env origin, not
    necessarily the URDF base link). Real-world detections must be transformed
    into this same frame before being fed to the trained policy.
    """
    cube_pos = cube_position(env)
    if noise_std <= 0.0:
        return cube_pos
    return cube_pos + torch.randn_like(cube_pos) * noise_std


def goal_xyz(env):
    """Fixed goal XYZ in the env frame (same as ``GOAL_POS`` in rewards)."""
    return goal_position(env)
