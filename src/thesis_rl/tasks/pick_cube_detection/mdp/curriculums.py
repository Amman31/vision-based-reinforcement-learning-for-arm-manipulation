"""Spawn-range curriculum for pick_cube_detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import modify_term_cfg

from thesis_rl.assets.props import CUBE_SPAWN_HALF_XY

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Must match PPORunnerCfg.num_steps_per_env (RSL-RL collects this many steps per iter).
_STEPS_PER_ITER = 32

# (min_iteration, half_xy, yaw_half). Final stage matches CUBE_SPAWN_HALF_XY (±10 cm).
SPAWN_CURRICULUM_STAGES: list[tuple[int, tuple[float, float], float]] = [
    (0, (0.05, 0.05), 0.20),
    (4000, (0.075, 0.075), 0.35),
    (8000, CUBE_SPAWN_HALF_XY, 0.50),
]


def pose_range_for_half(half_xy: tuple[float, float], yaw_half: float) -> dict[str, tuple[float, float]]:
    return {
        "x": (-half_xy[0], half_xy[0]),
        "y": (-half_xy[1], half_xy[1]),
        "yaw": (-yaw_half, yaw_half),
    }


def initial_spawn_pose_range() -> dict[str, tuple[float, float]]:
    """Pose range used at env construction (stage 0)."""
    _, half_xy, yaw_half = SPAWN_CURRICULUM_STAGES[0]
    return pose_range_for_half(half_xy, yaw_half)


def _stage_pose_range(common_step_counter: int) -> dict[str, tuple[float, float]]:
    approx_iter = common_step_counter // _STEPS_PER_ITER
    half_xy, yaw_half = SPAWN_CURRICULUM_STAGES[0][1], SPAWN_CURRICULUM_STAGES[0][2]
    for min_iter, stage_half_xy, stage_yaw in SPAWN_CURRICULUM_STAGES:
        if approx_iter >= min_iter:
            half_xy, yaw_half = stage_half_xy, stage_yaw
        else:
            break
    return pose_range_for_half(half_xy, yaw_half)


def expand_cube_spawn_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: dict,
) -> dict | object:
    """Grow ``events.reset_cube`` pose_range with training progress."""
    new_value = _stage_pose_range(env.common_step_counter)
    if old_value == new_value:
        return modify_term_cfg.NO_CHANGE
    return new_value
