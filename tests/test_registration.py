# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the generated task registrations."""

import gymnasium as gym

import thesis_rl.tasks  # noqa: F401


def test_task_registrations():
    """The generated tasks must expose valid environment and agent entry points."""
    expected = {
        "ThesisRl-Move-Cube-UR5E": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "thesis_rl.tasks.move_cube.config.ur5e.env_cfg:MoveCubeEnvCfg",
            "default_agent": "rsl_rl",
        },
        "ThesisRl-Move-Cube-Detection-UR5E": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "thesis_rl.tasks.move_cube_detection.config.ur5e.env_cfg:MoveCubeDetectionEnvCfg",
            "default_agent": "rsl_rl",
        },
        "ThesisRl-Pick-Cube-UR5E": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "thesis_rl.tasks.pick_cube.config.ur5e.env_cfg:PickCubeEnvCfg",
            "default_agent": "rsl_rl",
        },
        "ThesisRl-Pick-Cube-Final-Rb-UR5E": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "thesis_rl.tasks.pick_cube_final_rb.config.ur5e.env_cfg:PickCubeFinalRbEnvCfg",
            "default_agent": "rsl_rl",
        },
        "ThesisRl-Pick-Place-Cube-UR5E": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "thesis_rl.tasks.pick_place_cube.config.ur5e.env_cfg:PickPlaceCubeEnvCfg",
            "default_agent": "rsl_rl",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
