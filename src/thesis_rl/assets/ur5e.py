"""UR5e and Robotiq 2F-85 configuration for Astra's tabletop task.

The installed Isaac Lab checkout uses XYZW quaternions. The assembled USD
references NVIDIA-hosted robot payloads, which must be accessible or cached.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = Path(__file__).resolve().parent
TABLE_HEIGHT = 0.75
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "ur5e.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Compensate for the authored arm translation to place its base at
        # (-0.4, 0.0, TABLE_HEIGHT) in the environment frame.
        pos=(-0.2987749048307991, -0.20169373578909866, TABLE_HEIGHT),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.3708,
            "elbow_joint": 1.5708,
            "wrist_1_joint": -1.7708,
            "wrist_2_joint": -1.5708,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            ".*_inner_finger_joint": 0.0,
            ".*_inner_finger_knuckle_joint": 0.0,
            ".*_outer_.*_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            stiffness=400.0,
            damping=40.0,
            joint_effort_limit={
                "shoulder_.*": 150.0,
                "elbow_joint": 150.0,
                "wrist_.*": 28.0,
            },
            joint_velocity_limit=2.0,
        ),
        # The USD mimic constraints couple the other finger joints.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            joint_effort_limit=10.0,
            joint_velocity_limit=1.0,
            stiffness=40.0,
            damping=2.0,
        ),
    },
)

__all__ = ["ARM_JOINTS", "ASSET_DIR", "TABLE_HEIGHT", "UR5E_CFG"]


