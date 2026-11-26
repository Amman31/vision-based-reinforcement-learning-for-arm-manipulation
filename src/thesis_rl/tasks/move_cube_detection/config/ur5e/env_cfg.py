# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from thesis_rl.assets.props import (
    CUBE_CFG,
    SPAWN_OUTLINE_X_NEG_CFG,
    SPAWN_OUTLINE_X_POS_CFG,
    SPAWN_OUTLINE_Y_NEG_CFG,
    SPAWN_OUTLINE_Y_POS_CFG,
    TABLE_CFG,
)
from thesis_rl.assets.ur5e import ASSET_DIR, UR5E_CFG

from ... import mdp

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rewards.move_cube_detection import reward_weights as rw  # noqa: E402
from rewards.move_cube_detection import variables as rv  # noqa: E402


@configclass
class MoveCubeDetectionPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3, cone="pyramidal", impratio=1, integrator="implicitfast"),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


def _finger_contact(body: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/" + body,
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )


@configclass
class MoveCubeDetectionSceneCfg(InteractiveSceneCfg):
    """Same tabletop layout as pick-cube, plus the ZED used for red-cube detection."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = UR5E_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UR5E_CFG.spawn.replace(activate_contact_sensors=True),
    )

    table = TABLE_CFG

    cube = CUBE_CFG.replace(spawn=CUBE_CFG.spawn.replace(activate_contact_sensors=True))

    spawn_outline_x_neg = SPAWN_OUTLINE_X_NEG_CFG
    spawn_outline_x_pos = SPAWN_OUTLINE_X_POS_CFG
    spawn_outline_y_neg = SPAWN_OUTLINE_Y_NEG_CFG
    spawn_outline_y_pos = SPAWN_OUTLINE_Y_POS_CFG

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )

    # ZED X aimed at the pick-cube table. Sensor name stays ``camera`` for the detector.
    zed_x = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZedX",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "depth_camera.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.3, 0.3, 1.2),
            rot=(0.3582, 0.1338, -0.8291, 0.40791),
        ),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ZedX/depth_camera/base_link/ZED_X/CameraLeft",
        spawn=None,
        width=128,
        height=128,
        data_types=["rgb", "distance_to_image_plane"],
        update_period=0.0,
        renderer_cfg=IsaacRtxRendererCfg(),
    )

    grasp_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur5e/wrist_3_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ur5e/wrist_3_link",
                name="grasp_point",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.13), rot=(0.0, 0.0, 0.0, 1.0)),
            )
        ],
    )
    left_inner_finger_contact = _finger_contact("left_inner_finger")
    right_inner_finger_contact = _finger_contact("right_inner_finger")
    left_outer_finger_contact = _finger_contact("left_outer_finger")
    right_outer_finger_contact = _finger_contact("right_outer_finger")


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.35,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.48},
    )


_DETECTION_PARAMS = {
    "sensor_cfg": SceneEntityCfg("camera"),
    "max_depth": rv.DEPTH_MAX_METERS,
    "half_window": rv.DEPTH_MEDIAN_HALF_WINDOW,
    "hsv_lower_1": rv.RED_HSV_LOWER_1,
    "hsv_upper_1": rv.RED_HSV_UPPER_1,
    "hsv_lower_2": rv.RED_HSV_LOWER_2,
    "hsv_upper_2": rv.RED_HSV_UPPER_2,
    "min_contour_area": rv.RED_MIN_CONTOUR_AREA,
}


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # Vision: [xn, yn, depth_norm] from OpenCV red-cube center + median depth.
        cube_xy_depth = ObsTerm(func=mdp.cube_center_depth, params=dict(_DETECTION_PARAMS))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        state = ObsTerm(func=mdp.privileged_state, params={"body_names": rv.FINGER_BODY_NAMES})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube = EventTerm(
        func=mdp.reset_cube_xy_radius,
        mode="reset",
        params={"radius": rv.CUBE_SPAWN_RADIUS, "asset_cfg": SceneEntityCfg("cube")},
    )


_SUCCESS = {
    "sensor_names": rv.FINGER_SENSOR_NAMES,
    "force_threshold": rv.CONTACT_FORCE_THRESHOLD,
    "success_distance": rv.SUCCESS_XY_DISPLACEMENT,
    "dwell_steps": rv.SUCCESS_DWELL_STEPS,
    "body_names": rv.FINGER_BODY_NAMES,
    "contact_distance": rv.CONTACT_DISTANCE,
}


@configclass
class RewardsCfg:
    approach = RewTerm(
        func=mdp.approach_reward,
        weight=rw.APPROACH,
        params={"body_names": rv.FINGER_BODY_NAMES, "distance_scale": rv.APPROACH_DISTANCE_SCALE},
    )
    contact = RewTerm(
        func=mdp.contact_reward,
        weight=rw.CONTACT,
        params={
            "sensor_names": rv.FINGER_SENSOR_NAMES,
            "force_threshold": rv.CONTACT_FORCE_THRESHOLD,
            "body_names": rv.FINGER_BODY_NAMES,
            "contact_distance": rv.CONTACT_DISTANCE,
        },
    )
    push = RewTerm(
        func=mdp.push_reward,
        weight=rw.PUSH,
        params={
            "sensor_names": rv.FINGER_SENSOR_NAMES,
            "force_threshold": rv.CONTACT_FORCE_THRESHOLD,
            "success_distance": rv.SUCCESS_XY_DISPLACEMENT,
            "body_names": rv.FINGER_BODY_NAMES,
            "contact_distance": rv.CONTACT_DISTANCE,
        },
    )
    success = RewTerm(func=mdp.success_reward, weight=rw.SUCCESS, params=dict(_SUCCESS))
    step_penalty = RewTerm(func=mdp.step_penalty, weight=rw.STEP_PENALTY)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=rw.ACTION_RATE)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.cube_moved, params=dict(_SUCCESS))


@configclass
class MoveCubeDetectionEnvCfg(ManagerBasedRLEnvCfg):
    """Push the cube at least 1 cm in XY with any gripper finger."""

    scene: MoveCubeDetectionSceneCfg = MoveCubeDetectionSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = rv.EPISODE_LENGTH_S
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, 2.0, 2.0), lookat=(0.0, 0.0, 0.9))
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MoveCubeDetectionPhysicsCfg()
