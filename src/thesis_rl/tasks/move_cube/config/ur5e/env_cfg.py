# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from thesis_rl.assets.ur5e import UR5E_CFG

from ... import mdp

_REPO_ROOT = Path(__file__).resolve().parents[6]
_ASSETS = _REPO_ROOT / "src" / "thesis_rl" / "assets"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rewards.move_cube import reward_weights as rw  # noqa: E402
from rewards.move_cube import variables as rv  # noqa: E402


@configclass
class MoveCubePhysicsCfg(PresetCfg):
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
        filter_prim_paths_expr=["{ENV_REGEX_NS}/cube/.*"],
        debug_vis=False,
    )


@configclass
class MoveCubeSceneCfg(InteractiveSceneCfg):
    """UR5e + Robotiq + RGB-D + cube on table. No bowl."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot: ArticulationCfg = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_ASSETS / "table.usd"), scale=(0.0014, 0.00181, 0.00112)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.899038, 0.0, 0.074920), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    depth_camera_body = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/depth_camera",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_ASSETS / "depth_camera.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.4, 0.2, 1.3), rot=(0.41, 0.11, -0.87, 0.31)),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/depth_camera/depth_camera/base_link/ZED_X/CameraLeft",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=None,
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSETS / "cube.usd"),
            scale=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=rv.CUBE_START_POS, rot=(0.0, 0.0, 0.0, 1.0)),
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
        close_command_expr={"finger_joint": rv.GRASP_FINGER_TARGET},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RgbdCfg(ObsGroup):
        rgbd = ObsTerm(
            func=mdp.rgbd_image_chw,
            params={"sensor_cfg": SceneEntityCfg("camera"), "max_depth": rv.DEPTH_MAX_METERS},
        )

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
    rgbd: RgbdCfg = RgbdCfg()
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
class MoveCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Push the cube at least 1 cm in XY with any gripper finger."""

    scene: MoveCubeSceneCfg = MoveCubeSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = rv.EPISODE_LENGTH_S
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MoveCubePhysicsCfg()
