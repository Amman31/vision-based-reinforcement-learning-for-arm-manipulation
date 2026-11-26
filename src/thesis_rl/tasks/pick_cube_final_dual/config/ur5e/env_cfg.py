"""Play-only env: trained pick_cube .pt + two ZED X cameras.

Policy input matches pick_cube training::

    joint_pos, joint_vel*0.1, gripper_xyz, cube_xyz, goal_xyz, last_action  (history=3)

``cube_xyz`` is the fused estimate from the front CameraLeft and a second
ZED orbited 90 degrees to the -Y side. ``goal_xyz`` is the fixed hold target
from pick_cube. No training rewards/curriculum.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_tasks.utils import PresetCfg

from thesis_rl.assets.props import (
    CUBE_CFG,
    CUBE_SPAWN_HALF_XY,
    SPAWN_OUTLINE_X_NEG_CFG,
    SPAWN_OUTLINE_X_POS_CFG,
    SPAWN_OUTLINE_Y_NEG_CFG,
    SPAWN_OUTLINE_Y_POS_CFG,
    TABLE_CFG,
)
from thesis_rl.assets.ur5e import ARM_JOINTS, ASSET_DIR, UR5E_CFG
from ... import mdp

RANDOMIZE_CUBE_POSE = True

# Full workspace (±10 cm) — no curriculum for play.
_SPAWN_POSE_RANGE = {
    "x": (-CUBE_SPAWN_HALF_XY[0], CUBE_SPAWN_HALF_XY[0]),
    "y": (-CUBE_SPAWN_HALF_XY[1], CUBE_SPAWN_HALF_XY[1]),
    "yaw": (-0.50, 0.50),
}


@configclass
class PickCubeFinalDualPhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg()
    default = isaacsim_physx


@configclass
class PickCubeFinalDualSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    robot = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = TABLE_CFG
    cube = CUBE_CFG
    spawn_outline_x_neg = SPAWN_OUTLINE_X_NEG_CFG
    spawn_outline_x_pos = SPAWN_OUTLINE_X_POS_CFG
    spawn_outline_y_neg = SPAWN_OUTLINE_Y_NEG_CFG
    spawn_outline_y_pos = SPAWN_OUTLINE_Y_POS_CFG
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )
    zed_x = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZedX",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "depth_camera.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 1.2),
            rot=(0.33399, -0.01062, -0.94246, -0.01062),
        ),
    )
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ZedX/depth_camera/base_link/ZED_X/CameraLeft",
        spawn=None,
        width=400,
        height=400,
        data_types=["rgb", "distance_to_image_plane"],
        update_period=0.0,
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    # Same ZED, rigid-orbited -90 deg about Z around the cube spawn (0.20, 0).
    # Front view stays on +X; this one looks in from -Y, away from the hold goal.
    zed_x_side = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZedXSide",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "depth_camera.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.2, -0.3, 1.2),
            rot=(0.22866, -0.24368, -0.65891, -0.67393),
        ),
    )
    depth_camera_side = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ZedXSide/depth_camera/base_link/ZED_X/CameraLeft",
        spawn=None,
        width=400,
        height=400,
        data_types=["rgb", "distance_to_image_plane"],
        update_period=0.0,
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    lab_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Lab_Room",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "lab_room.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.01172, 0.0, -0.05248)),
    )


@configclass
class ActionsCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=0.75,
        use_default_offset=True, preserve_order=True, clip={".*": (-3.14, 3.14)},
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0}, close_command_expr={"finger_joint": 0.48},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        gripper_xyz = ObsTerm(func=mdp.gripper_position_local)
        cube_xyz = ObsTerm(func=mdp.cube_position_from_cameras)
        goal_xyz = ObsTerm(func=mdp.goal_xyz)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3

    @configclass
    class CriticCfg(PolicyCfg):
        # Must exist so the pick_cube .pt critic can load (unused for actions).
        cube_ground_truth = ObsTerm(func=mdp.cube_position)

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": _SPAWN_POSE_RANGE,
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Required by ManagerBasedRLEnvCfg; unused during play."""

    unused = RewTerm(func=mdp.zero_reward, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    success = DoneTerm(func=mdp.success)


@configclass
class PickCubeFinalDualEnvCfg(ManagerBasedRLEnvCfg):
    scene = PickCubeFinalDualSceneCfg(num_envs=1, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = None

    def __post_init__(self):
        if not RANDOMIZE_CUBE_POSE:
            self.events.reset_cube = None
        self.rerender_on_reset = True
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PickCubeFinalDualPhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, 2.0, 2.0), lookat=(0.0, 0.0, 0.9))
