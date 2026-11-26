"""Camera-driven UR5e cube pickup for the installed Isaac Sim / PhysX backend."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
    SPAWN_OUTLINE_X_NEG_CFG,
    SPAWN_OUTLINE_X_POS_CFG,
    SPAWN_OUTLINE_Y_NEG_CFG,
    SPAWN_OUTLINE_Y_POS_CFG,
    TABLE_CFG,
)

from thesis_rl.assets.ur5e import ARM_JOINTS, UR5E_CFG, ASSET_DIR
from ... import mdp

# True: randomize cube XY position and yaw at each episode reset.
# False: reset the cube to scene.cube.init_state every episode.
RANDOMIZE_CUBE_POSE = True


@configclass
class PickCubeDepthPhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg()
    default = isaacsim_physx


@configclass
class PickCubeDepthSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(),)

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
    # Existing ZED X asset, including its housing and stereo camera prims.
    zed_x = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZedX",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "depth_camera.usd")),
        # Rotate +X viewing direction down and place the left optical center
        # at (0.10, 0.0, 1.75), accounting for the asset's internal transforms.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.3, 0.3, 1.2),
            rot=(0.3582, 0.1338, -0.8291, 0.40791),
        ),
    )
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ZedX/depth_camera/base_link/ZED_X/CameraLeft",
        spawn=None,  # Use the actual ZED X camera and its authored optics.
        width=128, height=128, data_types=["rgb", "distance_to_image_plane"],
        update_period=0.0, renderer_cfg=IsaacRtxRendererCfg(),
    )



@configclass
class ActionsCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=0.75,
        use_default_offset=True, preserve_order=True, clip={".*": (-3.14, 3.14)},
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0}, close_command_expr={"finger_joint": 0.8},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        gripper_xyz = ObsTerm(func=mdp.gripper_position_local)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3

    @configclass
    class CriticCfg(PolicyCfg):
        cube_ground_truth = ObsTerm(func=mdp.cube_position)

    @configclass
    class RgbdCfg(ObsGroup):
        image = ObsTerm(func=mdp.camera_rgbd)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = 0  # Channels; preserve the image's C,H,W shape.
            self.history_length = 0

    policy = PolicyCfg()
    rgbd = RgbdCfg()
    critic = CriticCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            # Starts small; CurriculumCfg expands toward full CUBE_SPAWN_HALF_XY.
            "pose_range": mdp.initial_spawn_pose_range(),
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    reaching = RewTerm(func=mdp.reach_cube, weight=1.0)
    grasping = RewTerm(func=mdp.grasp_gate, weight=0.5)
    lifting = RewTerm(func=mdp.lift_cube, weight=8.0)
    success = RewTerm(func=mdp.success, weight=1500.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    dropping = RewTerm(func=mdp.cube_dropped, weight=-5.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    success = DoneTerm(func=mdp.success)


@configclass
class CurriculumCfg:
    """Grow cube spawn XY/yaw with training so early learning stays easy."""

    spawn_pose_range = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.reset_cube.params.pose_range",
            "modify_fn": mdp.expand_cube_spawn_range,
        },
    )


@configclass
class PickCubeDepthEnvCfg(ManagerBasedRLEnvCfg):
    scene = PickCubeDepthSceneCfg(num_envs=16, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        if not RANDOMIZE_CUBE_POSE:
            self.events.reset_cube = None
            self.curriculum.spawn_pose_range = None
        self.rerender_on_reset = True
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PickCubeDepthPhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, 2.0, 2.0), lookat=(0.0, 0.0, 0.9))