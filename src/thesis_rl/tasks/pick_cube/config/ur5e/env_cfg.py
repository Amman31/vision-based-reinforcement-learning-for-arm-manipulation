"""UR5e pick-and-hold: grasp the cube and hold it near a fixed goal XYZ."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
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
from thesis_rl.assets.ur5e import ARM_JOINTS, UR5E_CFG
from ... import mdp

# True: randomize cube XY position and yaw at each episode reset.
# False: reset the cube to scene.cube.init_state every episode.
RANDOMIZE_CUBE_POSE = True


@configclass
class PickCubePhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg()
    default = isaacsim_physx


@configclass
class PickCubeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

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
        # Simulated camera/perception XYZ (same env frame as gripper_xyz).
        cube_xyz = ObsTerm(func=mdp.cube_position_observed, params={"noise_std": 0.005})
        goal_xyz = ObsTerm(func=mdp.goal_xyz)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3

    @configclass
    class CriticCfg(PolicyCfg):
        # Privileged: perfect simulator cube pose (training only).
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
            # Starts small; CurriculumCfg expands toward full CUBE_SPAWN_HALF_XY.
            "pose_range": mdp.initial_spawn_pose_range(),
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    reaching = RewTerm(func=mdp.reach_cube, weight=2.0)
    grasping = RewTerm(func=mdp.grasp_gate, weight=10)
    to_goal = RewTerm(func=mdp.move_to_goal, weight=50.0)
    success = RewTerm(func=mdp.success, weight=1000.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.01)
    dropping = RewTerm(func=mdp.cube_dropped, weight=-50.0)


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
class PickCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Grasp the cube and hold it near GOAL_POS (-0.04, 0.32, 1.15)."""

    scene = PickCubeSceneCfg(num_envs=16, env_spacing=2.5)
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
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PickCubePhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, 2.0, 2.0), lookat=(0.0, 0.0, 0.9))
