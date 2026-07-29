from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOFBOT_ASSET_ROOT = PROJECT_ROOT / "assets" / "dofbot"
DOFBOT_USD_PATH = DOFBOT_ASSET_ROOT / "dofbot.usd"
DOFBOT_URDF_PATH = DOFBOT_ASSET_ROOT / "dofbot_info" / "urdf" / "dofbot.urdf"


def make_dofbot_spawn_cfg():
    """Prefer the checked-in USD, with URDF import as a local fallback."""
    common_props = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            disable_gravity=True,
        ),
        "articulation_props": sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    }

    if DOFBOT_USD_PATH.is_file():
        return sim_utils.UsdFileCfg(usd_path=str(DOFBOT_USD_PATH), **common_props)
    if DOFBOT_URDF_PATH.is_file():
        return sim_utils.UrdfFileCfg(
            asset_path=str(DOFBOT_URDF_PATH),
            fix_base=True,
            link_density=1000.0,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=2048.0, damping=53.0)
            ),
            **common_props,
        )
    raise FileNotFoundError(
        f"DOFBOT asset not found. Expected either '{DOFBOT_USD_PATH}' or '{DOFBOT_URDF_PATH}'."
    )


@configclass
class DofbotReachEnvCfg(DirectRLEnvCfg):
    """Direct RL environment configuration."""

    # env
    episode_length_s: float = 5.0
    decimation: int = 2
    action_space: int = 4
    observation_space: int = 21
    state_space: int = 0
    clip_observations: float = 5.0
    clip_actions: float = 1.0

    success_tolerance: float = 0.005
    success_hold_steps: int = 3
    target_precision: float = 0.003
    control_mode: str = "joint_delta"
    action_scale: float = 0.02
    precision_action_scale: float = 0.0015
    precision_action_radius: float = 0.03
    cartesian_action_scale: float = 0.02
    precision_cartesian_action_scale: float = 0.002
    cartesian_action_radius: float = 0.03
    distance_reward_scale: float = 50.0
    progress_reward_scale: float = 300.0
    reach_reward_scale: float = 0.5
    reach_reward_std: float = 0.08
    fine_reward_scale: float = 10.0
    fine_reward_std: float = 0.02
    precision_reward_scale: float = 80.0
    precision_reward_std: float = 0.008
    ultra_precision_reward_scale: float = 120.0
    ultra_precision_reward_std: float = 0.003
    funnel_reward_scale: float = 25.0
    funnel_reward_radius: float = 0.05
    best_distance_reward_scale: float = 600.0
    close_reward_2cm: float = 5.0
    close_reward_1cm: float = 20.0
    close_reward_0_5cm: float = 120.0
    near_action_penalty_radius: float = 0.015
    near_action_penalty_scale: float = 0.008
    action_penalty_scale: float = 0.0002
    joint_velocity_penalty_scale: float = 0.001
    time_penalty: float = 0.01
    success_bonus: float = 250.0
    use_previous_action: bool = True
    training_mode: bool = True
    demo_mode: bool = False
    use_ros_target: bool = False
    ros_target_topic: str = "/dofbot/target_position"
    disable_auto_reset: bool = False

    # target sampling
    target_x_range: tuple[float, float] = (-0.1, 0.1)
    target_y_range: tuple[float, float] = (0.1, 0.2)
    target_z_range: tuple[float, float] = (0.05, 0.2)

    arm_joint_names: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4")
    joint_lower_limits_deg: tuple[float, ...] = (-180.0, -90.0, -90.0, -90.0)
    joint_upper_limits_deg: tuple[float, ...] = (180.0, 90.0, 90.0, 90.0)

    # End-effector tip offset relative to link5. The USD has /World/link5/end_point
    # as an Xform, not a rigid body, so the environment reads link5 and applies this
    # local offset to measure the actual reach point.
    ee_tip_offset: tuple[float, float, float] = (0.0008844923, -0.0049213982, 0.1076194561)
    ee_body_name: str = "link5"

    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Dofbot",
        spawn=make_dofbot_spawn_cfg(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                ".*": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3", "joint4"],
                effort_limit_sim=100.0,
                velocity_limit_sim=150.0,
                stiffness=2048.0,
                damping=53.0,
            ),
            "wrist_hold": ImplicitActuatorCfg(
                joint_names_expr=["joint5"],
                effort_limit_sim=100.0,
                velocity_limit_sim=150.0,
                stiffness=2048.0,
                damping=53.0,
            ),
        },
    )

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.005,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.2, 0.2),
                metallic=0.0,
                roughness=0.35,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.18, 0.0, 0.22),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        use_fabric=True,
        physics=PhysxCfg(
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            enable_stabilization=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.75, replicate_physics=True)


@configclass
class DofbotReachIKEnvCfg(DofbotReachEnvCfg):
    """High-precision reach variant that commands Cartesian EE deltas through differential IK."""

    action_space: int = 3
    observation_space: int = 20
    control_mode: str = "cartesian_ik"

    cartesian_action_scale: float = 0.02
    precision_cartesian_action_scale: float = 0.002
    cartesian_action_radius: float = 0.03

    distance_reward_scale: float = 30.0
    progress_reward_scale: float = 150.0
    fine_reward_scale: float = 5.0
    precision_reward_scale: float = 40.0
    ultra_precision_reward_scale: float = 80.0
    funnel_reward_scale: float = 10.0
    best_distance_reward_scale: float = 200.0
    close_reward_2cm: float = 2.0
    close_reward_1cm: float = 10.0
    close_reward_0_5cm: float = 80.0
    success_bonus: float = 100.0
