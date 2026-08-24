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
DOFBOT_V2_USD_PATH = PROJECT_ROOT / "assets" / "dofbot_v2" / "dofbot.usd"


def make_dofbot_v2_spawn_cfg() -> sim_utils.UsdFileCfg:
    """Create the Isaac Sim 6 spawn configuration for the checked-in DOFBOT_V2 asset."""
    if not DOFBOT_V2_USD_PATH.is_file():
        raise FileNotFoundError(f"DOFBOT_V2 asset not found: '{DOFBOT_V2_USD_PATH}'.")
    return sim_utils.UsdFileCfg(
        usd_path=str(DOFBOT_V2_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.001,
            stabilization_threshold=0.0005,
            disable_gravity=True,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    )


@configclass
class DofbotPickPlaceEnvCfg(DirectRLEnvCfg):
    """Direct-RL Pick–Place task using the physical DOFBOT_V2 gripper."""

    episode_length_s: float = 8.0
    decimation: int = 2
    # joint1, joint2, joint3, joint4, wrist joint deltas + right-finger command.
    # The left finger is a PhysX mimic joint and must never receive a target.
    action_space: int = 6
    # Existing state (39) + grasp axis (3) + phase one-hot (5)
    # + object up-axis (3) + object angular velocity (3).
    observation_space: int = 53
    state_space: int = 0
    clip_observations: float = 5.0
    clip_actions: float = 1.0

    curriculum_stage: str = "pick_place"
    # joint2 moves conservatively while joint3/joint4 provide the counter-bend.
    joint_action_scales: tuple[float, ...] = (0.045, 0.012, 0.055, 0.065, 0.060)
    gripper_action_smoothing: float = 0.50
    success_hold_steps: int = 8

    arm_joint_names: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4")
    wrist_joint_name: str = "Wrist_Twist_RevoluteJoint"
    finger_observation_joint_names: tuple[str, ...] = (
        "Finger_Left_01_RevoluteJoint",
        "Finger_Right_01_RevoluteJoint",
    )
    gripper_driver_joint_name: str = "Finger_Right_01_RevoluteJoint"
    gripper_mimic_joint_name: str = "Finger_Left_01_RevoluteJoint"
    grasp_reference_body_name: str = "Finger_Right_02"
    grasp_point_offset: tuple[float, float, float] = (0.00004, 0.020316, 0.019236)
    base_body_name: str = "base_link"
    ee_body_name: str = "Wrist_Twist"
    fingertip_body_names: tuple[str, ...] = ("Finger_Left_03", "Finger_Right_03")
    # The physical operating range is identical for joint1 through joint5 (wrist).
    # Floor safety is enforced by shaping and clearance checks, not reduced limits.
    arm_lower_limits_deg: tuple[float, ...] = (-90.0, -90.0, -90.0, -90.0)
    arm_upper_limits_deg: tuple[float, ...] = (90.0, 90.0, 90.0, 90.0)
    # A non-singular, forward (+Y) pointing pose. joint1 is overwritten at reset
    # with the exact sampled cube heading relative to base_link.
    initial_arm_positions_rad: tuple[float, ...] = (0.0, -0.85, -1.10, -0.40)
    initial_joint_noise_rad: float = 0.02
    preferred_joint2_rad: float = -0.80
    preferred_joint3_rad: float = -1.10
    preferred_joint4_rad: float = -0.40
    wrist_lower_limit_deg: float = -90.0
    wrist_upper_limit_deg: float = 90.0
    # The asset's authored zero pose is constraint-consistent and already leaves
    # enough clearance for the 25 mm training cube.  Resetting only the two base
    # finger joints to +/-0.7 twists the closed-loop linkage and destabilizes PhysX.
    gripper_driver_open_target: float = 0.0
    gripper_driver_closed_target: float = 0.55
    # Phase 1 may finish only after the simulated driver has physically moved
    # away from its open pose.  A command alone is not evidence of a grasp.
    gripper_driver_grasp_threshold: float = 0.05
    # Used only to initialize the PhysX mimic pair in a constraint-consistent pose.
    gripper_mimic_open_position: float = 0.0

    table_size: tuple[float, float, float] = (0.46, 0.42, 0.03)
    table_center: tuple[float, float, float] = (0.0, 0.19, 0.015)
    table_top_z: float = 0.03
    object_size: float = 0.025
    object_mass: float = 0.025
    object_x_range: tuple[float, float] = (-0.075, 0.075)
    object_y_range: tuple[float, float] = (0.10, 0.17)
    goal_x_range: tuple[float, float] = (-0.085, 0.085)
    goal_y_range: tuple[float, float] = (0.22, 0.30)
    min_object_goal_distance: float = 0.10
    lift_height: float = 0.065
    reach_tolerance: float = 0.025
    place_tolerance: float = 0.035
    pregrasp_height: float = 0.080
    transport_clearance: float = 0.080
    pregrasp_tolerance: float = 0.050
    grasp_tolerance: float = 0.030
    grasp_dwell_tolerance: float = 0.040
    direct_grasp_entry_tolerance: float = 0.090
    transport_tolerance: float = 0.050
    vertical_alignment_threshold: float = 0.70

    reach_reward_scale: float = 4.0
    reach_progress_scale: float = 25.0
    near_object_bonus: float = 2.0
    close_near_object_scale: float = 4.0
    close_far_penalty_scale: float = 1.0
    grasp_phase_bonus_scale: float = 10.0
    lift_reward_scale: float = 8.0
    lift_progress_scale: float = 80.0
    transport_reward_scale: float = 6.0
    transport_progress_scale: float = 40.0
    place_reward_scale: float = 10.0
    release_reward_scale: float = 3.0
    success_bonus: float = 100.0
    action_penalty_scale: float = 0.002
    joint_velocity_penalty_scale: float = 0.0002
    posture_penalty_scale: float = 0.08
    vertical_alignment_reward_scale: float = 1.5
    phase_progress_reward_scale: float = 2.0
    action_rate_penalty_scale: float = 0.008
    carry_action_rate_penalty_scale: float = 0.020
    carry_angular_velocity_penalty_scale: float = 0.020
    carry_tilt_penalty_scale: float = 1.0
    premature_release_penalty_scale: float = 0.5
    joint2_posture_weight: float = 4.0
    joint3_posture_weight: float = 1.5
    joint2_action_penalty_scale: float = 0.030
    # Minimum calibrated grasp-point clearance above the table in phases 0..4.
    phase_min_grasp_clearance: tuple[float, ...] = (0.040, 0.008, 0.012, 0.012, 0.008)
    floor_sweep_penalty_scale: float = 20.0
    floor_collision_clearance: float = -0.002

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Dofbot",
        spawn=make_dofbot_v2_spawn_cfg(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            # Isaac Lab 3 uses xyzw quaternions. This is identity: base_link faces +Z.
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit_sim=20.0,
                velocity_limit_sim=5.0,
                stiffness=1000.0,
                damping=50.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["Wrist_Twist_RevoluteJoint"],
                effort_limit_sim=5.0,
                velocity_limit_sim=5.0,
                stiffness=500.0,
                damping=20.0,
            ),
            "gripper_driver": ImplicitActuatorCfg(
                joint_names_expr=["Finger_Right_01_RevoluteJoint"],
                effort_limit_sim=5.0,
                velocity_limit_sim=5.0,
                # The authored 6000/1000 drive is excessive. This lower drive can
                # overcome the closed-loop linkage while the zero-pose reset keeps
                # the mimic constraints consistent in batched rollouts.
                stiffness=1000.0,
                damping=10.0,
            ),
        },
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(object_size, object_size, object_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=object_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.12, 0.08),
                metallic=0.0,
                roughness=0.45,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.13, table_top_z + 0.5 * object_size + 0.002),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    goal_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.022,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.85, 0.20),
                metallic=0.0,
                roughness=0.35,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.25, table_top_z + 0.5 * object_size),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    table_cfg: sim_utils.CuboidCfg = sim_utils.CuboidCfg(
        size=table_size,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.3,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.28, 0.30, 0.34),
            metallic=0.0,
            roughness=0.7,
        ),
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        use_fabric=True,
        physics=PhysxCfg(
            bounce_threshold_velocity=0.1,
            friction_offset_threshold=0.02,
            friction_correlation_distance=0.015,
            enable_stabilization=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=0.8,
        replicate_physics=True,
    )


@configclass
class DofbotPickPlaceReachEnvCfg(DofbotPickPlaceEnvCfg):
    curriculum_stage: str = "reach"
    episode_length_s: float = 4.0


@configclass
class DofbotPickPlaceLiftEnvCfg(DofbotPickPlaceEnvCfg):
    curriculum_stage: str = "lift"
    episode_length_s: float = 6.0
