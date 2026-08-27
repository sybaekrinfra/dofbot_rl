from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
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
        activate_contact_sensors=True,
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
    # + object up-axis (3) + object angular velocity (3)
    # + normalized state-machine memory (5) + initial XY displacement (2)
    # + bilateral fingertip contact forces (2).
    observation_space: int = 62
    state_space: int = 0
    clip_observations: float = 5.0
    clip_actions: float = 1.0

    curriculum_stage: str = "pick_place"
    # All five arm joints are policy-controlled with the same per-step scale.
    joint_action_scales: tuple[float, ...] = (0.05, 0.05, 0.05, 0.05, 0.05)
    gripper_action_smoothing: float = 0.50
    # Once a settled place pose authorizes release, open the jaw over several
    # control frames instead of abruptly removing both normal forces.
    gripper_release_action_smoothing: float = 0.008
    # A terminal state must remain physically valid for one third of a second
    # at the 60 Hz control rate; a one-frame release is not task success.
    success_hold_steps: int = 20
    # Require every intermediate phase gate to remain true continuously.  The
    # entries correspond to 0->1 pre-grasp, 1->2 grasp, 2->3 lift, and 3->4
    # transport.  This prevents a single contact/velocity solver frame from
    # advancing the curriculum state machine.
    phase_transition_hold_steps: tuple[int, ...] = (4, 6, 8, 8)

    arm_joint_names: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4")
    wrist_joint_name: str = "Wrist_Twist_RevoluteJoint"
    finger_observation_joint_names: tuple[str, ...] = (
        "Finger_Left_01_RevoluteJoint",
        "Finger_Right_01_RevoluteJoint",
    )
    gripper_driver_joint_name: str = "Finger_Right_01_RevoluteJoint"
    gripper_mimic_joint_name: str = "Finger_Left_01_RevoluteJoint"
    # The user-supplied Finger_Right_02 point is the calibration source.  That
    # passive link moves during closure and its raw point is 1.82 mm off the
    # convex-pad midpoint, so using it directly makes the target move and causes
    # inevitable one-sided contact.  Express the corrected jaw center in the
    # rigid Wrist_Twist frame instead (zero-pose USD transform calibration).
    grasp_calibration_source_body_name: str = "Finger_Right_02"
    grasp_calibration_source_offset: tuple[float, float, float] = (
        0.00004,
        0.020316,
        0.019236,
    )
    grasp_reference_body_name: str = "Wrist_Twist"
    grasp_point_offset: tuple[float, float, float] = (
        -0.00164273,
        0.00027489,
        0.10364614,
    )
    base_body_name: str = "base_link"
    ee_body_name: str = "Wrist_Twist"
    fingertip_body_names: tuple[str, ...] = ("Finger_Left_03", "Finger_Right_03")
    # The physical operating range is identical for joint1 through joint5 (wrist).
    # Floor safety is enforced by shaping and clearance checks, not reduced limits.
    arm_lower_limits_deg: tuple[float, ...] = (-90.0, -90.0, -90.0, -90.0)
    arm_upper_limits_deg: tuple[float, ...] = (90.0, 90.0, 90.0, 90.0)
    # Every controlled DOFBOT joint starts at its authored zero pose.  In action
    # order this is joint1..joint4, wrist, right-finger = (0, 0, 0, 0, 0, 0).
    initial_arm_positions_rad: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    initial_joint_noise_rad: float = 0.0
    wrist_lower_limit_deg: float = -90.0
    wrist_upper_limit_deg: float = 90.0
    # Wrist twist does not change the vertical approach-axis metric, so a
    # separate hard gate is required to prevent the old +/-90 degree side grasp.
    wrist_zero_tolerance_deg: float = 8.0
    # The asset's authored zero pose is constraint-consistent and already leaves
    # enough clearance for the 25 mm training cube.  Resetting only the two base
    # finger joints to +/-0.7 twists the closed-loop linkage and destabilizes PhysX.
    # Reset remains exactly zero, then the phase-0 command moves the physical
    # right-finger driver negative to open.  +/-0.50 rad stays comfortably
    # inside the authored -57..+33 degree hard stops.
    initial_gripper_driver_position: float = 0.0
    gripper_driver_open_target: float = -0.50
    # Finger_Right_01_RevoluteJoint has exactly 90 degrees of travel: -57..+33.
    gripper_driver_lower_limit_deg: float = -57.0
    gripper_driver_upper_limit_deg: float = 33.0
    # Stay safely inside the +33 degree hard stop to avoid solver overshoot/chatter.
    gripper_driver_closed_target: float = 0.50
    # The USD-authored RIGHT driver range is -57..+33 degrees; the LEFT mimic is
    # -33..+57. Negative opens and positive closes the commanded right finger.
    # Use an absolute closure measurement.  This value is revalidated against
    # bilateral contact after correcting the fixed jaw-center TCP above.
    gripper_driver_grasp_min_position: float = 0.20
    gripper_open_gap: float = 0.0603
    # A grasp is valid only while the cube is geometrically captured between the
    # two physical fingertip bodies; finger travel by itself is not sufficient.
    # The unloaded fingertip-body center gap is about 32.8 mm when fully closed.
    # A 25 mm cube between the fingers therefore produces a larger center gap.
    gripper_grasp_min_gap: float = 0.0345
    gripper_grasp_max_gap: float = 0.052
    # The final 20 g / dynamic-friction 1.4 validation reports roughly
    # 0.09--0.15 N per jaw while held steadily.  Keep the threshold above
    # contact noise, but below that measured bilateral-grasp force.
    finger_contact_force_threshold: float = 0.02
    enable_finger_contact_sensors: bool = True
    # The corrected fixed grasp point denotes the cube center.  The old 32-mm
    # radius accepted a cube that was completely outside a 25-mm jaw.
    gripper_capture_tolerance: float = 0.010
    # Keep the fingers open during vertical descent, then permit closure only in
    # a small neighborhood around the calibrated grasp point.
    gripper_close_tolerance: float = 0.012
    # The lower finger geometry touches the table when the jaw midpoint is put
    # exactly at the resting cube center.  Runtime contact isolation found a
    # 4.5 mm upward offset to be the collision-free physical grasp center.
    grasp_center_height_offset: float = 0.0045
    gripper_limit_tolerance_rad: float = 0.02
    grasp_loss_grace_steps: int = 10
    # Used only to initialize the PhysX mimic pair in a constraint-consistent pose.
    gripper_mimic_open_position: float = 0.0

    table_size: tuple[float, float, float] = (0.46, 0.42, 0.03)
    table_center: tuple[float, float, float] = (0.0, 0.19, 0.015)
    table_top_z: float = 0.03
    object_size: float = 0.025
    # A solid 25 mm polymer cube is roughly 19--20 g (15.625 cm^3 volume).
    # The previous 25 g setting required near-static friction to lift and
    # amplified stick-slip impulses at the fingertips.
    object_mass: float = 0.020
    object_x_range: tuple[float, float] = (-0.050, 0.050)
    object_y_range: tuple[float, float] = (0.10, 0.13)
    # Batched FK at the actual 47-mm place TCP height measured vertical
    # alignment 0.916 at radius 0.19 m (and 0.938 within the 8-mm inner side of
    # the goal tolerance).  At radius 0.23 m it fell to 0.847, which necessarily
    # tipped a friction-held cube.  Keep goals inside the measured upright
    # workspace rather than asking PPO to solve an impossible pose.
    goal_x_range: tuple[float, float] = (-0.025, 0.025)
    goal_y_range: tuple[float, float] = (0.18, 0.19)
    min_object_goal_distance: float = 0.05
    # With joint1..joint4 constrained to +/-90 degrees, the downward-facing
    # grasp branch reaches joint4=-90 degrees at the table.  A 25 mm threshold
    # could be crossed for eight steps but settled back to 17--18 mm when held,
    # so it was a transient success rather than a stable carry.  At 15 mm the
    # cube bottom is still completely clear of the table and the held pose is
    # inside the measured sustainable workspace.
    lift_height: float = 0.015
    reach_tolerance: float = 0.025
    # Place success is a physical release on the table, not merely passing near
    # the goal marker while still carrying the cube.
    place_tolerance: float = 0.020
    # Six millimetres allowed release while the cube was still 5.1 mm above the
    # table; the ensuing drop hit 10.7 rad/s.  Two millimetres requires a true
    # near-contact placement before the jaw may open.
    place_height_tolerance: float = 0.002
    place_speed_tolerance: float = 0.030
    place_angular_speed_tolerance: float = 0.35
    # Tilt is represented as 1-cos(theta); 0.02 is about 11.5 degrees.
    place_tilt_tolerance: float = 0.020
    place_pose_hold_steps: int = 12
    # A valid release must cross into the negative (opening) half of travel.
    gripper_release_max_driver_position: float = -0.050
    gripper_release_min_gap: float = 0.057
    gripper_release_max_close_fraction: float = 0.10
    # Calibrated grasp point first moves 55 mm above the cube center (42.5 mm
    # above its top face) before entering the vertical descent phase.
    pregrasp_height: float = 0.055
    # Match transport height to the verified sustained lift rather than the
    # transient 25 mm peak, then enter place only when XY and height agree.
    transport_clearance: float = 0.015
    # Reach must first settle directly above the cube. A single loose 3-D
    # tolerance allowed diagonal approaches that knocked the cube away.
    pregrasp_xy_tolerance: float = 0.012
    pregrasp_height_tolerance: float = 0.012
    pregrasp_vertical_alignment_threshold: float = 0.70
    # XY alignment must dominate phase 0: replay showed Z within 6 mm while the
    # gripper still stopped 52--58 mm beside the cube.
    pregrasp_xy_cost_weight: float = 3.0
    pregrasp_height_cost_weight: float = 1.0
    descent_xy_tolerance: float = 0.012
    # Do not enter the lower/place phase while merely passing near the marker.
    # Loaded PhysX replay settles 14.4 mm from the marker (the unloaded FK
    # target is 6 mm inside it), so 15 mm is the tight measured achievable gate
    # and still prevents the previously observed 24.5 mm early transition.
    transport_tolerance: float = 0.015
    transport_height_tolerance: float = 0.006
    transport_vertical_alignment_threshold: float = 0.85
    carry_transition_speed_tolerance: float = 0.040
    carry_transition_angular_speed_tolerance: float = 0.60
    carry_transition_tilt_tolerance: float = 0.030
    vertical_alignment_threshold: float = 0.85

    # Pose shaping is a signed distance/attitude cost.  A positive exponential
    # reward at the pre-grasp boundary let the policy hover there indefinitely.
    reach_reward_scale: float = 1.0
    reach_progress_scale: float = 25.0
    close_near_object_scale: float = 2.0
    close_far_penalty_scale: float = 4.0
    grasp_phase_bonus_scale: float = 50.0
    pregrasp_phase_bonus_scale: float = 75.0
    grasp_hold_reward_scale: float = 2.0
    # These rewards are zero while holding still.  Progress is paid only when
    # the cube and calibrated grasp point move upward together.
    lift_reward_scale: float = 8.0
    lift_progress_scale: float = 200.0
    gripper_lift_guidance_scale: float = 5.0
    gripper_lift_progress_scale: float = 500.0
    grasp_separation_tolerance: float = 0.014
    grasp_separation_penalty_scale: float = 250.0
    transport_reward_scale: float = 2.0
    transport_progress_scale: float = 40.0
    place_reward_scale: float = 3.0
    release_reward_scale: float = 3.0
    # Paid during the physically gated terminal hold window.  The held-success
    # diagnostics, not a one-frame spatial crossing, decide stage completion.
    success_bonus: float = 250.0
    action_penalty_scale: float = 0.002
    joint_velocity_penalty_scale: float = 0.0002
    # The FK-verified vertical grasp branch needs joint3/joint4 near -89 deg.
    # Penalizing it from 75 deg made the pre-grasp target effectively
    # unreachable even though the asset's physical range is +/-90 deg.
    joint_soft_limit_deg: float = 89.0
    joint_soft_limit_penalty_scale: float = 10.0
    wrist_zero_penalty_scale: float = 2.0
    vertical_alignment_reward_scale: float = 0.5
    # Progress, phase completion and terminal success must dominate hovering.
    phase_progress_reward_scale: float = 50.0
    action_rate_penalty_scale: float = 0.008
    carry_action_rate_penalty_scale: float = 0.020
    carry_angular_velocity_penalty_scale: float = 0.020
    carry_tilt_penalty_scale: float = 1.0
    premature_release_penalty_scale: float = 0.5
    # Minimum calibrated grasp-point clearance above the table in phases 0..4.
    phase_min_grasp_clearance: tuple[float, ...] = (0.055, 0.008, 0.012, 0.012, 0.008)
    floor_sweep_penalty_scale: float = 20.0
    floor_collision_clearance: float = -0.002
    descent_corridor_penalty_scale: float = 120.0
    object_disturbance_penalty_scale: float = 200.0
    object_disturbance_deadband: float = 0.003
    object_disturbance_failure_distance: float = 0.020
    terminal_failure_penalty: float = 500.0

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
                # Contact during closure saturated the previous 5 Nm limit and
                # physically twisted the wrist away from its commanded 0 rad.
                # Match the arm drive authority so the policy's wrist target is
                # actually enforceable under a bilateral grasp.
                effort_limit_sim=20.0,
                velocity_limit_sim=5.0,
                stiffness=1000.0,
                damping=50.0,
            ),
            "gripper_driver": ImplicitActuatorCfg(
                joint_names_expr=["Finger_Right_01_RevoluteJoint"],
                effort_limit_sim=5.0,
                velocity_limit_sim=5.0,
                # Keep the finger near its commanded pose while the arm moves;
                # the previous weak drive let it swing from -21 to +8 degrees.
                stiffness=3000.0,
                damping=200.0,
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
                # With ~0.10 N normal force on each pad, 1.2 provides only
                # ~0.24 N dynamic-friction capacity, below the old 25 g cube's
                # 0.245 N weight.  With the physical 20 g cube, 1.4 provides
                # a reliable gravity margin.  Runtime A/B testing showed that
                # 1.3 increased separation/rotation while 1.5 increased the
                # peak stick-slip impulse, so 1.4 is the measured optimum.
                dynamic_friction=1.4,
                # Preserve the explicitly authored object friction when the
                # cube contacts finger meshes that use a lower/default material.
                friction_combine_mode="max",
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

    left_finger_contact_cfg: ContactSensorCfg = ContactSensorCfg(
        # The asset's rigid body and collision mesh share the Finger_03 name.
        # PhysX body filtering therefore resolves two entries per environment
        # and fails for batched scenes in Isaac Lab 3.0.0-beta2.  Use the
        # unfiltered fingertip force and project it onto the jaw axis in the env.
        prim_path="/World/envs/env_.*/Dofbot/dofbot/link5/Finger_Left_03",
        update_period=0.0,
        history_length=1,
    )
    right_finger_contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Dofbot/dofbot/link5/Finger_Right_03",
        update_period=0.0,
        history_length=1,
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
            # The installed PhysX backend explicitly recommends these settings
            # for stiff articulated grippers.  Runtime A/B at the original
            # 1.4 cube friction reduced peak grasp/lift angular velocity from
            # 2.51 to 0.68 rad/s without increasing jaw pressure.
            solve_articulation_contact_last=True,
            enable_external_forces_every_iteration=True,
            # At 120 Hz the extra stabilization pass is unnecessary and the
            # Isaac Lab API warns that it can corrupt contact-sensor forces.
            enable_stabilization=False,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=0.8,
        replicate_physics=True,
    )


@configclass
class DofbotPickPlaceReachEnvCfg(DofbotPickPlaceEnvCfg):
    curriculum_stage: str = "reach"
    episode_length_s: float = 4.0
    # Reach teaches the complete grasp prerequisite, but no lifting yet.
    grasp_hold_reward_scale: float = 5.0


@configclass
class DofbotPickPlaceLiftEnvCfg(DofbotPickPlaceEnvCfg):
    curriculum_stage: str = "lift"
    episode_length_s: float = 6.0
