from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, sample_uniform

from .dofbot_pick_place_cfg import DofbotPickPlaceEnvCfg


class DofbotPickPlaceEnv(DirectRLEnv):
    """Curriculum-friendly physical Pick–Place environment for DOFBOT_V2."""

    cfg: DofbotPickPlaceEnvCfg

    def __init__(self, cfg: DofbotPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.curriculum_stage not in ("reach", "lift", "pick_place"):
            raise ValueError(f"Unsupported curriculum stage: {self.cfg.curriculum_stage}")

        self.robot: Articulation = self.scene["robot"]
        self.object: RigidObject = self.scene["object"]
        self.goal: RigidObject = self.scene["goal"]

        self._arm_joint_ids = self._find_joint_ids(self.cfg.arm_joint_names)
        self._wrist_joint_id = self._find_joint_ids((self.cfg.wrist_joint_name,))[0]
        self._controlled_joint_ids = self._arm_joint_ids + [self._wrist_joint_id]
        self._finger_observation_joint_ids = self._find_joint_ids(
            self.cfg.finger_observation_joint_names
        )
        self._gripper_driver_joint_id = self._find_joint_ids(
            (self.cfg.gripper_driver_joint_name,)
        )[0]
        self._gripper_mimic_joint_id = self._find_joint_ids(
            (self.cfg.gripper_mimic_joint_name,)
        )[0]
        self._gripper_driver_lower = torch.deg2rad(
            torch.tensor(self.cfg.gripper_driver_lower_limit_deg, device=self.device)
        )
        self._gripper_driver_upper = torch.deg2rad(
            torch.tensor(self.cfg.gripper_driver_upper_limit_deg, device=self.device)
        )
        if not (
            self._gripper_driver_lower
            <= self.cfg.gripper_driver_open_target
            <= self._gripper_driver_upper
        ):
            raise ValueError("gripper_driver_open_target is outside the -57..+33 degree range")
        if not (
            self._gripper_driver_lower
            <= self.cfg.gripper_driver_closed_target
            <= self._gripper_driver_upper + 1.0e-6
        ):
            raise ValueError("gripper_driver_closed_target is outside the -57..+33 degree range")
        self._grasp_body_id = self._find_body_ids((self.cfg.grasp_reference_body_name,))[0]
        self._grasp_calibration_source_body_id = self._find_body_ids(
            (self.cfg.grasp_calibration_source_body_name,)
        )[0]
        self._base_body_id = self._find_body_ids((self.cfg.base_body_name,))[0]
        self._ee_body_id = self._find_body_ids((self.cfg.ee_body_name,))[0]
        self._fingertip_body_ids = self._find_body_ids(self.cfg.fingertip_body_names)

        self._controlled_lower = torch.deg2rad(
            torch.tensor(
                (*self.cfg.arm_lower_limits_deg, self.cfg.wrist_lower_limit_deg),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self._controlled_upper = torch.deg2rad(
            torch.tensor(
                (*self.cfg.arm_upper_limits_deg, self.cfg.wrist_upper_limit_deg),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self._wrist_zero_tolerance = torch.deg2rad(
            torch.tensor(
                self.cfg.wrist_zero_tolerance_deg,
                dtype=torch.float32,
                device=self.device,
            )
        )
        if not 0.0 < self.cfg.wrist_zero_tolerance_deg < 90.0:
            raise ValueError("wrist_zero_tolerance_deg must be between 0 and 90 degrees")
        self._joint_action_scales = torch.tensor(
            self.cfg.joint_action_scales, dtype=torch.float32, device=self.device
        )
        if len(self._joint_action_scales) != len(self._controlled_joint_ids):
            raise ValueError("joint_action_scales must have one value for each of the five driven joints")

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self.actions)
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._gripper_command = torch.zeros((self.num_envs,), device=self.device)
        self._goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._object_initial_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._object_initial_up_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._object_initial_up_w[:, 2] = 1.0
        self._previous_reach_dist = torch.zeros((self.num_envs,), device=self.device)
        self._previous_goal_dist = torch.zeros((self.num_envs,), device=self.device)
        self._previous_object_height = torch.zeros((self.num_envs,), device=self.device)
        self._previous_gripper_lift_dist = torch.zeros((self.num_envs,), device=self.device)
        self._previous_gripper_height = torch.zeros((self.num_envs,), device=self.device)
        self._previous_physical_close = torch.zeros((self.num_envs,), device=self.device)
        self._previous_phase_distance = torch.zeros((self.num_envs,), device=self.device)
        self._success_hold_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._place_pose_hold_count = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._place_release_authorized = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._phase_gate_hold_count = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._transport_waypoint_index = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._transport_waypoint_steps = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._phase_transition_hold_steps = torch.tensor(
            self.cfg.phase_transition_hold_steps, dtype=torch.long, device=self.device
        )
        if self._phase_transition_hold_steps.shape != (4,) or bool(
            torch.any(self._phase_transition_hold_steps < 1)
        ):
            raise ValueError(
                "phase_transition_hold_steps must contain four positive integers"
            )
        self._task_phase = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._grasp_loss_steps = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._pregrasp_completed = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._grasp_completed = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._phase_changed = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._task_failed = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._gripper_limit_violation = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._grasp_point_offset = torch.tensor(
            self.cfg.grasp_point_offset, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._grasp_calibration_source_offset = torch.tensor(
            self.cfg.grasp_calibration_source_offset,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._phase_min_grasp_clearance = torch.tensor(
            self.cfg.phase_min_grasp_clearance, dtype=torch.float32, device=self.device
        )
        if len(self._phase_min_grasp_clearance) != 5:
            raise ValueError("phase_min_grasp_clearance must contain one value per task phase")

    def _find_joint_ids(self, joint_names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for name in joint_names:
            joint_ids, _ = self.robot.find_joints(name)
            if len(joint_ids) != 1:
                raise RuntimeError(
                    f"Expected one DOFBOT_V2 joint named '{name}', found {len(joint_ids)}. "
                    f"Available joints: {self.robot.joint_names}"
                )
            ids.append(int(joint_ids[0]))
        return ids

    def _find_body_ids(self, body_names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for name in body_names:
            body_ids, _ = self.robot.find_bodies(name)
            if len(body_ids) != 1:
                raise RuntimeError(
                    f"Expected one DOFBOT_V2 body named '{name}', found {len(body_ids)}. "
                    f"Available bodies: {self.robot.body_names}"
                )
            ids.append(int(body_ids[0]))
        return ids

    def _setup_scene(self) -> None:
        self.cfg.table_cfg.func(
            "/World/envs/env_.*/Table",
            self.cfg.table_cfg,
            translation=self.cfg.table_center,
        )
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal = RigidObject(self.cfg.goal_cfg)
        self.left_finger_contact: ContactSensor | None = None
        self.right_finger_contact: ContactSensor | None = None
        if self.cfg.enable_finger_contact_sensors:
            self.left_finger_contact = ContactSensor(self.cfg.left_finger_contact_cfg)
            self.right_finger_contact = ContactSensor(self.cfg.right_finger_contact_cfg)
            self.scene.sensors["left_finger_contact"] = self.left_finger_contact
            self.scene.sensors["right_finger_contact"] = self.right_finger_contact

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            ),
            translation=(0.0, 0.0, -0.002),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal"] = self.goal

        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.78, 0.78, 0.78))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.to(self.device), -self.cfg.clip_actions, self.cfg.clip_actions)
        release_control_allowed = self._release_control_allowed_mask()
        default_smoothing = min(max(float(self.cfg.gripper_action_smoothing), 0.0), 1.0)
        release_smoothing = min(
            max(float(self.cfg.gripper_release_action_smoothing), 0.0), 1.0
        )
        smoothing = torch.full_like(self._gripper_command, default_smoothing)
        smoothing = torch.where(
            (self._task_phase == 4) & release_control_allowed,
            torch.full_like(smoothing, release_smoothing),
            smoothing,
        )
        requested_close = 0.5 * (self.actions[:, 5] + 1.0)
        smoothed_command = (1.0 - smoothing) * self._gripper_command + smoothing * requested_close
        grasp_dist = torch.linalg.norm(
            self._grasp_target_w() - self._gripper_center_w(), dim=-1
        )
        close_allowed = (
            (self._task_phase == 1)
            & (grasp_dist <= self.cfg.gripper_close_tolerance)
        )
        auto_close_allowed = (
            (self._task_phase == 1)
            & (grasp_dist <= self.cfg.gripper_auto_close_tolerance)
        )
        # Crossing the 6 mm gate starts an ordered close sequence.  Latch it
        # for the rest of phase 1 so contact reaction cannot reopen the jaw on
        # the next frame merely because the TCP moved a fraction outward.
        ordered_close = auto_close_allowed | (
            (self._task_phase == 1) & (self._gripper_command > 0.5)
        )
        # Action -1 opens and +1 closes.  Keep the jaw open during approach and
        # descent, enable the learned close action only at the calibrated grasp
        # point, and keep positive force after a physical capture.
        self._gripper_command = torch.where(
            (self.cfg.curriculum_stage == "reach")
            | (self._task_phase == 0)
            | ((self._task_phase == 1) & ~ordered_close),
            torch.zeros_like(smoothed_command),
            torch.where(
                ordered_close
                | (self._task_phase == 2)
                | (self._task_phase == 3)
                | ((self._task_phase == 4) & ~release_control_allowed),
                torch.ones_like(smoothed_command),
                smoothed_command,
            ),
        )
        # Incremental targets are effective for the four positioning joints;
        # their accumulated command is observable through joint_target_error.
        arm_targets = self._joint_targets[:, self._arm_joint_ids]
        normal_delta = self.actions[:, :4] * self._joint_action_scales[:4]

        # Phase 1 is an ordered, safety-critical vertical descent.  A pure PPO
        # continuation inherited Reach's hold-above behavior and stayed 70--80
        # mm from the grasp point for thousands of iterations.  Follow the
        # verified negative-joint3 seed at a bounded rate and retain a small
        # policy residual for contact/model correction.
        object_from_base = self.object.data.root_pos_w.torch - self._base_pos_w()
        object_yaw = -torch.atan2(object_from_base[:, 0], object_from_base[:, 1])
        grasp_seed = torch.tensor(
            self.cfg.phase1_grasp_joint_seed_rad,
            dtype=arm_targets.dtype,
            device=self.device,
        ).unsqueeze(0).expand(self.num_envs, -1).clone()
        grasp_seed[:, 0] += object_yaw
        grasp_seed = torch.clamp(
            grasp_seed, self._controlled_lower[:4], self._controlled_upper[:4]
        )
        scripted_delta = torch.clamp(
            grasp_seed - arm_targets,
            min=-self.cfg.phase1_descent_step_rad,
            max=self.cfg.phase1_descent_step_rad,
        )
        residual_delta = (
            self.actions[:, :4] * self.cfg.phase1_residual_action_scale_rad
        )
        phase1_delta = scripted_delta + residual_delta
        # Do not hand control back in the 12 mm -> 6 mm dead band.  The old
        # split stopped the safe descent before the ordered auto-close gate,
        # allowing PPO to learn to hover indefinitely.  Keep following the
        # vertical seed until auto-close actually becomes active.
        scripted_descent_active = (self._task_phase == 1) & ~ordered_close
        arm_delta = torch.where(
            scripted_descent_active.unsqueeze(-1), phase1_delta, normal_delta
        )

        # Phase 3 follows the same collision-free waypoint chain that passes
        # the deterministic full PickPlace test.  The failed PPO run reached
        # phase 3 in 84% of samples but satisfied goal XY in only 0.092%, then
        # catastrophically forgot Lift.  Preserve the grasp and make transport
        # an ordered motion with a bounded learned residual.
        transport_waypoints = torch.tensor(
            self.cfg.phase3_transport_joint_waypoints_rad,
            dtype=arm_targets.dtype,
            device=self.device,
        )
        waypoint_count = transport_waypoints.shape[0]
        waypoint_index = torch.clamp(
            self._transport_waypoint_index, min=0, max=waypoint_count - 1
        )
        transport_target = transport_waypoints[waypoint_index].clone()
        goal_from_base = self._goal_pos_w - self._base_pos_w()
        goal_yaw = -torch.atan2(goal_from_base[:, 0], goal_from_base[:, 1])
        transport_target[:, 0] += goal_yaw
        transport_target = torch.clamp(
            transport_target, self._controlled_lower[:4], self._controlled_upper[:4]
        )
        # Use the same fixed-duration segments as the verified deterministic
        # path.  Under load joint3 retains about 0.037 rad of servo error, so a
        # measured-error gate deadlocked indefinitely at waypoint zero.
        phase3_active = (
            (self.cfg.curriculum_stage == "pick_place") & (self._task_phase == 3)
        )
        self._transport_waypoint_steps = torch.where(
            phase3_active,
            self._transport_waypoint_steps + 1,
            self._transport_waypoint_steps,
        )
        advance_waypoint = (
            phase3_active
            & (self._transport_waypoint_steps >= self.cfg.phase3_waypoint_steps)
            & (self._transport_waypoint_index < waypoint_count - 1)
        )
        self._transport_waypoint_index = torch.where(
            advance_waypoint,
            self._transport_waypoint_index + 1,
            self._transport_waypoint_index,
        )
        self._transport_waypoint_steps = torch.where(
            advance_waypoint,
            torch.zeros_like(self._transport_waypoint_steps),
            self._transport_waypoint_steps,
        )
        transport_delta = torch.clamp(
            transport_target - arm_targets,
            min=-self.cfg.phase3_transport_step_rad,
            max=self.cfg.phase3_transport_step_rad,
        )
        transport_delta += (
            self.actions[:, :4] * self.cfg.phase3_residual_action_scale_rad
        )
        arm_delta = torch.where(phase3_active.unsqueeze(-1), transport_delta, arm_delta)

        # Phase 4 is a slow vertical place motion.  Hold this physical pose
        # until the existing position/velocity hold gate authorizes release.
        lower_target = torch.tensor(
            self.cfg.phase4_lower_joint_seed_rad,
            dtype=arm_targets.dtype,
            device=self.device,
        ).unsqueeze(0).expand(self.num_envs, -1).clone()
        lower_target[:, 0] += goal_yaw
        lower_target = torch.clamp(
            lower_target, self._controlled_lower[:4], self._controlled_upper[:4]
        )
        lower_delta = torch.clamp(
            lower_target - arm_targets,
            min=-self.cfg.phase4_lower_step_rad,
            max=self.cfg.phase4_lower_step_rad,
        )
        lower_delta += self.actions[:, :4] * self.cfg.phase4_residual_action_scale_rad
        phase4_active = (
            (self.cfg.curriculum_stage == "pick_place") & (self._task_phase == 4)
        )
        arm_delta = torch.where(phase4_active.unsqueeze(-1), lower_delta, arm_delta)
        arm_targets = arm_targets + arm_delta
        self._joint_targets[:, self._arm_joint_ids] = torch.clamp(
            arm_targets, self._controlled_lower[:4], self._controlled_upper[:4]
        )
        # Every valid Reach/Grasp/Lift/Place gate requires the twist wrist at
        # zero.  Letting PPO noise integrate this unused DOF produced a random
        # walk and made wrist-zero the sole bottleneck.  Enforce the physical
        # grasp safety posture instead of asking exploration to rediscover it.
        self._joint_targets[:, self._wrist_joint_id] = 0.0

    def _apply_action(self) -> None:
        controlled_targets = self._joint_targets[:, self._controlled_joint_ids]
        gripper_target = self.cfg.gripper_driver_open_target + self._gripper_command * (
            self.cfg.gripper_driver_closed_target - self.cfg.gripper_driver_open_target
        )
        gripper_target = torch.clamp(
            gripper_target, self._gripper_driver_lower, self._gripper_driver_upper
        )
        self._joint_targets[:, self._gripper_driver_joint_id] = gripper_target

        self.robot.set_joint_position_target_index(
            target=controlled_targets,
            joint_ids=self._controlled_joint_ids,
        )
        self.robot.set_joint_position_target_index(
            target=gripper_target.unsqueeze(-1),
            joint_ids=[self._gripper_driver_joint_id],
        )

    def _gripper_center_w(self) -> torch.Tensor:
        """Return the fixed jaw-midplane grasp point in the Wrist_Twist frame."""
        body_pos = self.robot.data.body_pos_w.torch[:, self._grasp_body_id]
        body_quat = self.robot.data.body_quat_w.torch[:, self._grasp_body_id]
        offset = self._grasp_point_offset.expand(self.num_envs, -1)
        return body_pos + quat_apply(body_quat, offset)

    def _grasp_target_w(self) -> torch.Tensor:
        """Return the collision-free cube grasp target in world coordinates."""
        target = self.object.data.root_pos_w.torch.clone()
        target[:, 2] += self.cfg.grasp_center_height_offset
        return target

    def _fingertip_gap(self) -> torch.Tensor:
        """Return the measured distance between the two physical fingertip bodies."""
        fingertip_pos = self.robot.data.body_pos_w.torch[:, self._fingertip_body_ids]
        return torch.linalg.norm(fingertip_pos[:, 0] - fingertip_pos[:, 1], dim=-1)

    def _object_between_fingertips(self) -> torch.Tensor:
        """Validate a materially closed gripper around the calibrated cube point."""
        grasp_distance = torch.linalg.norm(
            self._grasp_target_w() - self._gripper_center_w(), dim=-1
        )
        driver_position = self.robot.data.joint_pos[:, self._gripper_driver_joint_id]
        left_contact_force, right_contact_force = self._finger_contact_forces()
        return (
            (driver_position >= self.cfg.gripper_driver_grasp_min_position)
            & (self._fingertip_gap() >= self.cfg.gripper_grasp_min_gap)
            & (self._fingertip_gap() <= self.cfg.gripper_grasp_max_gap)
            & (grasp_distance <= self.cfg.gripper_capture_tolerance)
            & (left_contact_force >= self.cfg.finger_contact_force_threshold)
            & (right_contact_force >= self.cfg.finger_contact_force_threshold)
        )

    def _finger_contact_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return opposing jaw-axis contact force for both fingertips.

        DOFBOT_V2 gives each Finger_03 rigid body and its collision mesh the
        same leaf name.  PhysX's body filter expands both paths and is not
        batch-safe in Isaac Lab 3.0.0-beta2.  Projecting each unfiltered net
        normal force onto the outward jaw direction rejects table (vertical)
        contact while retaining the bilateral cube squeeze.  Distance, driver,
        and measured-gap gates in :meth:`_object_between_fingertips` provide the
        remaining object-specific geometry checks.
        """

        if self.left_finger_contact is None or self.right_finger_contact is None:
            zeros = torch.zeros((self.num_envs,), device=self.device)
            return zeros, zeros

        left_forces = self.left_finger_contact.data.net_forces_w
        right_forces = self.right_finger_contact.data.net_forces_w
        if left_forces is None or right_forces is None:
            zeros = torch.zeros((self.num_envs,), device=self.device)
            return zeros, zeros

        left_force_w = left_forces.torch.reshape(self.num_envs, -1, 3).sum(dim=1)
        right_force_w = right_forces.torch.reshape(self.num_envs, -1, 3).sum(dim=1)
        fingertip_pos = self.robot.data.body_pos_w.torch[:, self._fingertip_body_ids]
        jaw_axis = torch.nn.functional.normalize(
            fingertip_pos[:, 1] - fingertip_pos[:, 0], dim=-1
        )
        left_outward = torch.clamp(-torch.sum(left_force_w * jaw_axis, dim=-1), min=0.0)
        right_outward = torch.clamp(torch.sum(right_force_w * jaw_axis, dim=-1), min=0.0)
        return left_outward, right_outward

    def _gripper_physical_close_fraction(self) -> torch.Tensor:
        """Measure absolute positive closure, independent of a moving start pose."""
        driver_position = self.robot.data.joint_pos[:, self._gripper_driver_joint_id]
        driver_fraction = torch.clamp(
            driver_position / self.cfg.gripper_driver_grasp_min_position,
            0.0,
            1.0,
        )
        required_gap_reduction = self.cfg.gripper_open_gap - self.cfg.gripper_grasp_max_gap
        gap_fraction = torch.clamp(
            (self.cfg.gripper_open_gap - self._fingertip_gap()) / required_gap_reduction,
            0.0,
            1.0,
        )
        return torch.minimum(driver_fraction, gap_fraction)

    def _grasp_approach_axis_w(self) -> torch.Tensor:
        """Return the fixed Wrist_Twist grasp frame's local +Z in world coordinates."""
        body_quat = self.robot.data.body_quat_w.torch[:, self._grasp_body_id]
        local_z = torch.zeros((self.num_envs, 3), device=self.device)
        local_z[:, 2] = 1.0
        return quat_apply(body_quat, local_z)

    def _vertical_alignment(self) -> torch.Tensor:
        # Sign matters: only the downward approach is a valid grasp. abs() let
        # the wrist point at the ceiling and still pass every alignment gate,
        # which also flips the sign of the calibrated offset in
        # _gripper_center_w() and stalls pregrasp height ~2x that offset.
        return -self._grasp_approach_axis_w()[:, 2]

    def _wrist_zero_mask(self) -> torch.Tensor:
        """Return wrists inside the explicit zero-twist grasp tolerance."""
        return (
            torch.abs(self.robot.data.joint_pos[:, self._wrist_joint_id])
            <= self._wrist_zero_tolerance
        )

    def _base_pos_w(self) -> torch.Tensor:
        """Return the physical base_link origin used as Pick–Place coordinate zero."""
        return self.robot.data.body_pos_w.torch[:, self._base_body_id]

    def _task_state(self) -> tuple[torch.Tensor, ...]:
        gripper_pos = self._gripper_center_w()
        object_pos = self.object.data.root_pos_w.torch
        goal_pos = self._goal_pos_w
        reach_dist = torch.linalg.norm(self._grasp_target_w() - gripper_pos, dim=-1)
        goal_dist = torch.linalg.norm(goal_pos - object_pos, dim=-1)
        object_height = object_pos[:, 2] - (self.cfg.table_top_z + 0.5 * self.cfg.object_size)
        object_speed = torch.linalg.norm(self.object.data.root_lin_vel_w.torch, dim=-1)
        return gripper_pos, object_pos, goal_pos, reach_dist, goal_dist, object_height, object_speed

    def _active_task_target_error_w(
        self,
        gripper_pos: torch.Tensor,
        object_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Return the error to the target that owns the current task phase.

        The previous observation always exposed the final grasp-center error.
        During phase 0 that contradicted the reward and transition gate, whose
        true target is 55 mm higher.  This explicit phase target keeps the MDP
        observation consistent with the deterministic manipulation sequence.
        """

        grasp_target = self._grasp_target_w()
        pregrasp_target = grasp_target.clone()
        pregrasp_target[:, 2] += self.cfg.pregrasp_height
        gripper_lift_target = object_pos.clone()
        gripper_lift_target[:, 2] = (
            self._object_initial_pos_w[:, 2]
            + self.cfg.grasp_center_height_offset
            + self.cfg.lift_height
        )
        goal_above_target = goal_pos.clone()
        goal_above_target[:, 2] += self.cfg.transport_clearance

        phase = self._task_phase.unsqueeze(-1)
        return torch.where(
            (phase == 0) | (self.cfg.curriculum_stage == "reach"),
            pregrasp_target - gripper_pos,
            torch.where(
                phase == 1,
                grasp_target - gripper_pos,
                torch.where(
                    phase == 2,
                    gripper_lift_target - gripper_pos,
                    torch.where(
                        phase == 3,
                        goal_above_target - object_pos,
                        goal_pos - object_pos,
                    ),
                ),
            ),
        )

    def _place_spatial_mask(self) -> torch.Tensor:
        """Return cubes inside the final goal volume, independent of velocity."""
        _, object_pos, goal_pos, _, goal_dist, _, _ = self._task_state()
        return (
            (self._task_phase == 4)
            & (goal_dist < self.cfg.place_tolerance)
            & (
                torch.abs(object_pos[:, 2] - goal_pos[:, 2])
                < self.cfg.place_height_tolerance
            )
        )

    def _place_pose_mask(self) -> torch.Tensor:
        """Return environments whose cube is settled at the physical place pose."""
        object_speed = torch.linalg.norm(self.object.data.root_lin_vel_w.torch, dim=-1)
        object_angular_speed = torch.linalg.norm(
            self.object.data.root_ang_vel_w.torch, dim=-1
        )
        return (
            self._place_spatial_mask()
            & (object_speed < self.cfg.place_speed_tolerance)
            & (object_angular_speed < self.cfg.place_angular_speed_tolerance)
            & (self._object_tilt() < self.cfg.place_tilt_tolerance)
            & self._wrist_zero_mask()
        )

    def _release_control_allowed_mask(self) -> torch.Tensor:
        """Keep release authority only at the goal with a zero-twist wrist.

        The settled-pose authorization remains latched through the small
        velocity transient caused by opening, but it cannot be used after the
        cube leaves the goal volume or the wrist rotates away from zero.
        """
        return (
            self._place_release_authorized
            & self._place_spatial_mask()
            & self._wrist_zero_mask()
        )

    def _object_tilt(self) -> torch.Tensor:
        """Return 1-cos(theta) from each cube's reset up-axis."""
        object_local_up = torch.zeros((self.num_envs, 3), device=self.device)
        object_local_up[:, 2] = 1.0
        object_up_w = quat_apply(self.object.data.root_quat_w.torch, object_local_up)
        return 1.0 - torch.clamp(
            torch.sum(object_up_w * self._object_initial_up_w, dim=-1), -1.0, 1.0
        )

    def _gripper_released_mask(self) -> torch.Tensor:
        """Use measured joint/gap state, not only the policy command, for release."""
        driver_position = self.robot.data.joint_pos[:, self._gripper_driver_joint_id]
        return (
            (driver_position <= self.cfg.gripper_release_max_driver_position)
            & (self._fingertip_gap() >= self.cfg.gripper_release_min_gap)
            & (
                self._gripper_physical_close_fraction()
                <= self.cfg.gripper_release_max_close_fraction
            )
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        gripper_pos, object_pos, goal_pos, _, _, _, _ = self._task_state()
        base_pos = self._base_pos_w()
        controlled_pos = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        controlled_vel = self.robot.data.joint_vel[:, self._controlled_joint_ids]
        # Expose servo tracking error so the policy can distinguish a settled
        # pose from one whose actuator is still chasing the bounded offset.
        joint_target_error = (
            self._joint_targets[:, self._controlled_joint_ids] - controlled_pos
        )
        finger_pos = self.robot.data.joint_pos[:, self._finger_observation_joint_ids]
        finger_vel = self.robot.data.joint_vel[:, self._finger_observation_joint_ids]
        object_vel = self.object.data.root_lin_vel_w.torch
        object_ang_vel = self.object.data.root_ang_vel_w.torch
        object_up = torch.zeros((self.num_envs, 3), device=self.device)
        object_up[:, 2] = 1.0
        object_up = quat_apply(self.object.data.root_quat_w.torch, object_up)
        approach_axis = self._grasp_approach_axis_w()
        phase_one_hot = torch.nn.functional.one_hot(self._task_phase, num_classes=5).float()
        phase_required_hold = self._phase_transition_hold_steps[
            torch.clamp(self._task_phase, min=0, max=3)
        ].float()
        phase_hold_fraction = torch.clamp(
            self._phase_gate_hold_count.float() / phase_required_hold, 0.0, 1.0
        )
        place_hold_fraction = torch.clamp(
            self._place_pose_hold_count.float() / float(self.cfg.place_pose_hold_steps),
            0.0,
            1.0,
        )
        grasp_loss_fraction = torch.clamp(
            self._grasp_loss_steps.float() / float(self.cfg.grasp_loss_grace_steps),
            0.0,
            1.0,
        )
        success_hold_fraction = torch.clamp(
            self._success_hold_count.float() / float(self.cfg.success_hold_steps),
            0.0,
            1.0,
        )
        state_machine_memory = torch.stack(
            (
                phase_hold_fraction,
                place_hold_fraction,
                self._place_release_authorized.float(),
                grasp_loss_fraction,
                success_hold_fraction,
            ),
            dim=-1,
        )
        initial_xy_displacement = (
            object_pos[:, :2] - self._object_initial_pos_w[:, :2]
        ) / self.cfg.object_disturbance_failure_distance
        left_contact_force, right_contact_force = self._finger_contact_forces()
        normalized_contact_forces = torch.stack(
            (left_contact_force, right_contact_force), dim=-1
        ) / max(self.cfg.finger_contact_force_threshold, 1.0e-6)
        active_target_error = self._active_task_target_error_w(
            gripper_pos, object_pos, goal_pos
        )

        obs = torch.cat(
            (
                controlled_pos,
                controlled_vel,
                joint_target_error,
                finger_pos,
                finger_vel,
                gripper_pos - base_pos,
                object_pos - base_pos,
                goal_pos - base_pos,
                self._grasp_target_w() - gripper_pos,
                goal_pos - object_pos,
                object_vel,
                self._gripper_command.unsqueeze(-1),
                self._previous_actions,
                active_target_error,
                approach_axis,
                phase_one_hot,
                object_up,
                object_ang_vel,
                state_machine_memory,
                initial_xy_displacement,
                normalized_contact_forces,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -self.cfg.clip_observations, self.cfg.clip_observations)}

    def _success_mask(self) -> torch.Tensor:
        _, _, _, reach_dist, goal_dist, object_height, object_speed = self._task_state()
        object_angular_speed = torch.linalg.norm(
            self.object.data.root_ang_vel_w.torch, dim=-1
        )
        stable_carry = (
            (object_speed < self.cfg.carry_transition_speed_tolerance)
            & (
                object_angular_speed
                < self.cfg.carry_transition_angular_speed_tolerance
            )
            & (self._object_tilt() < self.cfg.carry_transition_tilt_tolerance)
        )
        gripper_physically_closed = (
            (self._gripper_physical_close_fraction() >= 1.0)
            & self._object_between_fingertips()
        )
        wrist_zero_ok = self._wrist_zero_mask()
        if self.cfg.curriculum_stage == "reach":
            gripper_pos = self._gripper_center_w()
            object_pos = self.object.data.root_pos_w.torch
            pregrasp_target = self._grasp_target_w()
            pregrasp_target[:, 2] += self.cfg.pregrasp_height
            return (
                (self._task_phase >= 1)
                & (
                    torch.linalg.norm(
                        gripper_pos[:, :2] - object_pos[:, :2], dim=-1
                    )
                    < self.cfg.pregrasp_xy_tolerance
                )
                & (
                    torch.abs(gripper_pos[:, 2] - pregrasp_target[:, 2])
                    < self.cfg.pregrasp_height_tolerance
                )
                & (
                    self._vertical_alignment()
                    > self.cfg.pregrasp_vertical_alignment_threshold
                )
                & wrist_zero_ok
            )
        if self.cfg.curriculum_stage == "lift":
            return (
                (self._task_phase >= 3)
                & gripper_physically_closed
                & (object_height > self.cfg.lift_height)
                & stable_carry
                & wrist_zero_ok
            )
        return self._place_pose_mask() & self._gripper_released_mask()

    def _update_task_phase(
        self,
        pregrasp_dist: torch.Tensor,
        pregrasp_xy_dist: torch.Tensor,
        pregrasp_height_error: torch.Tensor,
        grasp_dist: torch.Tensor,
        goal_above_dist: torch.Tensor,
        goal_xy_dist: torch.Tensor,
        goal_transport_height_error: torch.Tensor,
        object_height: torch.Tensor,
        vertical_alignment: torch.Tensor,
        gripper_physically_closed: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one step in the ordered Pick–Place process and return changed envs."""
        phase = self._task_phase
        object_speed = torch.linalg.norm(self.object.data.root_lin_vel_w.torch, dim=-1)
        object_angular_speed = torch.linalg.norm(
            self.object.data.root_ang_vel_w.torch, dim=-1
        )
        object_tilt = self._object_tilt()
        stable_carry = (
            (object_speed < self.cfg.carry_transition_speed_tolerance)
            & (
                object_angular_speed
                < self.cfg.carry_transition_angular_speed_tolerance
            )
            & (object_tilt < self.cfg.carry_transition_tilt_tolerance)
        )
        pregrasp_ready = (
            (pregrasp_xy_dist < self.cfg.pregrasp_xy_tolerance)
            & (pregrasp_height_error < self.cfg.pregrasp_height_tolerance)
            & (vertical_alignment > self.cfg.pregrasp_vertical_alignment_threshold)
            & self._wrist_zero_mask()
        )
        wrist_zero_ok = self._wrist_zero_mask()
        phase_gate_ready = (phase == 0) & pregrasp_ready
        if self.cfg.curriculum_stage != "reach":
            phase_gate_ready |= (
                (phase == 1)
                & (grasp_dist < self.cfg.gripper_capture_tolerance)
                & (self._gripper_command > 0.65)
                & gripper_physically_closed
                & (vertical_alignment > self.cfg.vertical_alignment_threshold)
                & wrist_zero_ok
            )
        if self.cfg.curriculum_stage != "reach":
            phase_gate_ready |= (
                (phase == 2)
                & gripper_physically_closed
                & (object_height > self.cfg.lift_height)
                & stable_carry
                & wrist_zero_ok
            )
        if self.cfg.curriculum_stage == "pick_place":
            phase_gate_ready |= (
                (phase == 3)
                & gripper_physically_closed
                & (goal_xy_dist < self.cfg.transport_tolerance)
                & (goal_transport_height_error < self.cfg.transport_height_tolerance)
                & (
                    vertical_alignment
                    > self.cfg.transport_vertical_alignment_threshold
                )
                & stable_carry
                & wrist_zero_ok
            )

        self._phase_gate_hold_count = torch.where(
            phase_gate_ready,
            self._phase_gate_hold_count + 1,
            torch.zeros_like(self._phase_gate_hold_count),
        )
        phase_index = torch.clamp(phase, min=0, max=3)
        required_hold = self._phase_transition_hold_steps[phase_index]
        transition_ready = phase_gate_ready & (
            self._phase_gate_hold_count >= required_hold
        )
        next_phase = torch.where(transition_ready, phase + 1, phase)

        # In phase 4 the grasp may be released only after the cube is physically
        # at rest at the place pose.  Until then, loss of bilateral capture is a
        # failed carry just as it is during lift and transport.
        release_allowed = self._release_control_allowed_mask()
        losing_grasp = (
            (phase == 2)
            | (phase == 3)
            | ((phase == 4) & ~release_allowed)
        ) & ~gripper_physically_closed
        self._grasp_loss_steps = torch.where(
            losing_grasp,
            self._grasp_loss_steps + 1,
            torch.zeros_like(self._grasp_loss_steps),
        )
        # A short grace period tolerates contact-solver chatter.  Once a grasp
        # is truly lost the episode fails; returning to phase 0 would allow the
        # policy to collect approach rewards again without completing the task.
        lost_grasp = (
            ((phase == 2) | (phase == 3) | ((phase == 4) & ~release_allowed))
            & (self._grasp_loss_steps >= self.cfg.grasp_loss_grace_steps)
        )
        self._task_failed |= lost_grasp
        changed = next_phase != phase
        self._task_phase = next_phase
        self._phase_gate_hold_count = torch.where(
            changed,
            torch.zeros_like(self._phase_gate_hold_count),
            self._phase_gate_hold_count,
        )
        return changed

    def _advance_task_phase_from_sim(self) -> torch.Tensor:
        """Update the ordered task state from the latest simulated poses."""
        grasp_pos, object_pos, goal_pos, reach_dist, _, object_height, _ = self._task_state()
        pregrasp_target = self._grasp_target_w()
        pregrasp_target[:, 2] += self.cfg.pregrasp_height
        goal_above_target = goal_pos.clone()
        goal_above_target[:, 2] += self.cfg.transport_clearance
        pregrasp_xy_dist = torch.linalg.norm(
            grasp_pos[:, :2] - object_pos[:, :2], dim=-1
        )
        gripper_physically_closed = (
            (self._gripper_physical_close_fraction() >= 1.0)
            & self._object_between_fingertips()
        )
        return self._update_task_phase(
            torch.linalg.norm(grasp_pos - pregrasp_target, dim=-1),
            pregrasp_xy_dist,
            torch.abs(grasp_pos[:, 2] - pregrasp_target[:, 2]),
            reach_dist,
            torch.linalg.norm(object_pos - goal_above_target, dim=-1),
            torch.linalg.norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1),
            torch.abs(
                object_pos[:, 2]
                - goal_pos[:, 2]
                - self.cfg.transport_clearance
            ),
            object_height,
            self._vertical_alignment(),
            gripper_physically_closed,
        )

    def _get_rewards(self) -> torch.Tensor:
        grasp_pos, object_pos, goal_pos, reach_dist, goal_dist, object_height, _ = self._task_state()
        close_command = self._gripper_command
        gripper_driver_position = self.robot.data.joint_pos[:, self._gripper_driver_joint_id]
        fingertip_gap = self._fingertip_gap()
        physical_close_fraction = self._gripper_physical_close_fraction()
        object_between_fingertips = self._object_between_fingertips()
        left_contact_force, right_contact_force = self._finger_contact_forces()
        gripper_physically_closed = (
            (physical_close_fraction >= 1.0) & object_between_fingertips
        )
        vertical_alignment = self._vertical_alignment()
        pregrasp_target = self._grasp_target_w()
        pregrasp_target[:, 2] += self.cfg.pregrasp_height
        goal_above_target = goal_pos.clone()
        goal_above_target[:, 2] += self.cfg.transport_clearance
        pregrasp_dist = torch.linalg.norm(grasp_pos - pregrasp_target, dim=-1)
        pregrasp_xy_dist = torch.linalg.norm(
            grasp_pos[:, :2] - object_pos[:, :2], dim=-1
        )
        pregrasp_height_error = torch.abs(grasp_pos[:, 2] - pregrasp_target[:, 2])
        goal_above_dist = torch.linalg.norm(object_pos - goal_above_target, dim=-1)
        goal_xy_dist = torch.linalg.norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
        goal_transport_height_error = torch.abs(
            object_pos[:, 2] - goal_pos[:, 2] - self.cfg.transport_clearance
        )

        # DirectRLEnv computes dones before rewards.  _get_dones() advances the
        # phase and caches this event so both calculations observe one state.
        phase_changed = self._phase_changed
        phase = self._task_phase
        if self.cfg.curriculum_stage == "reach":
            # Phase 1 means the held pre-grasp gate was reached.  Maintain the
            # same safe target until terminal success; descent starts in Lift.
            phase0 = (phase <= 1).float()
            phase1 = torch.zeros_like(phase0)
        else:
            phase0 = (phase == 0).float()
            phase1 = (phase == 1).float()
        phase2 = (phase == 2).float()
        phase3 = (phase == 3).float()
        phase4 = (phase == 4).float()
        enable_lift = float(self.cfg.curriculum_stage != "reach")
        enable_place = float(self.cfg.curriculum_stage == "pick_place")
        # Phase 4 remains a carry only while the cube is physically retained.
        # Once released, separation from the gripper is the desired outcome and
        # must not be punished as a lost grasp.
        release_allowed = self._place_release_authorized
        carry = (
            (phase == 2)
            | (phase == 3)
            | ((phase == 4) & (~release_allowed | gripper_physically_closed))
        ).float()
        grasp_maintained = (
            ((phase == 2) | (phase == 3) | (phase == 4))
            & gripper_physically_closed
        )

        lifted_fraction = torch.clamp(object_height / self.cfg.lift_height, 0.0, 1.5)
        lifted = object_height > self.cfg.lift_height
        # Keep smooth-motion shaping active throughout place/release as well.
        # Otherwise a policy can create a large opening impulse and merely wait
        # for the cube to settle before collecting terminal success.
        stability_active = torch.clamp(
            carry * lifted.float() + enable_place * phase4,
            min=0.0,
            max=1.0,
        )

        # Signed costs cannot be harvested by hovering just outside a phase
        # boundary.  Motion toward the target is separately rewarded below by
        # phase_progress, while phase completion receives a one-shot bonus.
        approach_reward = -phase0 * self.cfg.reach_reward_scale * (
            self.cfg.pregrasp_xy_cost_weight * pregrasp_xy_dist
            + self.cfg.pregrasp_height_cost_weight * pregrasp_height_error
        )
        grasp_reward = -phase1 * self.cfg.reach_reward_scale * reach_dist
        alignment_proximity = phase0 * torch.exp(-pregrasp_dist / 0.12) + phase1 * torch.exp(
            -reach_dist / 0.08
        )
        alignment_reward = (
            -self.cfg.vertical_alignment_reward_scale
            * (1.0 - vertical_alignment)
            * alignment_proximity
        )
        close_ready = (reach_dist <= self.cfg.gripper_close_tolerance).float()
        close_progress = torch.clamp(
            physical_close_fraction - self._previous_physical_close,
            min=-0.25,
            max=0.25,
        )
        close_near = (
            phase1
            * close_ready
            * self.cfg.close_near_object_scale
            * close_progress
            * torch.exp(-reach_dist / 0.010)
        )
        # Closure is an ordered safety action once the 12-mm gate is reached,
        # so the environment performs it directly.  PPO retains finger control
        # for final release but receives no contradictory phase-1 action loss.
        close_command_reward = torch.zeros_like(reach_dist)
        close_far = torch.zeros_like(reach_dist)

        # Lift vertically from the cube's current XY instead of pulling the
        # gripper back toward the sampled XY if contact shifts the cube slightly.
        gripper_lift_target = object_pos.clone()
        gripper_lift_target[:, 2] = (
            self._object_initial_pos_w[:, 2]
            + self.cfg.grasp_center_height_offset
            + self.cfg.lift_height
        )
        gripper_lift_dist = torch.linalg.norm(grasp_pos - gripper_lift_target, dim=-1)
        gripper_height = grasp_pos[:, 2] - (
            self._object_initial_pos_w[:, 2] + self.cfg.grasp_center_height_offset
        )
        gripper_lift_fraction = torch.clamp(
            gripper_height / self.cfg.lift_height, 0.0, 1.0
        )
        lift_reward = (
            -enable_lift
            * phase2
            * self.cfg.lift_reward_scale
            * torch.relu(self.cfg.lift_height - object_height)
        )
        lift_progress = (
            enable_lift
            * phase2
            * grasp_maintained.float()
            * self.cfg.lift_progress_scale
            * torch.clamp(
                object_height - self._previous_object_height, min=-0.01, max=0.01
            )
        )
        gripper_lift_guidance = (
            -enable_lift
            * phase2
            * grasp_maintained.float()
            * self.cfg.gripper_lift_guidance_scale
            * gripper_lift_dist
        )
        gripper_lift_progress = (
            enable_lift
            * phase2
            * grasp_maintained.float()
            * self.cfg.gripper_lift_progress_scale
            * torch.clamp(
                gripper_height - self._previous_gripper_height,
                min=-0.01,
                max=0.01,
            )
        )
        gripper_lift_progress = torch.where(
            phase_changed, torch.zeros_like(gripper_lift_progress), gripper_lift_progress
        )
        grasp_hold_reward = (
            -carry
            * self.cfg.grasp_hold_reward_scale
            * torch.relu(reach_dist - self.cfg.gripper_capture_tolerance)
        )
        transport_reward = (
            -enable_place
            * phase3
            * self.cfg.transport_reward_scale
            * goal_above_dist
        )
        place_reward = (
            -enable_place * phase4 * self.cfg.place_reward_scale * goal_dist
        )
        place_pose_ready = self._place_pose_mask()
        release_allowed = self._release_control_allowed_mask()
        release_progress = torch.clamp(
            self._previous_physical_close - physical_close_fraction,
            # Signed progress prevents an open/re-close oscillation from farming
            # only the positive half of the release motion.
            min=-0.25,
            max=0.25,
        )
        release_reward = (
            enable_place
            * phase4
            * self.cfg.release_reward_scale
            * release_allowed.float()
            * place_pose_ready.float()
            * release_progress
        )
        premature_release_penalty = -self.cfg.premature_release_penalty_scale * (
            enable_lift * (phase2 + phase3) * (1.0 - physical_close_fraction)
            + enable_place
            * phase4
            * (~place_pose_ready).float()
            * (1.0 - physical_close_fraction)
        )
        grasp_separation_penalty = (
            -enable_lift
            * carry
            * self.cfg.grasp_separation_penalty_scale
            * torch.relu(reach_dist - self.cfg.grasp_separation_tolerance)
        )

        phase_distance = torch.where(
            (phase == 0) | (self.cfg.curriculum_stage == "reach"),
            pregrasp_dist,
            torch.where(
                phase == 1,
                reach_dist,
                torch.where(
                    phase == 2,
                    gripper_lift_dist,
                    torch.where(phase == 3, goal_above_dist, goal_dist),
                ),
            ),
        )
        phase_progress = self.cfg.phase_progress_reward_scale * torch.clamp(
            self._previous_phase_distance - phase_distance, min=-0.02, max=0.02
        )
        progress_active = (
            phase0
            + phase1
            + enable_lift * phase2 * grasp_maintained.float()
            + enable_place * (phase3 * grasp_maintained.float() + phase4)
        )
        phase_progress *= progress_active
        phase_progress = torch.where(phase_changed, torch.zeros_like(phase_progress), phase_progress)
        first_grasp = phase_changed & (phase == 2) & ~self._grasp_completed
        grasp_phase_bonus = self.cfg.grasp_phase_bonus_scale * first_grasp.float()
        self._grasp_completed |= first_grasp
        first_pregrasp = phase_changed & (phase == 1) & ~self._pregrasp_completed
        pregrasp_phase_bonus = self.cfg.pregrasp_phase_bonus_scale * first_pregrasp.float()
        self._pregrasp_completed |= first_pregrasp
        stage_reward = (
            approach_reward
            + grasp_reward
            + alignment_reward
            + close_near
            + close_command_reward
            + close_far
            + lift_reward
            + lift_progress
            + gripper_lift_guidance
            + gripper_lift_progress
            + grasp_hold_reward
            + transport_reward
            + place_reward
            + release_reward
            + premature_release_penalty
            + grasp_separation_penalty
            + phase_progress
            + grasp_phase_bonus
            + pregrasp_phase_bonus
        )

        success = self._success_mask()
        success_bonus = self.cfg.success_bonus * success.float()
        action_penalty = -self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)
        action_rate = torch.sum((self.actions - self._previous_actions) ** 2, dim=-1)
        action_rate_penalty = -(
            self.cfg.action_rate_penalty_scale
            + stability_active * self.cfg.carry_action_rate_penalty_scale
        ) * action_rate
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        wrist_position = self.robot.data.joint_pos[:, self._wrist_joint_id]
        joint_soft_limit = torch.deg2rad(
            torch.tensor(self.cfg.joint_soft_limit_deg, device=self.device)
        )
        controlled_pos = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        joint_soft_limit_penalty = -self.cfg.joint_soft_limit_penalty_scale * torch.sum(
            torch.relu(torch.abs(controlled_pos) - joint_soft_limit) ** 2,
            dim=-1,
        )
        # This is a soft grasp-orientation objective, not a target override:
        # Wrist_Twist remains fully controllable over -90..+90 degrees.
        wrist_zero_penalty = -self.cfg.wrist_zero_penalty_scale * wrist_position**2
        controlled_vel = self.robot.data.joint_vel[:, self._controlled_joint_ids]
        velocity_penalty = -self.cfg.joint_velocity_penalty_scale * torch.sum(
            controlled_vel**2, dim=-1
        )
        object_ang_vel = self.object.data.root_ang_vel_w.torch
        angular_stability_penalty = (
            -stability_active
            * self.cfg.carry_angular_velocity_penalty_scale
            * torch.clamp(torch.sum(object_ang_vel**2, dim=-1), max=25.0)
        )
        object_up = torch.zeros((self.num_envs, 3), device=self.device)
        object_up[:, 2] = 1.0
        object_up = quat_apply(self.object.data.root_quat_w.torch, object_up)
        # Penalize orientation change from the reset pose rather than assuming
        # a particular absolute pose.  This also remains correct if object pose
        # randomization is introduced later.
        object_tilt = 1.0 - torch.clamp(
            torch.sum(object_up * self._object_initial_up_w, dim=-1), -1.0, 1.0
        )
        tilt_penalty = -stability_active * self.cfg.carry_tilt_penalty_scale * object_tilt
        grasp_clearance = (
            grasp_pos[:, 2] - self._base_pos_w()[:, 2] - self.cfg.table_top_z
        )
        required_clearance = self._phase_min_grasp_clearance[phase]
        floor_sweep_penalty = -self.cfg.floor_sweep_penalty_scale * torch.relu(
            required_clearance - grasp_clearance
        )
        # During descent, keep the calibrated grasp point inside a narrow
        # vertical corridor over the cube. Also reject policies that obtain a
        # low reach distance by pushing the cube across the table.
        descent_corridor_penalty = (
            -self.cfg.descent_corridor_penalty_scale
            * phase1
            * torch.relu(pregrasp_xy_dist - self.cfg.descent_xy_tolerance)
        )
        object_xy_displacement = torch.linalg.norm(
            object_pos[:, :2] - self._object_initial_pos_w[:, :2], dim=-1
        )
        # Object displacement is a failure only before the first capture.  The
        # old expression also penalized the intended transport and final
        # release because both move the object away from its reset position.
        # Rewards are evaluated after _get_dones() advances the phase.  Rebuild
        # the phase that owned this physics step so a transition cannot suppress
        # the disturbance cost on the very frame that moved the cube.
        phase_before_transition = phase - phase_changed.long()
        before_capture = (phase_before_transition <= 1).float()
        object_disturbance_penalty = (
            -self.cfg.object_disturbance_penalty_scale
            * before_capture
            * torch.relu(
                object_xy_displacement - self.cfg.object_disturbance_deadband
            )
        )
        object_local = object_pos - self._base_pos_w()
        fallen_now = (object_local[:, 2] < -0.01) | (
            torch.linalg.norm(object_local[:, :2], dim=-1) > 0.45
        )
        floor_collision_now = grasp_clearance < self.cfg.floor_collision_clearance
        disturbed_now = (
            (phase_before_transition <= 1)
            & (object_xy_displacement > self.cfg.object_disturbance_failure_distance)
        )
        gripper_limit_now = (
            ~torch.isfinite(gripper_driver_position)
            | (
                gripper_driver_position
                < self._gripper_driver_lower - self.cfg.gripper_limit_tolerance_rad
            )
            | (
                gripper_driver_position
                > self._gripper_driver_upper + self.cfg.gripper_limit_tolerance_rad
            )
        )
        terminal_failure_penalty = -self.cfg.terminal_failure_penalty * (
            fallen_now
            | floor_collision_now
            | disturbed_now
            | gripper_limit_now
            | self._task_failed
        ).float()
        reward = (
            stage_reward
            + success_bonus
            + action_penalty
            + action_rate_penalty
            + velocity_penalty
            + joint_soft_limit_penalty
            + wrist_zero_penalty
            + angular_stability_penalty
            + tilt_penalty
            + floor_sweep_penalty
            + descent_corridor_penalty
            + object_disturbance_penalty
            + terminal_failure_penalty
        )

        self._previous_reach_dist = reach_dist.detach()
        self._previous_goal_dist = goal_dist.detach()
        self._previous_object_height = object_height.detach()
        self._previous_gripper_lift_dist = gripper_lift_dist.detach()
        self._previous_gripper_height = gripper_height.detach()
        self._previous_physical_close = physical_close_fraction.detach()
        self._previous_phase_distance = phase_distance.detach()
        grasp_distance_ok = reach_dist < self.cfg.gripper_capture_tolerance
        xy_ok = pregrasp_xy_dist < self.cfg.pregrasp_xy_tolerance
        height_ok = pregrasp_height_error < self.cfg.pregrasp_height_tolerance
        pregrasp_orientation_ok = (
            vertical_alignment > self.cfg.pregrasp_vertical_alignment_threshold
        )
        wrist_zero_ok = self._wrist_zero_mask()
        pregrasp_ready = xy_ok & height_ok & pregrasp_orientation_ok & wrist_zero_ok
        close_allowed = reach_dist <= self.cfg.gripper_close_tolerance
        close_commanded = close_command > 0.65
        driver_close_ok = (
            gripper_driver_position >= self.cfg.gripper_driver_grasp_min_position
        )
        gap_lower_ok = fingertip_gap >= self.cfg.gripper_grasp_min_gap
        gap_upper_ok = fingertip_gap <= self.cfg.gripper_grasp_max_gap
        left_contact = left_contact_force >= self.cfg.finger_contact_force_threshold
        right_contact = right_contact_force >= self.cfg.finger_contact_force_threshold
        physical_close_ok = physical_close_fraction >= 1.0
        grasp_orientation_ok = vertical_alignment > self.cfg.vertical_alignment_threshold
        phase1_to2_ready = (
            grasp_distance_ok
            & close_commanded
            & gripper_physically_closed
            & grasp_orientation_ok
            & wrist_zero_ok
        )
        grasp_phase_reached = phase >= 2
        grasp_latched = ((phase == 2) | (phase == 3)) & gripper_physically_closed
        lift_threshold_met = object_height > self.cfg.lift_height
        transport_xy_ok = goal_xy_dist < self.cfg.transport_tolerance
        transport_height_ok = (
            goal_transport_height_error < self.cfg.transport_height_tolerance
        )
        place_height_error = torch.abs(object_pos[:, 2] - goal_pos[:, 2])
        place_distance_ok = goal_dist < self.cfg.place_tolerance
        place_height_ok = place_height_error < self.cfg.place_height_tolerance
        object_speed = torch.linalg.norm(self.object.data.root_lin_vel_w.torch, dim=-1)
        object_angular_speed = torch.linalg.norm(
            self.object.data.root_ang_vel_w.torch, dim=-1
        )
        object_tilt = self._object_tilt()
        place_speed_ok = object_speed < self.cfg.place_speed_tolerance
        place_angular_speed_ok = (
            object_angular_speed < self.cfg.place_angular_speed_tolerance
        )
        place_tilt_ok = object_tilt < self.cfg.place_tilt_tolerance
        carry_speed_ok = object_speed < self.cfg.carry_transition_speed_tolerance
        carry_angular_speed_ok = (
            object_angular_speed < self.cfg.carry_transition_angular_speed_tolerance
        )
        carry_tilt_ok = object_tilt < self.cfg.carry_transition_tilt_tolerance
        stable_carry = carry_speed_ok & carry_angular_speed_ok & carry_tilt_ok
        transport_orientation_ok = (
            vertical_alignment > self.cfg.transport_vertical_alignment_threshold
        )
        gripper_released = self._gripper_released_mask()
        success_held = self._success_hold_count >= self.cfg.success_hold_steps
        # _get_dones() advances the phase before this method.  Reconstruct the
        # phase in which each gate was evaluated so transition samples remain
        # in the correct conditional-rate denominator.
        phase_eval = phase - phase_changed.long()

        def conditioned_rate(condition: torch.Tensor, expected_phase: int) -> torch.Tensor:
            selector = phase_eval == expected_phase
            return (condition & selector).float().sum() / selector.float().sum().clamp_min(1.0)

        self.extras["log"] = {
            "Curriculum/stage": torch.tensor(
                {"reach": 0.0, "lift": 1.0, "pick_place": 2.0}[self.cfg.curriculum_stage],
                device=self.device,
            ),
            "Metrics/reach_distance": reach_dist.mean(),
            "Metrics/goal_distance": goal_dist.mean(),
            "Metrics/object_height": object_height.mean(),
            "Metrics/lift_rate": lifted.float().mean(),
            "Metrics/success_rate": success.float().mean(),
            "Metrics/gripper_close": close_command.mean(),
            "Metrics/gripper_driver_position": torch.clamp(
                gripper_driver_position, self._gripper_driver_lower, self._gripper_driver_upper
            ).mean(),
            "Metrics/fingertip_gap": fingertip_gap.mean(),
            "Metrics/gripper_physical_close": physical_close_fraction.mean(),
            "Metrics/object_between_fingertips": object_between_fingertips.float().mean(),
            "Metrics/grasp_maintained": grasp_maintained.float().mean(),
            "Metrics/grasp_loss_steps": self._grasp_loss_steps.float().mean(),
            "Metrics/gripper_latched": grasp_latched.float().mean(),
            "Metrics/grasp_phase_reached": grasp_phase_reached.float().mean(),
            "Metrics/grasp_ever_completed": self._grasp_completed.float().mean(),
            "Metrics/gripper_limit_violation": self._gripper_limit_violation.float().mean(),
            "Metrics/task_phase": phase.float().mean(),
            "Metrics/vertical_alignment": vertical_alignment.mean(),
            "Metrics/pregrasp_distance": pregrasp_dist.mean(),
            "Metrics/pregrasp_xy_distance": pregrasp_xy_dist.mean(),
            "Metrics/pregrasp_height_error": pregrasp_height_error.mean(),
            "Metrics/pregrasp_ready": pregrasp_ready.float().mean(),
            "Metrics/object_xy_displacement": object_xy_displacement.mean(),
            "Metrics/grasp_clearance": grasp_clearance.mean(),
            "Metrics/joint2_position": arm_pos[:, 1].mean(),
            "Metrics/joint3_position": arm_pos[:, 2].mean(),
            "Metrics/joint4_position": arm_pos[:, 3].mean(),
            "Metrics/wrist_position": wrist_position.mean(),
            "Metrics/wrist_abs_error": torch.abs(wrist_position).mean(),
            "Metrics/gripper_lift_distance": gripper_lift_dist.mean(),
            "Metrics/gripper_lift_fraction": gripper_lift_fraction.mean(),
            "Metrics/goal_xy_distance": goal_xy_dist.mean(),
            "Metrics/transport_height_error": goal_transport_height_error.mean(),
            "Metrics/place_height_error": place_height_error.mean(),
            "Metrics/object_speed": object_speed.mean(),
            "Metrics/object_angular_speed": object_angular_speed.mean(),
            "Metrics/object_tilt": object_tilt.mean(),
            "Metrics/task_failed": self._task_failed.float().mean(),
            "Metrics/phase_gate_hold_count": self._phase_gate_hold_count.float().mean(),
            "Metrics/place_pose_hold_count": self._place_pose_hold_count.float().mean(),
            "Phase/phase0_rate": (phase == 0).float().mean(),
            "Phase/phase1_rate": (phase == 1).float().mean(),
            "Phase/phase2_rate": (phase == 2).float().mean(),
            "Phase/phase3_rate": (phase == 3).float().mean(),
            "Phase/phase4_rate": (phase == 4).float().mean(),
            "Phase/eval_phase0_rate": (phase_eval == 0).float().mean(),
            "Phase/eval_phase1_rate": (phase_eval == 1).float().mean(),
            "Phase/eval_phase2_rate": (phase_eval == 2).float().mean(),
            "Conditions/grasp_distance_ok": grasp_distance_ok.float().mean(),
            "Conditions/xy_ok": xy_ok.float().mean(),
            "Conditions/height_ok": height_ok.float().mean(),
            "Conditions/pregrasp_orientation_ok": pregrasp_orientation_ok.float().mean(),
            "Conditions/wrist_zero_ok": wrist_zero_ok.float().mean(),
            "Conditions/pregrasp_ready": pregrasp_ready.float().mean(),
            "Conditions/close_allowed": close_allowed.float().mean(),
            "Conditions/close_commanded": close_commanded.float().mean(),
            "Conditions/driver_close_ok": driver_close_ok.float().mean(),
            "Conditions/gap_lower_ok": gap_lower_ok.float().mean(),
            "Conditions/gap_upper_ok": gap_upper_ok.float().mean(),
            "Conditions/left_finger_contact": left_contact.float().mean(),
            "Conditions/right_finger_contact": right_contact.float().mean(),
            "Conditions/object_between_fingers": object_between_fingertips.float().mean(),
            "Conditions/physical_close_ok": physical_close_ok.float().mean(),
            "Conditions/grasp_orientation_ok": grasp_orientation_ok.float().mean(),
            "Conditions/phase1_to2_ready": phase1_to2_ready.float().mean(),
            "Conditions/grasp_valid": gripper_physically_closed.float().mean(),
            "Conditions/grasp_latched": grasp_latched.float().mean(),
            "Conditions/lift_threshold_met": lift_threshold_met.float().mean(),
            "Conditions/carry_speed_ok": carry_speed_ok.float().mean(),
            "Conditions/carry_angular_speed_ok": carry_angular_speed_ok.float().mean(),
            "Conditions/carry_tilt_ok": carry_tilt_ok.float().mean(),
            "Conditions/stable_carry": stable_carry.float().mean(),
            "Conditions/transport_xy_ok": transport_xy_ok.float().mean(),
            "Conditions/transport_height_ok": transport_height_ok.float().mean(),
            "Conditions/transport_orientation_ok": transport_orientation_ok.float().mean(),
            "Conditions/place_distance_ok": place_distance_ok.float().mean(),
            "Conditions/place_height_ok": place_height_ok.float().mean(),
            "Conditions/place_speed_ok": place_speed_ok.float().mean(),
            "Conditions/place_angular_speed_ok": place_angular_speed_ok.float().mean(),
            "Conditions/place_tilt_ok": place_tilt_ok.float().mean(),
            "Conditions/place_pose_ready": place_pose_ready.float().mean(),
            "Conditions/place_release_authorized": self._place_release_authorized.float().mean(),
            "Conditions/gripper_released": gripper_released.float().mean(),
            "Conditions/success_condition": success.float().mean(),
            "Conditions/success_held": success_held.float().mean(),
            "Gates/phase0_xy_ok": conditioned_rate(xy_ok, 0),
            "Gates/phase0_height_ok": conditioned_rate(height_ok, 0),
            "Gates/phase0_orientation_ok": conditioned_rate(pregrasp_orientation_ok, 0),
            "Gates/phase0_wrist_zero_ok": conditioned_rate(wrist_zero_ok, 0),
            "Gates/phase0_pregrasp_ready": conditioned_rate(pregrasp_ready, 0),
            "Gates/phase1_close_allowed": conditioned_rate(close_allowed, 1),
            "Gates/phase1_capture_distance_ok": conditioned_rate(grasp_distance_ok, 1),
            "Gates/phase1_close_commanded": conditioned_rate(close_commanded, 1),
            "Gates/phase1_driver_close_ok": conditioned_rate(driver_close_ok, 1),
            "Gates/phase1_gap_lower_ok": conditioned_rate(gap_lower_ok, 1),
            "Gates/phase1_gap_upper_ok": conditioned_rate(gap_upper_ok, 1),
            "Gates/phase1_left_contact": conditioned_rate(left_contact, 1),
            "Gates/phase1_right_contact": conditioned_rate(right_contact, 1),
            "Gates/phase1_object_between": conditioned_rate(object_between_fingertips, 1),
            "Gates/phase1_physical_close_ok": conditioned_rate(physical_close_ok, 1),
            "Gates/phase1_orientation_ok": conditioned_rate(grasp_orientation_ok, 1),
            "Gates/phase1_wrist_zero_ok": conditioned_rate(wrist_zero_ok, 1),
            "Gates/phase1_to2_ready": conditioned_rate(phase1_to2_ready, 1),
            "Gates/phase2_lift_threshold_met": conditioned_rate(lift_threshold_met, 2),
            "Gates/phase2_stable_carry": conditioned_rate(stable_carry, 2),
            "Gates/phase3_transport_xy_ok": conditioned_rate(transport_xy_ok, 3),
            "Gates/phase3_transport_height_ok": conditioned_rate(transport_height_ok, 3),
            "Gates/phase3_orientation_ok": conditioned_rate(transport_orientation_ok, 3),
            "Gates/phase3_stable_carry": conditioned_rate(stable_carry, 3),
            "Gates/phase4_place_distance_ok": conditioned_rate(place_distance_ok, 4),
            "Gates/phase4_place_height_ok": conditioned_rate(place_height_ok, 4),
            "Gates/phase4_place_speed_ok": conditioned_rate(place_speed_ok, 4),
            "Gates/phase4_place_angular_speed_ok": conditioned_rate(
                place_angular_speed_ok, 4
            ),
            "Gates/phase4_place_tilt_ok": conditioned_rate(place_tilt_ok, 4),
            "Gates/phase4_release_authorized": conditioned_rate(
                self._place_release_authorized, 4
            ),
            "Gates/phase4_gripper_released": conditioned_rate(gripper_released, 4),
            "Margins/close_allowed": (self.cfg.gripper_close_tolerance - reach_dist).mean(),
            "Margins/pregrasp_xy": (self.cfg.pregrasp_xy_tolerance - pregrasp_xy_dist).mean(),
            "Margins/pregrasp_height": (
                self.cfg.pregrasp_height_tolerance - pregrasp_height_error
            ).mean(),
            "Margins/pregrasp_orientation": (
                vertical_alignment - self.cfg.pregrasp_vertical_alignment_threshold
            ).mean(),
            "Margins/grasp_orientation": (
                vertical_alignment - self.cfg.vertical_alignment_threshold
            ).mean(),
            "Margins/wrist_zero": (
                self._wrist_zero_tolerance - torch.abs(wrist_position)
            ).mean(),
            "Margins/close_command": (close_command - 0.65).mean(),
            "Margins/driver_close": (
                gripper_driver_position - self.cfg.gripper_driver_grasp_min_position
            ).mean(),
            "Margins/gap_lower": (fingertip_gap - self.cfg.gripper_grasp_min_gap).mean(),
            "Margins/gap_upper": (self.cfg.gripper_grasp_max_gap - fingertip_gap).mean(),
            "Margins/physical_close": (physical_close_fraction - 1.0).mean(),
            "Margins/capture_distance": (
                self.cfg.gripper_capture_tolerance - reach_dist
            ).mean(),
            "Margins/left_contact_force": (
                left_contact_force - self.cfg.finger_contact_force_threshold
            ).mean(),
            "Margins/right_contact_force": (
                right_contact_force - self.cfg.finger_contact_force_threshold
            ).mean(),
            "Margins/lift_height": (object_height - self.cfg.lift_height).mean(),
            "Margins/carry_speed": (
                self.cfg.carry_transition_speed_tolerance - object_speed
            ).mean(),
            "Margins/carry_angular_speed": (
                self.cfg.carry_transition_angular_speed_tolerance
                - object_angular_speed
            ).mean(),
            "Margins/carry_tilt": (
                self.cfg.carry_transition_tilt_tolerance - object_tilt
            ).mean(),
            "Margins/transport_xy": (self.cfg.transport_tolerance - goal_xy_dist).mean(),
            "Margins/transport_height": (
                self.cfg.transport_height_tolerance - goal_transport_height_error
            ).mean(),
            "Margins/place_distance": (self.cfg.place_tolerance - goal_dist).mean(),
            "Margins/place_height": (
                self.cfg.place_height_tolerance - place_height_error
            ).mean(),
            "Margins/place_speed": (self.cfg.place_speed_tolerance - object_speed).mean(),
            "Margins/place_angular_speed": (
                self.cfg.place_angular_speed_tolerance - object_angular_speed
            ).mean(),
            "Margins/place_tilt": (self.cfg.place_tilt_tolerance - object_tilt).mean(),
            "Margins/release_driver": (
                self.cfg.gripper_release_max_driver_position - gripper_driver_position
            ).mean(),
            "Margins/release_gap": (
                fingertip_gap - self.cfg.gripper_release_min_gap
            ).mean(),
            "Margins/release_close_fraction": (
                self.cfg.gripper_release_max_close_fraction - physical_close_fraction
            ).mean(),
            "Margins/success_hold_steps": (
                self._success_hold_count.float() - self.cfg.success_hold_steps
            ).mean(),
            "Margins/phase_gate_hold_steps": (
                self._phase_gate_hold_count.float()
                - self._phase_transition_hold_steps[
                    torch.clamp(phase_eval, min=0, max=3)
                ].float()
            ).mean(),
            "Margins/place_pose_hold_steps": (
                self._place_pose_hold_count.float() - self.cfg.place_pose_hold_steps
            ).mean(),
            "Forces/left_finger_contact": left_contact_force.mean(),
            "Forces/right_finger_contact": right_contact_force.mean(),
            "Reward/approach": approach_reward.mean(),
            "Reward/grasp": grasp_reward.mean(),
            "Reward/gripper_close_progress": close_near.mean(),
            "Reward/gripper_close_command": close_command_reward.mean(),
            "Reward/grasp_phase_bonus": grasp_phase_bonus.mean(),
            "Reward/pregrasp_phase_bonus": pregrasp_phase_bonus.mean(),
            "Reward/alignment": alignment_reward.mean(),
            "Reward/phase_progress": phase_progress.mean(),
            "Reward/lift": lift_reward.mean(),
            "Reward/gripper_lift_guidance": gripper_lift_guidance.mean(),
            "Reward/gripper_lift_progress": gripper_lift_progress.mean(),
            "Reward/grasp_hold": grasp_hold_reward.mean(),
            "Reward/transport": transport_reward.mean(),
            "Reward/place": place_reward.mean(),
            "Reward/release": release_reward.mean(),
            "Reward/success": success_bonus.mean(),
            "Penalty/joint_soft_limit": joint_soft_limit_penalty.mean(),
            "Penalty/wrist_zero": wrist_zero_penalty.mean(),
            "Penalty/action_rate": action_rate_penalty.mean(),
            "Penalty/floor_sweep": floor_sweep_penalty.mean(),
            "Penalty/descent_corridor": descent_corridor_penalty.mean(),
            "Penalty/close_far": close_far.mean(),
            "Penalty/object_disturbance": object_disturbance_penalty.mean(),
            "Penalty/grasp_separation": grasp_separation_penalty.mean(),
            "Penalty/terminal_failure": terminal_failure_penalty.mean(),
            "Penalty/carry_stability": (angular_stability_penalty + tilt_penalty).mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Isaac Lab calls dones before rewards.  Advance the phase here so the
        # success, failure and reward calculations all use the same transition.
        phase_before_transition = self._task_phase.clone()
        self._phase_changed = self._advance_task_phase_from_sim()
        place_pose_now = self._place_pose_mask()
        self._place_pose_hold_count = torch.where(
            place_pose_now,
            self._place_pose_hold_count + 1,
            torch.zeros_like(self._place_pose_hold_count),
        )
        self._place_release_authorized |= (
            self._place_pose_hold_count >= self.cfg.place_pose_hold_steps
        )
        success_now = self._success_mask()
        self._success_hold_count = torch.where(
            success_now,
            self._success_hold_count + 1,
            torch.zeros_like(self._success_hold_count),
        )
        success = self._success_hold_count >= self.cfg.success_hold_steps

        object_local = self.object.data.root_pos_w.torch - self._base_pos_w()
        fallen = (object_local[:, 2] < -0.01) | (torch.linalg.norm(object_local[:, :2], dim=-1) > 0.45)
        grasp_clearance = (
            self._gripper_center_w()[:, 2] - self._base_pos_w()[:, 2] - self.cfg.table_top_z
        )
        floor_collision = grasp_clearance < self.cfg.floor_collision_clearance
        object_xy_displacement = torch.linalg.norm(
            self.object.data.root_pos_w.torch[:, :2]
            - self._object_initial_pos_w[:, :2],
            dim=-1,
        )
        captured_now = (
            (self._gripper_physical_close_fraction() >= 1.0)
            & self._object_between_fingertips()
        )
        disturbed_before_capture = (
            (phase_before_transition <= 1)
            & (object_xy_displacement > self.cfg.object_disturbance_failure_distance)
        )
        # Releasing is legal only after the cube has been lowered and settled
        # at the goal.  Gate on a substantial measured opening plus the command
        # to reject transient contact-sensor chatter.
        physical_close_fraction = self._gripper_physical_close_fraction()
        release_control_allowed = self._release_control_allowed_mask()
        premature_place_release = (
            (self._task_phase == 4)
            & (physical_close_fraction < 0.5)
            & ~release_control_allowed
        )
        self._task_failed |= premature_place_release
        gripper_driver_position = self.robot.data.joint_pos[:, self._gripper_driver_joint_id]
        self._gripper_limit_violation = (
            ~torch.isfinite(gripper_driver_position)
            | (
                gripper_driver_position
                < self._gripper_driver_lower - self.cfg.gripper_limit_tolerance_rad
            )
            | (
                gripper_driver_position
                > self._gripper_driver_upper + self.cfg.gripper_limit_tolerance_rad
            )
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return (
            success
            | fallen
            | floor_collision
            | disturbed_before_capture
            | self._gripper_limit_violation
            | self._task_failed
        ), time_out

    def _sample_object_and_goal(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(env_ids)
        object_xy = torch.cat(
            (
                sample_uniform(*self.cfg.object_x_range, (count, 1), device=self.device),
                sample_uniform(*self.cfg.object_y_range, (count, 1), device=self.device),
            ),
            dim=-1,
        )
        goal_xy = torch.cat(
            (
                sample_uniform(*self.cfg.goal_x_range, (count, 1), device=self.device),
                sample_uniform(*self.cfg.goal_y_range, (count, 1), device=self.device),
            ),
            dim=-1,
        )
        for _ in range(8):
            too_close = torch.linalg.norm(goal_xy - object_xy, dim=-1) < self.cfg.min_object_goal_distance
            if not bool(too_close.any()):
                break
            goal_xy[too_close, 0:1] = sample_uniform(
                *self.cfg.goal_x_range, (int(too_close.sum()), 1), device=self.device
            )
            goal_xy[too_close, 1:2] = sample_uniform(
                *self.cfg.goal_y_range, (int(too_close.sum()), 1), device=self.device
            )

        object_z = self.cfg.table_top_z + 0.5 * self.cfg.object_size + 0.002
        goal_z = self.cfg.table_top_z + 0.5 * self.cfg.object_size
        object_local = torch.cat((object_xy, torch.full((count, 1), object_z, device=self.device)), dim=-1)
        goal_local = torch.cat((goal_xy, torch.full((count, 1), goal_z, device=self.device)), dim=-1)
        return object_local, goal_local

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        robot_root = self.robot.data.default_root_state[env_ids].clone()
        robot_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(robot_root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(robot_root[:, 7:], env_ids=env_ids)

        object_local, goal_local = self._sample_object_and_goal(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        initial_arm_pos = torch.tensor(
            self.cfg.initial_arm_positions_rad, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        arm_pos = initial_arm_pos.expand(len(env_ids), -1)
        joint_pos[:, self._arm_joint_ids] = torch.clamp(
            arm_pos, self._controlled_lower[:4], self._controlled_upper[:4]
        )
        joint_pos[:, self._gripper_driver_joint_id] = (
            self.cfg.initial_gripper_driver_position
        )
        joint_pos[:, self._gripper_mimic_joint_id] = self.cfg.gripper_mimic_open_position
        joint_pos[:, self._wrist_joint_id] = 0.0
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_pos

        object_state = self.object.data.default_root_state[env_ids].clone()
        # base_link is the articulation root. robot_root is its desired world pose,
        # so all sampled task coordinates are explicitly relative to base_link=(0, 0, 0).
        base_pos_w = robot_root[:, :3]
        object_state[:, :3] = object_local + base_pos_w
        # Isaac Lab 3 stores quaternions as (x, y, z, w).  Identity therefore
        # has w at root-state index 6, not index 3 (which is quaternion x).
        object_state[:, 3:7] = 0.0
        object_state[:, 6] = 1.0
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)
        self._object_initial_pos_w[env_ids] = object_state[:, :3]
        object_local_up = torch.zeros((len(env_ids), 3), device=self.device)
        object_local_up[:, 2] = 1.0
        self._object_initial_up_w[env_ids] = quat_apply(
            object_state[:, 3:7], object_local_up
        )

        goal_state = self.goal.data.default_root_state[env_ids].clone()
        goal_state[:, :3] = goal_local + base_pos_w
        goal_state[:, 3:7] = 0.0
        goal_state[:, 6] = 1.0
        goal_state[:, 7:] = 0.0
        self.goal.write_root_pose_to_sim(goal_state[:, :7], env_ids=env_ids)
        self.goal.write_root_velocity_to_sim(goal_state[:, 7:], env_ids=env_ids)
        self._goal_pos_w[env_ids] = goal_state[:, :3]

        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._gripper_command[env_ids] = 0.0
        self._success_hold_count[env_ids] = 0
        self._place_pose_hold_count[env_ids] = 0
        self._place_release_authorized[env_ids] = False
        self._phase_gate_hold_count[env_ids] = 0
        self._transport_waypoint_index[env_ids] = 0
        self._transport_waypoint_steps[env_ids] = 0
        self._task_phase[env_ids] = 0
        self._grasp_loss_steps[env_ids] = 0
        self._pregrasp_completed[env_ids] = False
        self._grasp_completed[env_ids] = False
        self._phase_changed[env_ids] = False
        self._task_failed[env_ids] = False
        self._gripper_limit_violation[env_ids] = False
        gripper_pos = self._gripper_center_w()[env_ids]
        grasp_target = object_state[:, :3].clone()
        grasp_target[:, 2] += self.cfg.grasp_center_height_offset
        reach_dist = torch.linalg.norm(grasp_target - gripper_pos, dim=-1)
        goal_dist = torch.linalg.norm(goal_state[:, :3] - object_state[:, :3], dim=-1)
        self._previous_reach_dist[env_ids] = reach_dist
        self._previous_goal_dist[env_ids] = goal_dist
        self._previous_object_height[env_ids] = object_state[:, 2] - (
            self.cfg.table_top_z + 0.5 * self.cfg.object_size
        )
        gripper_lift_target = object_state[:, :3].clone()
        gripper_lift_target[:, 2] += (
            self.cfg.grasp_center_height_offset + self.cfg.lift_height
        )
        self._previous_gripper_lift_dist[env_ids] = torch.linalg.norm(
            gripper_pos - gripper_lift_target, dim=-1
        )
        self._previous_gripper_height[env_ids] = (
            gripper_pos[:, 2]
            - object_state[:, 2]
            - self.cfg.grasp_center_height_offset
        )
        self._previous_physical_close[env_ids] = 0.0
        pregrasp_target = grasp_target.clone()
        pregrasp_target[:, 2] += self.cfg.pregrasp_height
        self._previous_phase_distance[env_ids] = torch.linalg.norm(
            gripper_pos - pregrasp_target, dim=-1
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._previous_actions = self.actions.clone()
        return obs, reward, terminated, truncated, info
