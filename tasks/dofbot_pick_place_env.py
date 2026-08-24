from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
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
        self._grasp_body_id = self._find_body_ids((self.cfg.grasp_reference_body_name,))[0]
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
        self._previous_reach_dist = torch.zeros((self.num_envs,), device=self.device)
        self._previous_goal_dist = torch.zeros((self.num_envs,), device=self.device)
        self._previous_object_height = torch.zeros((self.num_envs,), device=self.device)
        self._previous_phase_distance = torch.zeros((self.num_envs,), device=self.device)
        self._success_hold_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._task_phase = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._grasp_point_offset = torch.tensor(
            self.cfg.grasp_point_offset, dtype=torch.float32, device=self.device
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
        smoothing = min(max(float(self.cfg.gripper_action_smoothing), 0.0), 1.0)
        requested_close = 0.5 * (self.actions[:, 5] + 1.0)
        self._gripper_command = (1.0 - smoothing) * self._gripper_command + smoothing * requested_close

        # Once the calibrated grasp point is on the cube and closing has begun,
        # hold the arm target long enough for the physical mimic linkage to close.
        # Without this dwell, a learned arm delta can sweep straight past the cube.
        grasp_dist = torch.linalg.norm(
            self.object.data.root_pos_w.torch - self._gripper_center_w(), dim=-1
        )
        grasp_dwell = (
            (self._task_phase == 1)
            & (grasp_dist < self.cfg.grasp_dwell_tolerance)
            & (self._gripper_command > 0.65)
        )
        controlled_targets = self._joint_targets[:, self._controlled_joint_ids]
        arm_delta = self.actions[:, :5] * self._joint_action_scales
        arm_delta = torch.where(grasp_dwell.unsqueeze(-1), torch.zeros_like(arm_delta), arm_delta)
        controlled_targets = controlled_targets + arm_delta
        self._joint_targets[:, self._controlled_joint_ids] = torch.clamp(
            controlled_targets, self._controlled_lower, self._controlled_upper
        )

    def _apply_action(self) -> None:
        controlled_targets = self._joint_targets[:, self._controlled_joint_ids]
        gripper_target = self.cfg.gripper_driver_open_target + self._gripper_command * (
            self.cfg.gripper_driver_closed_target - self.cfg.gripper_driver_open_target
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
        """Return the calibrated physical grasp point on Finger_Right_02."""
        body_pos = self.robot.data.body_pos_w.torch[:, self._grasp_body_id]
        body_quat = self.robot.data.body_quat_w.torch[:, self._grasp_body_id]
        offset = self._grasp_point_offset.expand(self.num_envs, -1)
        return body_pos + quat_apply(body_quat, offset)

    def _grasp_approach_axis_w(self) -> torch.Tensor:
        """Return Finger_Right_02 local +Z in world coordinates."""
        body_quat = self.robot.data.body_quat_w.torch[:, self._grasp_body_id]
        local_z = torch.zeros((self.num_envs, 3), device=self.device)
        local_z[:, 2] = 1.0
        return quat_apply(body_quat, local_z)

    def _vertical_alignment(self) -> torch.Tensor:
        # Axis sign is irrelevant: both +Z and -Z normal to the table are vertical.
        return torch.abs(self._grasp_approach_axis_w()[:, 2])

    def _base_pos_w(self) -> torch.Tensor:
        """Return the physical base_link origin used as Pick–Place coordinate zero."""
        return self.robot.data.body_pos_w.torch[:, self._base_body_id]

    def _task_state(self) -> tuple[torch.Tensor, ...]:
        gripper_pos = self._gripper_center_w()
        object_pos = self.object.data.root_pos_w.torch
        goal_pos = self._goal_pos_w
        reach_dist = torch.linalg.norm(object_pos - gripper_pos, dim=-1)
        goal_dist = torch.linalg.norm(goal_pos - object_pos, dim=-1)
        object_height = object_pos[:, 2] - (self.cfg.table_top_z + 0.5 * self.cfg.object_size)
        object_speed = torch.linalg.norm(self.object.data.root_lin_vel_w.torch, dim=-1)
        return gripper_pos, object_pos, goal_pos, reach_dist, goal_dist, object_height, object_speed

    def _get_observations(self) -> dict[str, torch.Tensor]:
        gripper_pos, object_pos, goal_pos, _, _, _, _ = self._task_state()
        base_pos = self._base_pos_w()
        controlled_pos = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        controlled_vel = self.robot.data.joint_vel[:, self._controlled_joint_ids]
        finger_pos = self.robot.data.joint_pos[:, self._finger_observation_joint_ids]
        finger_vel = self.robot.data.joint_vel[:, self._finger_observation_joint_ids]
        object_vel = self.object.data.root_lin_vel_w.torch
        object_ang_vel = self.object.data.root_ang_vel_w.torch
        object_up = torch.zeros((self.num_envs, 3), device=self.device)
        object_up[:, 2] = 1.0
        object_up = quat_apply(self.object.data.root_quat_w.torch, object_up)
        approach_axis = self._grasp_approach_axis_w()
        phase_one_hot = torch.nn.functional.one_hot(self._task_phase, num_classes=5).float()

        obs = torch.cat(
            (
                controlled_pos,
                controlled_vel,
                finger_pos,
                finger_vel,
                gripper_pos - base_pos,
                object_pos - base_pos,
                goal_pos - base_pos,
                object_pos - gripper_pos,
                goal_pos - object_pos,
                object_vel,
                self._gripper_command.unsqueeze(-1),
                self._previous_actions,
                approach_axis,
                phase_one_hot,
                object_up,
                object_ang_vel,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -self.cfg.clip_observations, self.cfg.clip_observations)}

    def _success_mask(self) -> torch.Tensor:
        _, _, _, reach_dist, goal_dist, object_height, object_speed = self._task_state()
        gripper_physically_closed = (
            self.robot.data.joint_pos[:, self._gripper_driver_joint_id]
            > self.cfg.gripper_driver_grasp_threshold
        )
        if self.cfg.curriculum_stage == "reach":
            return (
                (self._task_phase >= 2)
                & gripper_physically_closed
                & (reach_dist < 1.5 * self.cfg.grasp_tolerance)
                & (self._vertical_alignment() > self.cfg.vertical_alignment_threshold)
            )
        if self.cfg.curriculum_stage == "lift":
            return (
                (self._task_phase >= 3)
                & gripper_physically_closed
                & (object_height > self.cfg.lift_height)
            )
        target_height = self.cfg.table_top_z + 0.5 * self.cfg.object_size
        object_z = self.object.data.root_pos_w.torch[:, 2]
        return (
            (goal_dist < self.cfg.place_tolerance)
            & (torch.abs(object_z - target_height) < 0.025)
            & (object_speed < 0.15)
            & (self._gripper_command < 0.35)
        )

    def _update_task_phase(
        self,
        pregrasp_dist: torch.Tensor,
        grasp_dist: torch.Tensor,
        goal_above_dist: torch.Tensor,
        goal_xy_dist: torch.Tensor,
        object_height: torch.Tensor,
        vertical_alignment: torch.Tensor,
        gripper_physically_closed: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one step in the ordered Pick–Place process and return changed envs."""
        phase = self._task_phase
        next_phase = phase.clone()
        pregrasp_ready = (
            (pregrasp_dist < self.cfg.pregrasp_tolerance)
            & (vertical_alignment > self.cfg.vertical_alignment_threshold)
        )
        direct_grasp_ready = (
            (grasp_dist < self.cfg.direct_grasp_entry_tolerance)
            & (vertical_alignment > 0.60)
        )
        next_phase = torch.where(
            (phase == 0) & (pregrasp_ready | direct_grasp_ready),
            torch.ones_like(next_phase),
            next_phase,
        )
        next_phase = torch.where(
            (phase == 1)
            & (grasp_dist < self.cfg.grasp_tolerance)
            & (self._gripper_command > 0.65)
            & gripper_physically_closed,
            torch.full_like(next_phase, 2),
            next_phase,
        )
        next_phase = torch.where(
            (phase == 2) & (object_height > self.cfg.lift_height),
            torch.full_like(next_phase, 3),
            next_phase,
        )
        next_phase = torch.where(
            (phase == 3)
            & (goal_xy_dist < self.cfg.transport_tolerance)
            & (goal_above_dist < 0.075),
            torch.full_like(next_phase, 4),
            next_phase,
        )

        # If a grasp is lost before transport, retry from the pre-grasp phase.
        lost_grasp = (
            ((phase == 2) | (phase == 3))
            & (grasp_dist > 0.075)
            & (object_height < 0.025)
        )
        next_phase = torch.where(lost_grasp, torch.zeros_like(next_phase), next_phase)
        missed_grasp = (phase == 1) & (grasp_dist > 0.10)
        next_phase = torch.where(missed_grasp, torch.zeros_like(next_phase), next_phase)
        changed = next_phase != phase
        self._task_phase = next_phase
        return changed

    def _get_rewards(self) -> torch.Tensor:
        grasp_pos, object_pos, goal_pos, reach_dist, goal_dist, object_height, _ = self._task_state()
        close_command = self._gripper_command
        gripper_driver_position = self.robot.data.joint_pos[:, self._gripper_driver_joint_id]
        close_travel = (
            self.cfg.gripper_driver_grasp_threshold
            - self.cfg.gripper_driver_open_target
        )
        physical_close_fraction = torch.clamp(
            (gripper_driver_position - self.cfg.gripper_driver_open_target) / close_travel,
            0.0,
            1.0,
        )
        gripper_physically_closed = (
            gripper_driver_position > self.cfg.gripper_driver_grasp_threshold
        )
        vertical_alignment = self._vertical_alignment()
        pregrasp_target = object_pos.clone()
        pregrasp_target[:, 2] += self.cfg.pregrasp_height
        goal_above_target = goal_pos.clone()
        goal_above_target[:, 2] += self.cfg.transport_clearance
        pregrasp_dist = torch.linalg.norm(grasp_pos - pregrasp_target, dim=-1)
        goal_above_dist = torch.linalg.norm(object_pos - goal_above_target, dim=-1)
        goal_xy_dist = torch.linalg.norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)

        phase_changed = self._update_task_phase(
            pregrasp_dist,
            reach_dist,
            goal_above_dist,
            goal_xy_dist,
            object_height,
            vertical_alignment,
            gripper_physically_closed,
        )
        phase = self._task_phase
        phase0 = (phase == 0).float()
        phase1 = (phase == 1).float()
        phase2 = (phase == 2).float()
        phase3 = (phase == 3).float()
        phase4 = (phase == 4).float()
        carry = (phase >= 2).float()

        lifted_fraction = torch.clamp(object_height / self.cfg.lift_height, 0.0, 1.5)
        lifted = object_height > 0.025

        approach_reward = phase0 * self.cfg.reach_reward_scale * torch.exp(-pregrasp_dist / 0.06)
        grasp_reward = phase1 * self.cfg.reach_reward_scale * torch.exp(-reach_dist / 0.04)
        alignment_proximity = phase0 * torch.exp(-pregrasp_dist / 0.12) + phase1 * torch.exp(
            -reach_dist / 0.08
        )
        alignment_reward = (
            self.cfg.vertical_alignment_reward_scale
            * vertical_alignment
            * alignment_proximity
        )
        close_near = (
            phase1
            * self.cfg.close_near_object_scale
            * (0.35 * close_command + 0.65 * physical_close_fraction)
            * torch.exp(-reach_dist / 0.025)
        )
        close_far = -phase0 * self.cfg.close_far_penalty_scale * close_command

        lift_reward = phase2 * self.cfg.lift_reward_scale * lifted_fraction
        lift_progress = phase2 * self.cfg.lift_progress_scale * torch.clamp(
            object_height - self._previous_object_height, min=-0.01, max=0.01
        )
        grasp_hold_reward = (
            (phase2 + phase3)
            * physical_close_fraction
            * torch.exp(-reach_dist / 0.055)
        )
        transport_reward = (
            phase3
            * self.cfg.transport_reward_scale
            * torch.exp(-goal_above_dist / 0.08)
        )
        place_reward = phase4 * self.cfg.place_reward_scale * torch.exp(-goal_dist / 0.040)
        release_reward = (
            phase4
            * self.cfg.release_reward_scale
            * (goal_dist < self.cfg.place_tolerance).float()
            * (1.0 - close_command)
        )
        premature_release_penalty = (
            -self.cfg.premature_release_penalty_scale * (phase2 + phase3) * (1.0 - close_command)
        )

        lift_distance = torch.clamp(self.cfg.lift_height - object_height, min=0.0)
        phase_distance = torch.where(
            phase == 0,
            pregrasp_dist,
            torch.where(
                phase == 1,
                reach_dist,
                torch.where(phase == 2, lift_distance, torch.where(phase == 3, goal_above_dist, goal_dist)),
            ),
        )
        phase_progress = self.cfg.phase_progress_reward_scale * torch.clamp(
            self._previous_phase_distance - phase_distance, min=-0.02, max=0.02
        )
        phase_progress = torch.where(phase_changed, torch.zeros_like(phase_progress), phase_progress)
        grasp_phase_bonus = self.cfg.grasp_phase_bonus_scale * (
            phase_changed & (phase == 2)
        ).float()

        stage_reward = (
            approach_reward
            + grasp_reward
            + alignment_reward
            + close_near
            + close_far
            + lift_reward
            + lift_progress
            + grasp_hold_reward
            + transport_reward
            + place_reward
            + release_reward
            + premature_release_penalty
            + phase_progress
            + grasp_phase_bonus
        )

        success = self._success_mask()
        success_bonus = self.cfg.success_bonus * success.float()
        action_penalty = -self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)
        joint2_action_penalty = (
            -self.cfg.joint2_action_penalty_scale * self.actions[:, 1] ** 2
        )
        action_rate = torch.sum((self.actions - self._previous_actions) ** 2, dim=-1)
        action_rate_penalty = -(
            self.cfg.action_rate_penalty_scale
            + carry * self.cfg.carry_action_rate_penalty_scale
        ) * action_rate
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        posture_error = (
            self.cfg.joint2_posture_weight
            * (arm_pos[:, 1] - self.cfg.preferred_joint2_rad) ** 2
            + self.cfg.joint3_posture_weight
            * (arm_pos[:, 2] - self.cfg.preferred_joint3_rad) ** 2
            + 0.5 * (arm_pos[:, 3] - self.cfg.preferred_joint4_rad) ** 2
        )
        posture_penalty = -self.cfg.posture_penalty_scale * torch.clamp(posture_error, max=2.0)
        controlled_vel = self.robot.data.joint_vel[:, self._controlled_joint_ids]
        velocity_penalty = -self.cfg.joint_velocity_penalty_scale * torch.sum(
            controlled_vel**2, dim=-1
        )
        object_ang_vel = self.object.data.root_ang_vel_w.torch
        angular_stability_penalty = (
            -carry
            * self.cfg.carry_angular_velocity_penalty_scale
            * torch.clamp(torch.sum(object_ang_vel**2, dim=-1), max=25.0)
        )
        object_up = torch.zeros((self.num_envs, 3), device=self.device)
        object_up[:, 2] = 1.0
        object_up = quat_apply(self.object.data.root_quat_w.torch, object_up)
        object_tilt = 1.0 - torch.clamp(object_up[:, 2], -1.0, 1.0)
        tilt_penalty = -carry * self.cfg.carry_tilt_penalty_scale * object_tilt
        grasp_clearance = (
            grasp_pos[:, 2] - self._base_pos_w()[:, 2] - self.cfg.table_top_z
        )
        required_clearance = self._phase_min_grasp_clearance[phase]
        floor_sweep_penalty = -self.cfg.floor_sweep_penalty_scale * torch.relu(
            required_clearance - grasp_clearance
        )
        reward = (
            stage_reward
            + success_bonus
            + action_penalty
            + joint2_action_penalty
            + action_rate_penalty
            + velocity_penalty
            + posture_penalty
            + angular_stability_penalty
            + tilt_penalty
            + floor_sweep_penalty
        )

        self._previous_reach_dist = reach_dist.detach()
        self._previous_goal_dist = goal_dist.detach()
        self._previous_object_height = object_height.detach()
        self._previous_phase_distance = phase_distance.detach()
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
            "Metrics/gripper_driver_position": gripper_driver_position.mean(),
            "Metrics/gripper_physical_close": physical_close_fraction.mean(),
            "Metrics/task_phase": phase.float().mean(),
            "Metrics/vertical_alignment": vertical_alignment.mean(),
            "Metrics/pregrasp_distance": pregrasp_dist.mean(),
            "Metrics/grasp_clearance": grasp_clearance.mean(),
            "Metrics/joint2_position": arm_pos[:, 1].mean(),
            "Metrics/joint3_position": arm_pos[:, 2].mean(),
            "Metrics/joint4_position": arm_pos[:, 3].mean(),
            "Reward/approach": approach_reward.mean(),
            "Reward/grasp": grasp_reward.mean(),
            "Reward/grasp_phase_bonus": grasp_phase_bonus.mean(),
            "Reward/alignment": alignment_reward.mean(),
            "Reward/phase_progress": phase_progress.mean(),
            "Reward/lift": lift_reward.mean(),
            "Reward/transport": transport_reward.mean(),
            "Reward/place": place_reward.mean(),
            "Reward/success": success_bonus.mean(),
            "Reward/posture": posture_penalty.mean(),
            "Penalty/action_rate": action_rate_penalty.mean(),
            "Penalty/joint2_action": joint2_action_penalty.mean(),
            "Penalty/floor_sweep": floor_sweep_penalty.mean(),
            "Penalty/carry_stability": (angular_stability_penalty + tilt_penalty).mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success | fallen | floor_collision, time_out

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
        arm_noise = sample_uniform(
            -self.cfg.initial_joint_noise_rad,
            self.cfg.initial_joint_noise_rad,
            (len(env_ids), len(self._arm_joint_ids)),
            device=self.device,
        )
        initial_arm_pos = torch.tensor(
            self.cfg.initial_arm_positions_rad, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        arm_pos = initial_arm_pos + arm_noise
        # For this asset, joint1=0 points the arm along base_link +Y and positive
        # joint1 rotates +Y toward -X. Therefore -atan2(x, y) faces the cube.
        arm_pos[:, 0] = -torch.atan2(object_local[:, 0], object_local[:, 1]) + arm_noise[:, 0]
        joint_pos[:, self._arm_joint_ids] = torch.clamp(
            arm_pos, self._controlled_lower[:4], self._controlled_upper[:4]
        )
        joint_pos[:, self._gripper_driver_joint_id] = self.cfg.gripper_driver_open_target
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
        object_state[:, 3:7] = 0.0
        object_state[:, 3] = 1.0
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)

        goal_state = self.goal.data.default_root_state[env_ids].clone()
        goal_state[:, :3] = goal_local + base_pos_w
        goal_state[:, 3:7] = 0.0
        goal_state[:, 3] = 1.0
        goal_state[:, 7:] = 0.0
        self.goal.write_root_pose_to_sim(goal_state[:, :7], env_ids=env_ids)
        self.goal.write_root_velocity_to_sim(goal_state[:, 7:], env_ids=env_ids)
        self._goal_pos_w[env_ids] = goal_state[:, :3]

        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._gripper_command[env_ids] = 0.0
        self._success_hold_count[env_ids] = 0
        self._task_phase[env_ids] = 0
        gripper_pos = self._gripper_center_w()[env_ids]
        reach_dist = torch.linalg.norm(object_state[:, :3] - gripper_pos, dim=-1)
        goal_dist = torch.linalg.norm(goal_state[:, :3] - object_state[:, :3], dim=-1)
        self._previous_reach_dist[env_ids] = reach_dist
        self._previous_goal_dist[env_ids] = goal_dist
        self._previous_object_height[env_ids] = object_state[:, 2] - (
            self.cfg.table_top_z + 0.5 * self.cfg.object_size
        )
        pregrasp_target = object_state[:, :3].clone()
        pregrasp_target[:, 2] += self.cfg.pregrasp_height
        self._previous_phase_distance[env_ids] = torch.linalg.norm(
            gripper_pos - pregrasp_target, dim=-1
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._previous_actions = self.actions.clone()
        return obs, reward, terminated, truncated, info
