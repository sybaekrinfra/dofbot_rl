from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, sample_uniform, subtract_frame_transforms

from .dofbot_reach_cfg import DofbotReachEnvCfg


class DofbotReachEnv(DirectRLEnv):
    """PPO-ready DOFBOT reaching environment built with Isaac Lab's direct workflow."""

    cfg: DofbotReachEnvCfg

    def __init__(self, cfg: DofbotReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self.target: RigidObject = self.scene["target"]
        self._control_mode = self.cfg.control_mode
        if self._control_mode not in ("joint_delta", "cartesian_ik"):
            raise ValueError(f"Unsupported DOFBOT control_mode: {self._control_mode}")

        # Cache only the arm joints. The USD also contains gripper joints under
        # link5, but the reach policy should not command those.
        self._joint_ids: list[int] = []
        self._joint_names: list[str] = []
        for joint_name in self.cfg.arm_joint_names:
            joint_ids, joint_names = self.robot.find_joints(joint_name)
            if len(joint_ids) == 0:
                all_joint_ids, all_joint_names = self.robot.find_joints(".*")
                raise RuntimeError(
                    f"Could not find DOFBOT arm joint '{joint_name}'. "
                    f"Available joints: {all_joint_names}"
                )
            self._joint_ids.append(int(joint_ids[0]))
            self._joint_names.append(joint_names[0])
        if self._control_mode == "joint_delta" and len(self._joint_ids) != self.cfg.action_space:
            raise RuntimeError(
                f"Configured {len(self._joint_ids)} arm joints ({self._joint_names}), "
                f"but action_space={self.cfg.action_space}."
            )
        if self._control_mode == "cartesian_ik" and self.cfg.action_space != 3:
            raise RuntimeError(f"Cartesian IK control expects action_space=3, but got {self.cfg.action_space}.")

        ee_body_ids, ee_body_names = self.robot.find_bodies(self.cfg.ee_body_name)
        if len(ee_body_ids) == 0:
            raise RuntimeError(f"Could not find end-effector body matching '{self.cfg.ee_body_name}'.")
        self._ee_body_id = int(ee_body_ids[0])
        self._ee_body_name = ee_body_names[0]
        self._ee_tip_offset = torch.tensor(self.cfg.ee_tip_offset, dtype=torch.float32, device=self.device)

        self._action_scale = float(self.cfg.action_scale)
        self._target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._last_cartesian_delta_norm = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._last_joint_target_delta_norm = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._episode_success_rate = torch.zeros((), dtype=torch.float32, device=self.device)
        self._episode_done_rate = torch.zeros((), dtype=torch.float32, device=self.device)
        self._episode_min_dist = torch.full((self.num_envs,), float("inf"), dtype=torch.float32, device=self.device)
        self._prev_dist = torch.full((self.num_envs,), float("inf"), dtype=torch.float32, device=self.device)
        self._success_hold_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._episode_min_dist_ema = torch.zeros((), dtype=torch.float32, device=self.device)
        self._success_distance_ema = torch.zeros((), dtype=torch.float32, device=self.device)
        self._demo_mode = bool(self.cfg.demo_mode)
        self._disable_auto_reset = bool(self.cfg.disable_auto_reset or self.cfg.demo_mode)

        joint_limits = self._get_configured_joint_limits()
        self._joint_lower_limits = joint_limits[:, 0]
        self._joint_upper_limits = joint_limits[:, 1]
        self._ee_jacobi_idx = self._ee_body_id - 1 if self.robot.is_fixed_base else self._ee_body_id
        self._diff_ik_controller: DifferentialIKController | None = None
        if self._control_mode == "cartesian_ik":
            ik_cfg = DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.03},
            )
            self._diff_ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

    def _get_configured_joint_limits(self) -> torch.Tensor:
        """Use explicit DOFBOT limits, falling back to asset limits when omitted."""
        lower_limits = getattr(self.cfg, "joint_lower_limits_deg", ())
        upper_limits = getattr(self.cfg, "joint_upper_limits_deg", ())
        if len(lower_limits) == len(self._joint_ids) and len(upper_limits) == len(self._joint_ids):
            return torch.deg2rad(
                torch.tensor(
                    list(zip(lower_limits, upper_limits)),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        return self._get_asset_joint_limits()

    def _get_asset_joint_limits(self) -> torch.Tensor:
        """Read controlled joint limits from the loaded USD/URDF articulation."""
        joint_limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
        if joint_limits is not None:
            if joint_limits.dim() == 3:
                joint_limits = joint_limits[0, self._joint_ids]
            else:
                joint_limits = joint_limits[self._joint_ids]
            joint_limits = joint_limits.to(device=self.device, dtype=torch.float32)
            valid_limits = torch.isfinite(joint_limits).all(dim=-1) & (joint_limits[:, 1] > joint_limits[:, 0])
            if bool(valid_limits.all()):
                return joint_limits

        # Fallback matching dofbot.usda: joint1~joint4 are +/-90 degrees.
        fallback = math.pi / 2
        return torch.tensor(
            [[-fallback, fallback]] * len(self._joint_ids),
            dtype=torch.float32,
            device=self.device,
        )

    def _setup_scene(self):
        """Spawn the scene assets and ground plane."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.target = RigidObject(self.cfg.target_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.to(self.device), -self.cfg.clip_actions, self.cfg.clip_actions)

    def _apply_action(self) -> None:
        """Convert normalized actions into joint position targets."""
        if self._control_mode == "cartesian_ik":
            self._apply_cartesian_ik_action()
        else:
            self._apply_joint_delta_action()

    def _apply_joint_delta_action(self) -> None:
        """Convert normalized actions into arm joint-position deltas."""
        ee_pos = self._get_end_effector_position()
        target_pos = self._get_target_position()
        dist = torch.linalg.norm(ee_pos - target_pos, dim=-1)
        precision_blend = torch.clamp(
            (dist - self.cfg.success_tolerance) / (self.cfg.precision_action_radius - self.cfg.success_tolerance),
            min=0.0,
            max=1.0,
        ).unsqueeze(-1)
        action_scale = self.cfg.precision_action_scale + (
            self._action_scale - self.cfg.precision_action_scale
        ) * precision_blend
        delta = self.actions * action_scale
        previous_joint_targets = self._joint_targets[:, self._joint_ids].clone()
        self._joint_targets[:, self._joint_ids] = torch.clamp(
            self._joint_targets[:, self._joint_ids] + delta,
            self._joint_lower_limits,
            self._joint_upper_limits,
        )
        self._last_cartesian_delta_norm[:] = 0.0
        self._last_joint_target_delta_norm = torch.linalg.norm(
            self._joint_targets[:, self._joint_ids] - previous_joint_targets,
            dim=-1,
        )
        self.robot.set_joint_position_target(self._joint_targets[:, self._joint_ids], joint_ids=self._joint_ids)

    def _apply_cartesian_ik_action(self) -> None:
        """Convert normalized Cartesian EE deltas into arm joint targets through differential IK."""
        if self._diff_ik_controller is None:
            raise RuntimeError("Differential IK controller is not initialized.")

        ee_tip_pos_w = self._get_end_effector_position()
        target_pos_w = self._get_target_position()
        dist = torch.linalg.norm(ee_tip_pos_w - target_pos_w, dim=-1)
        precision_blend = torch.clamp(
            (dist - self.cfg.success_tolerance) / (self.cfg.cartesian_action_radius - self.cfg.success_tolerance),
            min=0.0,
            max=1.0,
        ).unsqueeze(-1)
        action_scale = self.cfg.precision_cartesian_action_scale + (
            self.cfg.cartesian_action_scale - self.cfg.precision_cartesian_action_scale
        ) * precision_blend

        delta_pos_w = self.actions * action_scale
        root_quat_w = self.robot.data.root_quat_w.torch
        delta_pos_b = quat_apply(quat_inv(root_quat_w), delta_pos_w)

        jacobi_joint_ids = [joint_id + self.robot.num_base_dofs for joint_id in self._joint_ids]
        jacobian_w = self.robot.data.body_link_jacobian_w.torch[
            :, self._ee_jacobi_idx, :, jacobi_joint_ids
        ]
        root_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

        body_pos_w = self.robot.data.body_pos_w.torch[:, self._ee_body_id]
        body_quat_w = self.robot.data.body_quat_w.torch[:, self._ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch,
            root_quat_w,
            body_pos_w,
            body_quat_w,
        )
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids]

        self._diff_ik_controller.set_command(delta_pos_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = self._diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        joint_pos_des = torch.clamp(joint_pos_des, self._joint_lower_limits, self._joint_upper_limits)

        previous_joint_targets = self._joint_targets[:, self._joint_ids].clone()
        self._joint_targets[:, self._joint_ids] = joint_pos_des
        self._last_cartesian_delta_norm = torch.linalg.norm(delta_pos_w, dim=-1)
        self._last_joint_target_delta_norm = torch.linalg.norm(joint_pos_des - previous_joint_targets, dim=-1)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)

    def _get_end_effector_position(self) -> torch.Tensor:
        body_state = getattr(self.robot.data, "body_state_w", None)
        if body_state is None:
            body_state = getattr(self.robot.data, "body_com_state_w", None)
        if body_state is None:
            raise RuntimeError("Robot articulation does not expose body state buffers in this Isaac Lab build.")

        ee_state = body_state[:, self._ee_body_id]
        ee_pos = ee_state[:, :3]
        ee_quat = ee_state[:, 3:7]
        ee_tip_offset = self._ee_tip_offset.expand(ee_quat.shape[0], -1)
        return ee_pos + quat_apply(ee_quat, ee_tip_offset)

    def _get_target_position(self) -> torch.Tensor:
        """Return the latest target position used by policy observations and reward."""
        return self._target_pos

    def set_target_position(
        self,
        target_pos: torch.Tensor | Sequence[float],
        *,
        frame: str = "env",
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update the reach target buffer and visual target prim from an external source.

        Args:
            target_pos: Target xyz. Shape can be (3,), (1, 3), or (num_envs, 3).
            frame: ``"env"`` for robot-base/environment-local coordinates or ``"world"`` for world coordinates.
            env_ids: Optional environment ids to update. Defaults to all envs.

        Returns:
            The target position in world coordinates.
        """
        if frame not in ("env", "world"):
            raise ValueError(f"Unsupported target frame '{frame}'. Expected 'env' or 'world'.")
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        target = torch.as_tensor(target_pos, dtype=torch.float32, device=self.device)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if target.shape[-1] != 3:
            raise ValueError(f"Expected target xyz with last dimension 3, got shape {tuple(target.shape)}.")
        if target.shape[0] == 1 and len(env_ids_tensor) > 1:
            target = target.expand(len(env_ids_tensor), -1)
        elif target.shape[0] != len(env_ids_tensor):
            raise ValueError(
                f"Target batch size {target.shape[0]} does not match env_ids length {len(env_ids_tensor)}."
            )

        target_w = target if frame == "world" else target + self.scene.env_origins[env_ids_tensor]
        self._target_pos[env_ids_tensor] = target_w

        target_root = self.target.data.root_state_w[env_ids_tensor].clone()
        target_root[:, :3] = target_w
        target_root[:, 3:7] = self.target.data.default_root_state[env_ids_tensor, 3:7]
        self.target.write_root_pose_to_sim(target_root[:, :7], env_ids=env_ids_tensor)
        return target_w

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._joint_ids]
        ee_pos = self._get_end_effector_position()
        target_pos = self._get_target_position()
        ee_pos_env = ee_pos - self.scene.env_origins
        target_pos_env = target_pos - self.scene.env_origins
        target_delta = target_pos_env - ee_pos_env

        obs_parts = [joint_pos, joint_vel, ee_pos_env, target_pos_env, target_delta]
        if self.cfg.use_previous_action:
            obs_parts.append(self._prev_action)
        obs = torch.cat(obs_parts, dim=-1)

        self._ee_pos = ee_pos
        return {"policy": torch.clamp(obs, -self.cfg.clip_observations, self.cfg.clip_observations)}

    def _compute_reward(
        self,
        update_progress: bool = False,
        held_success: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos = self._get_end_effector_position()
        target_pos = self._get_target_position()
        dist = torch.linalg.norm(ee_pos - target_pos, dim=-1)
        success = dist < self.cfg.success_tolerance
        if held_success is None:
            predicted_hold_count = torch.where(
                success,
                self._success_hold_count + 1,
                torch.zeros_like(self._success_hold_count),
            )
            held_success = predicted_hold_count >= self.cfg.success_hold_steps

        distance_reward = -self.cfg.distance_reward_scale * dist
        prev_dist = torch.where(torch.isfinite(self._prev_dist), self._prev_dist, dist)
        progress_reward = self.cfg.progress_reward_scale * (prev_dist - dist)
        prev_best_dist = torch.where(torch.isfinite(self._episode_min_dist), self._episode_min_dist, dist)
        best_distance_reward = self.cfg.best_distance_reward_scale * torch.clamp(prev_best_dist - dist, min=0.0)
        reach_reward = self.cfg.reach_reward_scale * torch.exp(-dist / self.cfg.reach_reward_std)
        funnel_progress = torch.clamp(
            (self.cfg.funnel_reward_radius - dist) / (self.cfg.funnel_reward_radius - self.cfg.success_tolerance),
            min=0.0,
            max=1.0,
        )
        funnel_reward = self.cfg.funnel_reward_scale * torch.square(funnel_progress)
        fine_reward = self.cfg.fine_reward_scale * torch.exp(-torch.square(dist / self.cfg.fine_reward_std))
        precision_reward = self.cfg.precision_reward_scale * torch.exp(
            -torch.square(dist / self.cfg.precision_reward_std)
        )
        ultra_precision_reward = self.cfg.ultra_precision_reward_scale * torch.exp(
            -torch.square(dist / self.cfg.ultra_precision_reward_std)
        )
        close_reward_2cm = self.cfg.close_reward_2cm * torch.clamp((0.02 - dist) / 0.02, min=0.0, max=1.0)
        close_reward_1cm = self.cfg.close_reward_1cm * torch.clamp((0.01 - dist) / 0.01, min=0.0, max=1.0)
        close_reward_0_5cm = self.cfg.close_reward_0_5cm * torch.clamp(
            (0.005 - dist) / 0.005,
            min=0.0,
            max=1.0,
        )
        action_penalty = -self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)
        near_action_weight = torch.clamp(
            (self.cfg.near_action_penalty_radius - dist) / self.cfg.near_action_penalty_radius,
            min=0.0,
            max=1.0,
        )
        near_action_penalty = -self.cfg.near_action_penalty_scale * near_action_weight * torch.sum(
            torch.square(self.actions - self._prev_action),
            dim=-1,
        )
        joint_vel = self.robot.data.joint_vel[:, self._joint_ids]
        joint_velocity_penalty = -self.cfg.joint_velocity_penalty_scale * torch.sum(torch.square(joint_vel), dim=-1)
        time_penalty = -self.cfg.time_penalty * torch.ones_like(dist)
        success_bonus = torch.where(
            held_success,
            torch.full_like(dist, self.cfg.success_bonus),
            torch.zeros_like(dist),
        )

        reward = (
            distance_reward
            + progress_reward
            + best_distance_reward
            + reach_reward
            + funnel_reward
            + fine_reward
            + precision_reward
            + ultra_precision_reward
            + close_reward_2cm
            + close_reward_1cm
            + close_reward_0_5cm
            + action_penalty
            + near_action_penalty
            + joint_velocity_penalty
            + time_penalty
            + success_bonus
        )
        if update_progress:
            self._prev_dist = dist.detach()
        self._episode_min_dist = torch.minimum(self._episode_min_dist, dist)
        self._ee_pos = ee_pos
        self.extras["log"] = {
            "Metrics/ee_target_distance": dist.mean(),
            "Metrics/instant_success_rate": success.float().mean(),
            "Metrics/episode_success_rate_ema": self._episode_success_rate,
            "Metrics/episode_done_rate_ema": self._episode_done_rate,
            "Metrics/episode_min_distance_ema": self._episode_min_dist_ema,
            "Metrics/success_distance_ema": self._success_distance_ema,
            "Metrics/current_episode_min_distance": self._episode_min_dist.mean(),
            "Metrics/action_mode": torch.tensor(
                1.0 if self._control_mode == "cartesian_ik" else 0.0,
                dtype=torch.float32,
                device=self.device,
            ),
            "Metrics/cartesian_delta_norm": self._last_cartesian_delta_norm.mean(),
            "Metrics/joint_target_delta_norm": self._last_joint_target_delta_norm.mean(),
            "Metrics/held_success_rate": (
                self._success_hold_count >= self.cfg.success_hold_steps
            ).float().mean(),
            "Metrics/success_hold_count": self._success_hold_count.float().mean(),
            "Metrics/close_rate_0_5cm": (dist < 0.005).float().mean(),
            "Metrics/close_rate_0_3cm": (dist < self.cfg.target_precision).float().mean(),
            "Metrics/close_rate_1cm": (dist < 0.01).float().mean(),
            "Metrics/close_rate_2cm": (dist < 0.02).float().mean(),
            "Metrics/close_rate_5cm": (dist < 0.05).float().mean(),
            "Metrics/close_rate_8cm": (dist < 0.08).float().mean(),
            "Metrics/close_rate_10cm": (dist < 0.10).float().mean(),
            "Reward/distance": distance_reward.mean(),
            "Reward/progress": progress_reward.mean(),
            "Reward/best_distance": best_distance_reward.mean(),
            "Reward/reach": reach_reward.mean(),
            "Reward/funnel": funnel_reward.mean(),
            "Reward/fine": fine_reward.mean(),
            "Reward/precision": precision_reward.mean(),
            "Reward/ultra_precision": ultra_precision_reward.mean(),
            "Reward/close_2cm": close_reward_2cm.mean(),
            "Reward/close_1cm": close_reward_1cm.mean(),
            "Reward/close_0_5cm": close_reward_0_5cm.mean(),
            "Reward/near_action_penalty": near_action_penalty.mean(),
            "Reward/joint_velocity_penalty": joint_velocity_penalty.mean(),
            "Reward/success_bonus": success_bonus.mean(),
        }
        return reward, success

    def _get_rewards(self) -> torch.Tensor:
        reward, _ = self._compute_reward(update_progress=True)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, instant_success = self._compute_reward(update_progress=False)
        next_success_hold_count = torch.where(
            instant_success,
            self._success_hold_count + 1,
            torch.zeros_like(self._success_hold_count),
        )
        self._success_hold_count = next_success_hold_count
        success = self._success_hold_count >= self.cfg.success_hold_steps
        if self._disable_auto_reset:
            never_done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.extras.setdefault("log", {})
            self.extras["log"]["Metrics/demo_mode"] = torch.tensor(
                1.0 if self._demo_mode else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
            return never_done, never_done

        episode_buf = getattr(self, "episode_length_buf", None)
        if episode_buf is None:
            episode_buf = getattr(self, "progress_buf")
        time_out = episode_buf >= (self.max_episode_length - 1)
        done = success | time_out
        done_count = done.sum()
        if done_count > 0:
            episode_success_rate = success[done].float().mean()
            episode_done_rate = done.float().mean()
            episode_min_dist = self._episode_min_dist[done].mean()
            self._episode_success_rate = 0.9 * self._episode_success_rate + 0.1 * episode_success_rate
            self._episode_done_rate = 0.9 * self._episode_done_rate + 0.1 * episode_done_rate
            self._episode_min_dist_ema = 0.9 * self._episode_min_dist_ema + 0.1 * episode_min_dist
            if bool(success.any()):
                success_distance = self._episode_min_dist[success].mean()
                self._success_distance_ema = 0.9 * self._success_distance_ema + 0.1 * success_distance
        return success, time_out

    def _sample_target_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        num = len(env_ids)
        x = sample_uniform(self.cfg.target_x_range[0], self.cfg.target_x_range[1], (num, 1), device=self.device)
        y = sample_uniform(self.cfg.target_y_range[0], self.cfg.target_y_range[1], (num, 1), device=self.device)
        z = sample_uniform(self.cfg.target_z_range[0], self.cfg.target_z_range[1], (num, 1), device=self.device)

        # Place the target in front of the robot in world coordinates.
        target = torch.cat([x, y, z], dim=-1)
        target += self.scene.env_origins[env_ids]
        return target

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        # Reset the robot to a safe default pose and clear cached commands.
        robot_root = self.robot.data.default_root_state[env_ids].clone()
        robot_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(robot_root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(robot_root[:, 7:], env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids][:, self._joint_ids].clone()
        joint_pos += sample_uniform(-0.05, 0.05, joint_pos.shape, device=self.device)
        joint_pos = torch.clamp(joint_pos, self._joint_lower_limits, self._joint_upper_limits)
        self._joint_targets[env_ids[:, None], self._joint_ids] = joint_pos
        joint_state = self.robot.data.default_joint_pos[env_ids].clone()
        joint_state[:, self._joint_ids] = joint_pos
        joint_vel = torch.zeros_like(joint_state)
        joint_vel_state = self.robot.data.default_joint_vel[env_ids].clone()
        joint_vel_state[:] = joint_vel
        self.robot.write_joint_state_to_sim(joint_state, joint_vel_state, env_ids=env_ids)

        target_root = self.target.data.default_root_state[env_ids].clone()
        if self._demo_mode or self.cfg.use_ros_target:
            target_root[:, :3] += self.scene.env_origins[env_ids]
            self.set_target_position(target_root[:, :3], frame="world", env_ids=env_ids)
        else:
            # Randomize the target position every training reset.
            self.set_target_position(self._sample_target_positions(env_ids), frame="world", env_ids=env_ids)

        self._episode_min_dist[env_ids] = float("inf")
        self._prev_dist[env_ids] = float("inf")
        self._success_hold_count[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._last_cartesian_delta_norm[env_ids] = 0.0
        self._last_joint_target_delta_norm[env_ids] = 0.0
        if self._diff_ik_controller is not None:
            self._diff_ik_controller.reset(env_ids)

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        self._prev_action = self.actions.clone()
        return obs, rew, terminated, truncated, info
