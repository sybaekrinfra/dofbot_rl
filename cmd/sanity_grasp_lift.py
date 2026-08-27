from __future__ import annotations

"""Deterministic DOFBOT_V2 grasp/lift or full Pick-Place preflight.

This deliberately does not load an RL checkpoint.  Differential IK drives the
normal six-action environment interface so gripper gating, phase transitions,
contact checks, rewards, termination, and reset logic are all exercised.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Run a deterministic DOFBOT grasp/lift or full Pick-Place sanity test."
)
parser.add_argument("--object_x", type=float, default=0.0)
parser.add_argument("--object_y", type=float, default=0.13)
parser.add_argument("--goal_x", type=float, default=0.0)
parser.add_argument("--goal_y", type=float, default=0.19)
parser.add_argument("--log_interval", type=int, default=30)
parser.add_argument(
    "--gripper_closed_target",
    type=float,
    default=None,
    help="Optional diagnostic override for the right-finger closed target in radians.",
)
parser.add_argument(
    "--gripper_effort_limit",
    type=float,
    default=None,
    help="Optional diagnostic override for the right-finger actuator effort limit.",
)
parser.add_argument(
    "--arm_effort_limit",
    type=float,
    default=None,
    help="Optional diagnostic override for joint1..joint5 actuator effort limits.",
)
parser.add_argument(
    "--object_static_friction",
    type=float,
    default=None,
    help="Optional diagnostic override for the cube static-friction coefficient.",
)
parser.add_argument(
    "--object_dynamic_friction",
    type=float,
    default=None,
    help="Optional diagnostic override for the cube dynamic-friction coefficient.",
)
parser.add_argument(
    "--grip_solver_mode",
    action="store_true",
    help=(
        "Diagnostic PhysX mode: solve articulation contacts last, apply external "
        "forces each TGS iteration, and disable the unnecessary stabilization pass."
    ),
)
parser.add_argument(
    "--robot_solver_position_iterations",
    type=int,
    default=None,
    help="Optional diagnostic articulation/rigid-body position solver iteration count.",
)
parser.add_argument(
    "--robot_solver_velocity_iterations",
    type=int,
    default=None,
    help="Optional diagnostic articulation/rigid-body velocity solver iteration count.",
)
parser.add_argument(
    "--full_pick_place",
    action="store_true",
    help="Continue after lift through transport, lower, measured release, and held success.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply, skew_symmetric_matrix

import dofbot_rl.tasks  # noqa: F401
from dofbot_rl.tasks import PICK_PLACE_ENV_ID, PICK_PLACE_LIFT_ENV_ID
from dofbot_rl.tasks.dofbot_pick_place_cfg import (
    DofbotPickPlaceEnvCfg,
    DofbotPickPlaceLiftEnvCfg,
)


class SanityFailure(RuntimeError):
    """A deterministic manipulation invariant did not pass."""


def _value(value: torch.Tensor | float | int | bool) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def _log_value(info: dict, name: str) -> float:
    return _value(info.get("log", {}).get(name, 0.0))


def main() -> None:
    cfg = DofbotPickPlaceEnvCfg() if args_cli.full_pick_place else DofbotPickPlaceLiftEnvCfg()
    env_id = PICK_PLACE_ENV_ID if args_cli.full_pick_place else PICK_PLACE_LIFT_ENV_ID
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 60.0
    cfg.initial_joint_noise_rad = 0.0
    cfg.object_x_range = (args_cli.object_x, args_cli.object_x)
    cfg.object_y_range = (args_cli.object_y, args_cli.object_y)
    cfg.goal_x_range = (args_cli.goal_x, args_cli.goal_x)
    cfg.goal_y_range = (args_cli.goal_y, args_cli.goal_y)
    if args_cli.gripper_closed_target is not None:
        cfg.gripper_driver_closed_target = args_cli.gripper_closed_target
    if args_cli.gripper_effort_limit is not None:
        cfg.robot_cfg.actuators["gripper_driver"].effort_limit_sim = (
            args_cli.gripper_effort_limit
        )
    if args_cli.arm_effort_limit is not None:
        cfg.robot_cfg.actuators["arm"].effort_limit_sim = args_cli.arm_effort_limit
        cfg.robot_cfg.actuators["wrist"].effort_limit_sim = args_cli.arm_effort_limit
    if args_cli.object_static_friction is not None:
        cfg.object_cfg.spawn.physics_material.static_friction = (
            args_cli.object_static_friction
        )
    if args_cli.object_dynamic_friction is not None:
        cfg.object_cfg.spawn.physics_material.dynamic_friction = (
            args_cli.object_dynamic_friction
        )
    if args_cli.grip_solver_mode:
        cfg.sim.physics.solve_articulation_contact_last = True
        cfg.sim.physics.enable_external_forces_every_iteration = True
        cfg.sim.physics.enable_stabilization = False
    if args_cli.robot_solver_position_iterations is not None:
        cfg.robot_cfg.spawn.rigid_props.solver_position_iteration_count = (
            args_cli.robot_solver_position_iterations
        )
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = (
            args_cli.robot_solver_position_iterations
        )
    if args_cli.robot_solver_velocity_iterations is not None:
        cfg.robot_cfg.spawn.rigid_props.solver_velocity_iteration_count = (
            args_cli.robot_solver_velocity_iterations
        )
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = (
            args_cli.robot_solver_velocity_iterations
        )
    env = gym.make(env_id, cfg=cfg, render_mode=None)
    u = env.unwrapped
    arm_ids = list(u._arm_joint_ids)
    jacobian_columns = [joint_id + u.robot.num_base_dofs for joint_id in arm_ids]
    jacobian_body = u._grasp_body_id - 1 if u.robot.is_fixed_base else u._grasp_body_id
    # Batched-FK-verified branch for local object (x=0, y=0.13).  All values
    # obey the physical +/-90-degree limits.  In particular joint3 stays on
    # the negative branch requested for this mechanism.
    pregrasp_joint_target = torch.tensor(
        [[0.00634881, 0.58330178, -1.49433029, -1.56902897]], device=u.device
    )
    grasp_joint_target = torch.tensor(
        [[0.00772026, -0.07229876, -1.28563094, -1.57079633]], device=u.device
    )
    zero_action = torch.zeros((1, u.cfg.action_space), device=u.device)
    zero_action[:, 5] = -1.0

    last_info: dict = {}
    completed_steps = 0

    def snapshot(label: str, step: int) -> dict[str, float]:
        grasp_pos = u._gripper_center_w()
        object_pos = u.object.data.root_pos_w.torch
        left_force, right_force = u._finger_contact_forces()
        left_by_body = torch.linalg.norm(
            u.left_finger_contact.data.net_forces_w.torch, dim=-1
        )[0].tolist()
        right_by_body = torch.linalg.norm(
            u.right_finger_contact.data.net_forces_w.torch, dim=-1
        )[0].tolist()
        left_matrix = u.left_finger_contact.data.force_matrix_w
        right_matrix = u.right_finger_contact.data.force_matrix_w
        left_object_force_w = (
            left_matrix.torch.sum(dim=(1, 2))
            if left_matrix is not None
            else u.left_finger_contact.data.net_forces_w.torch.sum(dim=1)
        )
        right_object_force_w = (
            right_matrix.torch.sum(dim=(1, 2))
            if right_matrix is not None
            else u.right_finger_contact.data.net_forces_w.torch.sum(dim=1)
        )
        object_local_z = torch.zeros((1, 3), device=u.device)
        object_local_z[:, 2] = 1.0
        object_up_w = quat_apply(u.object.data.root_quat_w.torch, object_local_z)
        values = {
            "phase": _value(u._task_phase),
            "distance": _value(torch.linalg.norm(object_pos - grasp_pos, dim=-1)),
            "xy": _value(torch.linalg.norm(object_pos[:, :2] - grasp_pos[:, :2], dim=-1)),
            "height": _value(
                object_pos[:, 2]
                - (u.cfg.table_top_z + 0.5 * u.cfg.object_size)
            ),
            "gap": _value(u._fingertip_gap()),
            "driver": _value(u.robot.data.joint_pos[:, u._gripper_driver_joint_id]),
            "close_fraction": _value(u._gripper_physical_close_fraction()),
            "left_force": _value(left_force),
            "right_force": _value(right_force),
            "between": _value(u._object_between_fingertips()),
            "vertical": _value(u._vertical_alignment()),
            "wrist": abs(_value(u.robot.data.joint_pos[:, u._wrist_joint_id])),
            "object_xy_drift": _value(
                torch.linalg.norm(
                    object_pos[:, :2] - u._object_initial_pos_w[:, :2], dim=-1
                )
            ),
            "goal_xy": _value(
                torch.linalg.norm(object_pos[:, :2] - u._goal_pos_w[:, :2], dim=-1)
            ),
            "object_tilt": _value(1.0 - object_up_w[:, 2]),
            "object_ang_speed": _value(
                torch.linalg.norm(u.object.data.root_ang_vel_w.torch, dim=-1)
            ),
        }
        object_delta = (
            u.object.data.root_pos_w.torch - u._gripper_center_w()
        )[0].tolist()
        fingertip_delta = (
            u.robot.data.body_pos_w.torch[:, u._fingertip_body_ids]
            - u.object.data.root_pos_w.torch.unsqueeze(1)
        )[0]
        fingertip_pos = u.robot.data.body_pos_w.torch[:, u._fingertip_body_ids]
        jaw_midpoint = fingertip_pos.mean(dim=1)
        jaw_axis = torch.nn.functional.normalize(
            fingertip_pos[:, 1] - fingertip_pos[:, 0], dim=-1
        )
        object_lateral_error = torch.sum(
            (u.object.data.root_pos_w.torch - jaw_midpoint) * jaw_axis, dim=-1
        )
        grasp_body_pos = u.robot.data.body_pos_w.torch[:, u._grasp_body_id]
        grasp_body_quat = u.robot.data.body_quat_w.torch[:, u._grasp_body_id]
        fixed_tcp = grasp_body_pos + quat_apply(
            grasp_body_quat, u._grasp_point_offset.expand(u.num_envs, -1)
        )
        raw_body_pos = u.robot.data.body_pos_w.torch[
            :, u._grasp_calibration_source_body_id
        ]
        raw_body_quat = u.robot.data.body_quat_w.torch[
            :, u._grasp_calibration_source_body_id
        ]
        raw_tcp = raw_body_pos + quat_apply(
            raw_body_quat,
            u._grasp_calibration_source_offset.expand(u.num_envs, -1),
        )
        fixed_tcp_lateral_bias = torch.sum((fixed_tcp - jaw_midpoint) * jaw_axis, dim=-1)
        raw_tcp_lateral_bias = torch.sum((raw_tcp - jaw_midpoint) * jaw_axis, dim=-1)
        joint_pos = u.robot.data.joint_pos[0, arm_ids]
        arm_target = u._joint_targets[0, arm_ids]
        arm_torque = u.robot.data.applied_torque[0, arm_ids]
        wrist_target = _value(u._joint_targets[:, u._wrist_joint_id])
        wrist_velocity = _value(u.robot.data.joint_vel[:, u._wrist_joint_id])
        wrist_torque = _value(u.robot.data.applied_torque[:, u._wrist_joint_id])
        driver_target = _value(u._joint_targets[:, u._gripper_driver_joint_id])
        driver_velocity = _value(u.robot.data.joint_vel[:, u._gripper_driver_joint_id])
        driver_torque = _value(
            u.robot.data.applied_torque[:, u._gripper_driver_joint_id]
        )
        object_velocity = u.object.data.root_lin_vel_w.torch[0].tolist()
        print(
            f"[SANITY:{label}] step={step:04d} phase={values['phase']:.0f} "
            f"dist={1000.0 * values['distance']:.2f}mm "
            f"xy={1000.0 * values['xy']:.2f}mm "
            f"height={1000.0 * values['height']:.2f}mm "
            f"gap={1000.0 * values['gap']:.2f}mm "
            f"driver={values['driver']:+.3f} close={values['close_fraction']:.2f} "
            f"command={_value(u._gripper_command):.2f} "
            f"contact=({values['left_force']:.3f},{values['right_force']:.3f})N "
            f"between={values['between']:.0f} vertical={values['vertical']:.3f} "
            f"wrist={values['wrist']:.3f}rad target={wrist_target:+.3f} "
            f"vel={wrist_velocity:+.3f} torque={wrist_torque:+.3f}Nm "
            f"grip_target={driver_target:+.3f} grip_vel={driver_velocity:+.3f} "
            f"grip_torque={driver_torque:+.3f}Nm "
            f"drift={1000.0 * values['object_xy_drift']:.2f}mm "
            f"goal_xy={1000.0 * values['goal_xy']:.2f}mm",
            f"tilt={values['object_tilt']:.3f} omega={values['object_ang_speed']:.3f}rad/s "
            f"v={[round(float(v), 4) for v in object_velocity]}m/s "
            f"FobjL={[round(float(v), 4) for v in left_object_force_w[0].tolist()]}N "
            f"FobjR={[round(float(v), 4) for v in right_object_force_w[0].tolist()]}N "
            f"lateral={1000.0 * _value(object_lateral_error):+.2f}mm "
            f"tcp_bias=(fixed={1000.0 * _value(fixed_tcp_lateral_bias):+.2f},"
            f"raw={1000.0 * _value(raw_tcp_lateral_bias):+.2f})mm "
            f"q={[round(float(q), 3) for q in joint_pos.tolist()]} "
            f"q_target={[round(float(q), 3) for q in arm_target.tolist()]} "
            f"tau={[round(float(t), 2) for t in arm_torque.tolist()]} "
            f"L={dict(zip(u.left_finger_contact.body_names, [round(v, 3) for v in left_by_body]))} "
            f"R={dict(zip(u.right_finger_contact.body_names, [round(v, 3) for v in right_by_body]))} "
            f"delta_mm={[round(1000.0 * v, 2) for v in object_delta]} "
            f"tip_delta_mm={[[round(1000.0 * float(v), 2) for v in row] for row in fingertip_delta.tolist()]} "
            f"object_quat={[round(v, 4) for v in u.object.data.root_quat_w.torch[0].tolist()]}",
            flush=True,
        )
        return values

    def joint_action(
        desired_joint_pos: torch.Tensor,
        gripper_action: float,
        max_action: float,
    ) -> torch.Tensor:
        """Track an FK-verified waypoint through the environment action path."""
        action = torch.zeros((1, u.cfg.action_space), device=u.device)
        action[:, :4] = torch.clamp(
            (desired_joint_pos - u._joint_targets[:, arm_ids])
            / u._joint_action_scales[:4],
            -max_action,
            max_action,
        )
        action[:, 4] = torch.clamp(
            (0.0 - u._joint_targets[:, u._wrist_joint_id])
            / u._joint_action_scales[4],
            -max_action,
            max_action,
        )
        action[:, 5] = gripper_action
        return action

    def cartesian_action(
        target_w: torch.Tensor,
        gripper_action: float,
        max_action: float,
        target_approach_angle: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Local task-space correction while retaining the negative-joint3 branch."""
        body_pos_w = u.robot.data.body_pos_w.torch[:, u._grasp_body_id]
        body_quat_w = u.robot.data.body_quat_w.torch[:, u._grasp_body_id]
        offset_w = quat_apply(
            body_quat_w, u._grasp_point_offset.expand(u.num_envs, -1)
        )
        point_pos_w = body_pos_w + offset_w
        jacobian_w = u.robot.data.body_link_jacobian_w.torch[
            :, jacobian_body, :, jacobian_columns
        ].clone()
        jacobian_w[:, :3, :] -= torch.bmm(
            skew_symmetric_matrix(offset_w), jacobian_w[:, 3:, :]
        )
        # Position-only control is used for approach/centering.  During lift we
        # additionally constrain the single physically controllable planar
        # pitch DOF.  This is a square 4-task/4-joint solve; unlike a full 6-D
        # pose command it does not over-constrain the DOFBOT arm.
        task_jacobian = jacobian_w[:, :3, :]
        task_error = target_w - point_pos_w
        if target_approach_angle is not None:
            local_z = torch.zeros((u.num_envs, 3), device=u.device)
            local_z[:, 2] = 1.0
            approach_w = quat_apply(body_quat_w, local_z)
            current_angle = torch.atan2(-approach_w[:, 1], approach_w[:, 2])
            angle_error = target_approach_angle - current_angle
            angle_error = torch.atan2(torch.sin(angle_error), torch.cos(angle_error))
            orientation_length_scale = 0.15
            task_jacobian = torch.cat(
                (
                    task_jacobian,
                    orientation_length_scale * jacobian_w[:, 3:4, :],
                ),
                dim=1,
            )
            task_error = torch.cat(
                (task_error, orientation_length_scale * angle_error.unsqueeze(-1)),
                dim=1,
            )
        jacobian_t = task_jacobian.transpose(1, 2)
        damped = (0.02**2) * torch.eye(
            task_jacobian.shape[1], device=u.device
        ).unsqueeze(0)
        delta_q = (
            jacobian_t
            @ torch.linalg.solve(
                task_jacobian @ jacobian_t + damped,
                task_error.unsqueeze(-1),
            )
        ).squeeze(-1)
        desired_q = u.robot.data.joint_pos.torch[:, arm_ids] + delta_q
        desired_q[:, 0] = torch.clamp(desired_q[:, 0], -0.0873, 0.0873)
        desired_q = torch.clamp(
            desired_q, u._controlled_lower[:4], u._controlled_upper[:4]
        )
        return joint_action(desired_q, gripper_action, max_action)

    def step(action: torch.Tensor, label: str, local_step: int) -> tuple[dict[str, float], bool]:
        nonlocal last_info, completed_steps
        _, _, terminated, truncated, last_info = env.step(action)
        completed_steps += 1
        values = snapshot(label, local_step) if (
            (
                label == "CLOSE"
                and (local_step < 20 or (local_step + 1) % 10 == 0)
            )
            or (
                label == "LIFT"
                and (local_step < 20 or (local_step + 1) % 30 == 0)
            )
            or local_step == 0
            or (local_step + 1) % args_cli.log_interval == 0
        ) else {}
        ended = bool(terminated.item() or truncated.item())
        if ended:
            success_held = _log_value(last_info, "Conditions/success_held") >= 0.5
            if not success_held:
                object_local = u.object.data.root_pos_w.torch - u._base_pos_w()
                object_drift = torch.linalg.norm(
                    u.object.data.root_pos_w.torch[:, :2]
                    - u._object_initial_pos_w[:, :2], dim=-1
                )
                grasp_clearance = (
                    u._gripper_center_w()[:, 2]
                    - u._base_pos_w()[:, 2]
                    - u.cfg.table_top_z
                )
                raise SanityFailure(
                    f"{label.upper()}_TERMINATED_WITHOUT_SUCCESS "
                    f"terminated={bool(terminated.item())} truncated={bool(truncated.item())} "
                    f"phase={int(u._task_phase.item())} "
                    f"failed={bool(u._task_failed.item())} "
                    f"grasp_loss_steps={int(u._grasp_loss_steps.item())} "
                    f"fallen={bool((object_local[:, 2] < -0.01).item())} "
                    f"drift={1000.0 * _value(object_drift):.2f}mm "
                    f"clearance={1000.0 * _value(grasp_clearance):.2f}mm "
                    f"limit_violation={bool(u._gripper_limit_violation.item())}"
                )
        return values, ended

    def assert_pose_safety(label: str, min_vertical: float | None = None) -> None:
        vertical = _value(u._vertical_alignment())
        wrist = abs(_value(u.robot.data.joint_pos[:, u._wrist_joint_id]))
        required_vertical = (
            u.cfg.vertical_alignment_threshold
            if min_vertical is None
            else min_vertical
        )
        if vertical <= required_vertical:
            raise SanityFailure(
                f"{label}_APPROACH_NOT_VERTICAL alignment={vertical:.4f} "
                f"required>{required_vertical:.4f}"
            )
        if wrist >= 0.10:
            raise SanityFailure(f"{label}_WRIST_NOT_ZERO abs_wrist={wrist:.4f}rad")

    try:
        env.reset()
        initial_driven = u.robot.data.joint_pos[0, u._controlled_joint_ids]
        initial_driver = u.robot.data.joint_pos[0, u._gripper_driver_joint_id]
        if not bool(torch.allclose(initial_driven, torch.zeros_like(initial_driven), atol=1.0e-6)):
            raise SanityFailure(f"RESET_ARM_NOT_ZERO {initial_driven.tolist()}")
        if abs(_value(initial_driver)) > 1.0e-6:
            raise SanityFailure(f"RESET_GRIPPER_NOT_ZERO {_value(initial_driver):+.8f}")

        for settle_step in range(12):
            step(zero_action, "SETTLE", settle_step)
        initial_object_pos = u.object.data.root_pos_w.torch.clone()
        object_target = initial_object_pos.clone()
        grasp_target = object_target.clone()
        grasp_target[:, 2] += u.cfg.grasp_center_height_offset
        pregrasp_target = grasp_target.clone()
        pregrasp_target[:, 2] += u.cfg.pregrasp_height
        print(
            f"[SANITY] fixed_object={initial_object_pos[0].tolist()} "
            f"arm_ids={arm_ids}",
            flush=True,
        )

        best_pregrasp_error = float("inf")
        stagnant_steps = 0
        pregrasp_passed = False
        for local_step in range(420):
            action = joint_action(pregrasp_joint_target, -1.0, max_action=0.12)
            step(action, "PREGRASP", local_step)
            grasp_pos = u._gripper_center_w()
            error = _value(torch.linalg.norm(pregrasp_target - grasp_pos, dim=-1))
            if error < best_pregrasp_error - 1.0e-4:
                best_pregrasp_error = error
                stagnant_steps = 0
            else:
                stagnant_steps += 1
            wrist = abs(_value(u.robot.data.joint_pos[:, u._wrist_joint_id]))
            if wrist >= 0.10:
                raise SanityFailure(f"PREGRASP_WRIST_NOT_ZERO abs_wrist={wrist:.4f}rad")
            object_drift = _value(
                torch.linalg.norm(
                    u.object.data.root_pos_w.torch[:, :2]
                    - initial_object_pos[:, :2],
                    dim=-1,
                )
            )
            if object_drift >= 0.003:
                raise SanityFailure(
                    f"CUBE_DISTURBED_ABOVE_PREGRASP drift={1000.0 * object_drift:.2f}mm"
                )
            xy_error = _value(
                torch.linalg.norm(grasp_pos[:, :2] - object_target[:, :2], dim=-1)
            )
            height_error = abs(_value(grasp_pos[:, 2] - pregrasp_target[:, 2]))
            if (
                int(u._task_phase.item()) == 1
                and error < 0.008
                and xy_error < u.cfg.pregrasp_xy_tolerance
                and height_error < u.cfg.pregrasp_height_tolerance
                and _value(u._vertical_alignment())
                > u.cfg.pregrasp_vertical_alignment_threshold
                and abs(_value(u.robot.data.joint_pos[:, u._wrist_joint_id])) < 0.05
            ):
                pregrasp_passed = True
                print(
                    f"[SANITY:PASS] PREGRASP error={1000.0 * error:.2f}mm",
                    flush=True,
                )
                break
            if local_step > 100 and stagnant_steps >= 100:
                raise SanityFailure(
                    f"WAYPOINT_STALLED_PREGRASP best_error={1000.0 * best_pregrasp_error:.2f}mm"
                )
        if not pregrasp_passed:
            raise SanityFailure(
                f"PREGRASP_NOT_REACHED best_error={1000.0 * best_pregrasp_error:.2f}mm"
            )

        descent_passed = False
        for local_step in range(360):
            action = joint_action(grasp_joint_target, -1.0, max_action=0.08)
            step(action, "DESCEND", local_step)
            # The high pregrasp pose cannot kinematically reach the stricter
            # close orientation.  Enforce the achievable high-pose threshold
            # while the descending path continuously improves alignment; the
            # CENTER/CLOSE phases below still require the strict grasp value.
            assert_pose_safety(
                "DESCEND", u.cfg.pregrasp_vertical_alignment_threshold
            )
            safe_grasp_target = u._grasp_target_w()
            reach_distance = _value(
                torch.linalg.norm(safe_grasp_target - u._gripper_center_w(), dim=-1)
            )
            object_drift = _value(
                torch.linalg.norm(
                    u.object.data.root_pos_w.torch[:, :2]
                    - initial_object_pos[:, :2],
                    dim=-1,
                )
            )
            if object_drift >= 0.003:
                raise SanityFailure(
                    f"CUBE_DISTURBED_BEFORE_CLOSE drift={1000.0 * object_drift:.2f}mm"
                )
            if (
                int(u._task_phase.item()) == 1
                and reach_distance < u.cfg.gripper_close_tolerance
                and abs(_value(u.robot.data.joint_pos[:, u._wrist_joint_id])) < 0.05
            ):
                descent_passed = True
                print(
                    f"[SANITY:PASS] DESCEND distance={1000.0 * reach_distance:.2f}mm "
                    f"cube_drift={1000.0 * object_drift:.2f}mm",
                    flush=True,
                )
                break
        if not descent_passed:
            raise SanityFailure("DESCENT_TARGET_NOT_REACHED")

        # The coarse FK waypoint intentionally stops outside contact.  Center
        # the *open* jaw in XY before issuing any close command.  Runtime
        # contact isolation showed that placing the jaw midpoint exactly at the
        # resting cube center makes the lower finger geometry hit the table.
        # Keep the empirically collision-free jaw center 4.5 mm above the cube.
        fine_center_target = u._grasp_target_w().clone()
        fine_center_passed = False
        best_fine_center_distance = float("inf")
        for local_step in range(240):
            fine_center_target[:] = u._grasp_target_w()
            action = cartesian_action(fine_center_target, -1.0, max_action=0.03)
            step(action, "CENTER", local_step)
            assert_pose_safety("CENTER")
            center_delta = fine_center_target - u._gripper_center_w()
            center_distance = _value(torch.linalg.norm(center_delta, dim=-1))
            center_xy = _value(torch.linalg.norm(center_delta[:, :2], dim=-1))
            best_fine_center_distance = min(best_fine_center_distance, center_distance)
            object_drift = _value(
                torch.linalg.norm(
                    u.object.data.root_pos_w.torch[:, :2]
                    - initial_object_pos[:, :2],
                    dim=-1,
                )
            )
            if object_drift >= 0.003:
                raise SanityFailure(
                    f"CUBE_DISTURBED_DURING_FINE_CENTER "
                    f"drift={1000.0 * object_drift:.2f}mm"
                )
            if center_distance < 0.0020 and center_xy < 0.0015:
                fine_center_passed = True
                print(
                    f"[SANITY:PASS] FINE_CENTER "
                    f"distance={1000.0 * center_distance:.2f}mm "
                    f"xy={1000.0 * center_xy:.2f}mm",
                    flush=True,
                )
                break
        if not fine_center_passed:
            raise SanityFailure(
                f"FINE_CENTER_NOT_REACHED "
                f"best={1000.0 * best_fine_center_distance:.2f}mm"
            )

        close_hold = 0
        close_passed = False
        captured_joint_target: torch.Tensor | None = None
        for local_step in range(180):
            if int(u._task_phase.item()) >= 2:
                if captured_joint_target is None:
                    captured_joint_target = u.robot.data.joint_pos.torch[:, arm_ids].clone()
                # Once the physics gates confirm capture, stop chasing tiny
                # contact-induced object motion and hold the arm steady while
                # bilateral contact settles.
                action = joint_action(captured_joint_target, 1.0, max_action=0.05)
            else:
                action = cartesian_action(
                    fine_center_target, 1.0, max_action=0.05
                )
            _, ended = step(action, "CLOSE", local_step)
            if ended:
                close_passed = True
                break
            assert_pose_safety("CLOSE")
            left_force, right_force = u._finger_contact_forces()
            gap = _value(u._fingertip_gap())
            valid = (
                int(u._task_phase.item()) >= 2
                and _value(u._gripper_command) > 0.65
                and _value(u.robot.data.joint_pos[:, u._gripper_driver_joint_id])
                >= u.cfg.gripper_driver_grasp_min_position
                and u.cfg.gripper_grasp_min_gap <= gap <= u.cfg.gripper_grasp_max_gap
                and _value(left_force) >= u.cfg.finger_contact_force_threshold
                and _value(right_force) >= u.cfg.finger_contact_force_threshold
                and bool(u._object_between_fingertips().item())
                and _value(u._gripper_physical_close_fraction()) >= 1.0
            )
            close_hold = close_hold + 1 if valid else 0
            if close_hold >= 10:
                close_passed = True
                print(
                    f"[SANITY:PASS] CLOSE bilateral_object_contact held={close_hold}",
                    flush=True,
                )
                break
        if not close_passed:
            raise SanityFailure("BILATERAL_GRASP_NOT_ESTABLISHED")

        # Lift from the *captured* pose along a jerk-free vertical Cartesian
        # path.  A direct jump toward a remote joint-space seed created an
        # avoidable 2.26-rad/s angular impulse on the cube.
        lift_start_grasp = u._gripper_center_w().clone()
        lift_target = lift_start_grasp.clone()
        lift_distance = 0.075
        lift_trajectory_steps = 900
        success = False
        lift_passed = False
        lift_hold = 0
        max_object_ang_speed = 0.0
        max_object_ang_speed_step = -1
        max_release_ang_speed = 0.0
        max_release_lin_speed = 0.0
        unstable_hold = 0

        def assert_carry_safety(label: str, local_step: int, tilt_limit: float) -> float:
            nonlocal max_object_ang_speed, max_object_ang_speed_step, unstable_hold
            assert_pose_safety(label)
            separation = _value(
                torch.linalg.norm(
                    u.object.data.root_pos_w.torch - u._gripper_center_w(), dim=-1
                )
            )
            if separation >= u.cfg.grasp_separation_tolerance:
                raise SanityFailure(
                    f"OBJECT_SEPARATED_DURING_{label} "
                    f"distance={1000.0 * separation:.2f}mm"
                )
            object_local_z = torch.zeros((1, 3), device=u.device)
            object_local_z[:, 2] = 1.0
            object_up_w = quat_apply(u.object.data.root_quat_w.torch, object_local_z)
            object_tilt = _value(
                1.0 - torch.sum(object_up_w * u._object_initial_up_w, dim=-1)
            )
            object_ang_speed = _value(
                torch.linalg.norm(u.object.data.root_ang_vel_w.torch, dim=-1)
            )
            if object_ang_speed > max_object_ang_speed:
                max_object_ang_speed = object_ang_speed
                max_object_ang_speed_step = completed_steps
            if object_tilt > tilt_limit:
                raise SanityFailure(
                    f"OBJECT_TILTED_DURING_{label} tilt={object_tilt:.4f} "
                    f"limit={tilt_limit:.4f} local_step={local_step}"
                )
            unstable_hold = unstable_hold + 1 if object_ang_speed > 2.0 else 0
            if unstable_hold >= 5:
                raise SanityFailure(
                    f"OBJECT_UNSTABLE_DURING_{label} "
                    f"omega={object_ang_speed:.3f}rad/s "
                    f"max_omega={max_object_ang_speed:.3f}rad/s "
                    f"held={unstable_hold} local_step={local_step}"
                )
            return object_ang_speed

        lift_step_limit = 700 if args_cli.full_pick_place else 1200
        for local_step in range(lift_step_limit):
            linear_fraction = min((local_step + 1) / lift_trajectory_steps, 1.0)
            smooth_fraction = linear_fraction * linear_fraction * (3.0 - 2.0 * linear_fraction)
            lift_target[:, :2] = lift_start_grasp[:, :2]
            lift_target[:, 2] = lift_start_grasp[:, 2] + lift_distance * smooth_fraction
            action = cartesian_action(lift_target, 1.0, max_action=0.02)
            _, ended = step(action, "LIFT", local_step)
            # Gym auto-resets a terminated DirectRLEnv before returning the
            # observation.  Read the terminal success log before evaluating
            # any post-step pose invariant against that reset state.
            if ended:
                success = _log_value(last_info, "Conditions/success_held") >= 0.5
                break
            assert_carry_safety("LIFT", local_step, tilt_limit=0.05)
            if args_cli.full_pick_place:
                object_height = _value(
                    u.object.data.root_pos_w.torch[:, 2]
                    - (u.cfg.table_top_z + 0.5 * u.cfg.object_size)
                )
                # Do not stop at the environment's minimum carry clearance.
                # Command an additional 10 mm before settling so a transient
                # threshold crossing cannot masquerade as a sustainable lift.
                lift_ready = (
                    int(u._task_phase.item()) >= 3
                    and object_height > u.cfg.lift_height + 0.010
                    and bool(u._object_between_fingertips().item())
                )
                lift_hold = lift_hold + 1 if lift_ready else 0
                if lift_hold >= 8:
                    lift_passed = True
                    print(
                        f"[SANITY:PASS] LIFT height={1000.0 * object_height:.2f}mm "
                        f"phase={int(u._task_phase.item())} held={lift_hold}",
                        flush=True,
                    )
                    break

        if not args_cli.full_pick_place and not success:
            raise SanityFailure("LIFT_SUCCESS_HOLD_NOT_REACHED")

        if args_cli.full_pick_place and not lift_passed:
            raise SanityFailure("LIFT_PHASE3_NOT_REACHED")

        if not args_cli.full_pick_place:
            print(
                f"[SANITY:PASS] GRASP_AND_LIFT completed_steps={completed_steps} "
                f"success_hold={_log_value(last_info, 'Conditions/success_held'):.0f} "
                f"max_omega={max_object_ang_speed:.3f}rad/s "
                f"max_omega_step={max_object_ang_speed_step}",
                flush=True,
            )
            return

        # Crossing the lift threshold can coincide with a short contact impulse
        # (the first full run measured 3.22 rad/s).  Do not begin lateral motion
        # until the held cube has actually settled; a phase integer alone is not
        # evidence of stable lift completion.
        lift_settle_joint_target = u.robot.data.joint_pos.torch[:, arm_ids].clone()
        lift_settle_hold = 0
        unstable_hold = 0
        for local_step in range(480):
            action = joint_action(
                lift_settle_joint_target, 1.0, max_action=0.01
            )
            _, ended = step(action, "LIFT_SETTLE", local_step)
            if ended:
                raise SanityFailure("UNEXPECTED_TERMINATION_DURING_LIFT_SETTLE")
            object_ang_speed = assert_carry_safety(
                "LIFT_SETTLE", local_step, tilt_limit=0.05
            )
            object_speed = _value(
                torch.linalg.norm(u.object.data.root_lin_vel_w.torch, dim=-1)
            )
            object_height = _value(
                u.object.data.root_pos_w.torch[:, 2]
                - (u.cfg.table_top_z + 0.5 * u.cfg.object_size)
            )
            stable_lift = (
                object_height > u.cfg.lift_height
                and object_ang_speed < 0.50
                and object_speed < 0.030
                and bool(u._object_between_fingertips().item())
            )
            lift_settle_hold = lift_settle_hold + 1 if stable_lift else 0
            if lift_settle_hold >= 20:
                print(
                    f"[SANITY:PASS] LIFT_SETTLE height={1000.0 * object_height:.2f}mm "
                    f"omega={object_ang_speed:.3f}rad/s speed={object_speed:.3f}m/s",
                    flush=True,
                )
                break
        if lift_settle_hold < 20:
            raise SanityFailure("LIFT_DID_NOT_SETTLE_BEFORE_TRANSPORT")

        # Move laterally only after lift is physically confirmed.  A single
        # start-to-finish joint interpolation dipped to 1.8 mm above the table
        # and saturated joint2/joint3 even though both endpoint FK poses were
        # valid.  These six continuous waypoints were found by a GPU FK search.
        # Their commanded TCP height is 77--85 mm: the measured 15 mm dynamic
        # tracking offset then leaves the carried cube at its 15 mm clearance
        # instead of allowing the fingertips to reach the table.
        transport_start_joint = u._joint_targets[:, arm_ids].clone()
        transport_final = u._goal_pos_w.clone()
        # Stop 6 mm on the base-facing side of the marker.  This is well inside
        # the physical 20-mm place tolerance and the batched FK sweep showed it
        # preserves an upright carry at the low final placement height.
        goal_radius = torch.linalg.norm(transport_final[:, :2], dim=-1, keepdim=True)
        inward_xy = transport_final[:, :2] / torch.clamp(goal_radius, min=1.0e-6)
        transport_final[:, :2] -= 0.006 * inward_xy
        target_yaw = -torch.atan2(transport_final[:, 0], transport_final[:, 1])
        transport_joint_targets = torch.tensor(
            [
                [-0.00612399, 0.45399830, -1.46824086, -1.57079637],
                [-0.00583472, 0.27263775, -1.34727573, -1.54318631],
                [-0.00575087, 0.08220811, -1.18038702, -1.54920959],
                [-0.00538859, -0.04132476, -1.08928907, -1.51298249],
                [-0.00495655, -0.10750190, -1.06922424, -1.44557965],
                [-0.00517364, -0.30802634, -0.77982640, -1.57079637],
            ],
            device=u.device,
        )
        transport_joint_targets[:, 0] += target_yaw[0]
        transport_segment_steps = 160
        transport_hold = 0
        transport_passed = False
        unstable_hold = 0
        local_step = 0
        for segment_target_row in transport_joint_targets:
            segment_target = segment_target_row.unsqueeze(0)
            for segment_step in range(transport_segment_steps):
                linear_fraction = (segment_step + 1) / transport_segment_steps
                smooth_fraction = linear_fraction * linear_fraction * (
                    3.0 - 2.0 * linear_fraction
                )
                waypoint_joint = transport_start_joint + (
                    segment_target - transport_start_joint
                ) * smooth_fraction
                action = joint_action(
                    waypoint_joint,
                    1.0,
                    max_action=0.025,
                )
                _, ended = step(action, "TRANSPORT", local_step)
                if ended:
                    raise SanityFailure("UNEXPECTED_TERMINATION_DURING_TRANSPORT")
                assert_carry_safety("TRANSPORT", local_step, tilt_limit=0.08)
                object_pos = u.object.data.root_pos_w.torch
                goal_pos = u._goal_pos_w
                goal_xy = _value(
                    torch.linalg.norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
                )
                height_error = abs(
                    _value(object_pos[:, 2] - goal_pos[:, 2])
                    - u.cfg.transport_clearance
                )
                # Require the same strict physical transport gate as the RL
                # task.  Phase alone previously allowed lowering 24.5 mm early.
                transport_ready = (
                    int(u._task_phase.item()) == 4
                    and goal_xy < u.cfg.transport_tolerance
                    and height_error < u.cfg.transport_height_tolerance
                    and bool(u._object_between_fingertips().item())
                )
                transport_hold = transport_hold + 1 if transport_ready else 0
                if transport_hold >= 8:
                    transport_passed = True
                    print(
                        f"[SANITY:PASS] TRANSPORT xy={1000.0 * goal_xy:.2f}mm "
                        f"height_error={1000.0 * height_error:.2f}mm "
                        f"held={transport_hold}",
                        flush=True,
                    )
                    break
                local_step += 1
            if transport_passed:
                break
            transport_start_joint = segment_target

        # Give the final physical pose time to settle without relaxing any
        # phase, capture, height, or XY gate.
        if not transport_passed:
            final_target = transport_joint_targets[-1:].clone()
            for _ in range(240):
                action = joint_action(final_target, 1.0, max_action=0.015)
                _, ended = step(action, "TRANSPORT", local_step)
                if ended:
                    raise SanityFailure("UNEXPECTED_TERMINATION_DURING_TRANSPORT")
                assert_carry_safety("TRANSPORT", local_step, tilt_limit=0.08)
                object_pos = u.object.data.root_pos_w.torch
                goal_pos = u._goal_pos_w
                goal_xy = _value(
                    torch.linalg.norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
                )
                height_error = abs(
                    _value(object_pos[:, 2] - goal_pos[:, 2])
                    - u.cfg.transport_clearance
                )
                transport_ready = (
                    int(u._task_phase.item()) == 4
                    and goal_xy < u.cfg.transport_tolerance
                    and height_error < u.cfg.transport_height_tolerance
                    and bool(u._object_between_fingertips().item())
                )
                transport_hold = transport_hold + 1 if transport_ready else 0
                if transport_hold >= 8:
                    transport_passed = True
                    print(
                        f"[SANITY:PASS] TRANSPORT xy={1000.0 * goal_xy:.2f}mm "
                        f"height_error={1000.0 * height_error:.2f}mm "
                        f"held={transport_hold}",
                        flush=True,
                    )
                    break
                local_step += 1
        if not transport_passed:
            raise SanityFailure("TRANSPORT_PHASE4_NOT_REACHED")

        # Lower vertically until the cube center is at the goal's authored
        # table height.  Keep the grasp closed until pose and velocity have both
        # remained inside the place gate.
        lower_start_joint = u._joint_targets[:, arm_ids].clone()
        lower_final = u._goal_pos_w.clone()
        lower_yaw = -torch.atan2(lower_final[:, 0], lower_final[:, 1])
        # Loaded replay of the nominal 190-mm FK pose settled 16.5 mm short.
        # This 207-mm / 47-mm TCP solution (0.50-mm FK error, 0.914 vertical)
        # compensates that measured dynamic offset while retaining joint3<0.
        lower_joint_target = torch.tensor(
            [[-0.00633477, -0.73609257, -0.44272771, -1.54824066]],
            device=u.device,
        )
        lower_joint_target[:, 0] = lower_yaw - 0.00633477
        lower_trajectory_steps = 520
        place_hold = 0
        place_ready = False
        unstable_hold = 0
        for local_step in range(760):
            linear_fraction = min((local_step + 1) / lower_trajectory_steps, 1.0)
            smooth_fraction = linear_fraction * linear_fraction * (3.0 - 2.0 * linear_fraction)
            waypoint_joint = lower_start_joint + (
                lower_joint_target - lower_start_joint
            ) * smooth_fraction
            action = joint_action(
                waypoint_joint,
                1.0,
                max_action=0.02,
            )
            _, ended = step(action, "LOWER", local_step)
            if ended:
                raise SanityFailure("UNEXPECTED_TERMINATION_DURING_LOWER")
            assert_carry_safety("LOWER", local_step, tilt_limit=0.10)
            physical_place = bool(u._place_pose_mask().item())
            place_hold = place_hold + 1 if physical_place else 0
            if place_hold >= 12:
                place_ready = True
                goal_error = _value(
                    torch.linalg.norm(
                        u.object.data.root_pos_w.torch - u._goal_pos_w, dim=-1
                    )
                )
                print(
                    f"[SANITY:PASS] LOWER_AND_SETTLE "
                    f"goal_error={1000.0 * goal_error:.2f}mm held={place_hold}",
                    flush=True,
                )
                break
        if not place_ready:
            raise SanityFailure("PLACE_POSE_DID_NOT_SETTLE")

        # Open by commanding the independent right driver; the left jaw remains
        # purely PhysX-mimic driven.  Hold the arm fixed until measured driver,
        # gap, cube pose and cube speed satisfy the held success gate.  The env
        # keeps the jaw closed until its place-pose authorization is latched.
        release_joint_target = u.robot.data.joint_pos.torch[:, arm_ids].clone()
        success = False
        release_unstable_hold = 0
        for local_step in range(300):
            action = joint_action(release_joint_target, -1.0, max_action=0.01)
            _, ended = step(action, "OPEN", local_step)
            # Terminal tensors have already auto-reset, so use the metrics copied
            # into info for both terminal and non-terminal release samples.
            release_ang_speed = _log_value(last_info, "Metrics/object_angular_speed")
            release_lin_speed = _log_value(last_info, "Metrics/object_speed")
            release_tilt = _log_value(last_info, "Metrics/object_tilt")
            max_release_ang_speed = max(max_release_ang_speed, release_ang_speed)
            max_release_lin_speed = max(max_release_lin_speed, release_lin_speed)
            release_unstable_hold = (
                release_unstable_hold + 1 if release_ang_speed > 2.0 else 0
            )
            if release_unstable_hold >= 5:
                raise SanityFailure(
                    "OBJECT_UNSTABLE_DURING_OPEN "
                    f"omega={release_ang_speed:.3f}rad/s held={release_unstable_hold}"
                )
            if release_tilt > 0.10:
                raise SanityFailure(
                    f"OBJECT_TILTED_DURING_OPEN tilt={release_tilt:.4f}"
                )
            if ended:
                success = _log_value(last_info, "Conditions/success_held") >= 0.5
                break
            if bool(u._task_failed.item()):
                raise SanityFailure("TASK_FAILED_DURING_MEASURED_RELEASE")
            wrist = abs(_value(u.robot.data.joint_pos[:, u._wrist_joint_id]))
            if wrist >= 0.10:
                raise SanityFailure(f"OPEN_WRIST_NOT_ZERO abs_wrist={wrist:.4f}rad")
        if not success:
            raise SanityFailure(
                "PICK_PLACE_SUCCESS_HOLD_NOT_REACHED "
                f"released={_log_value(last_info, 'Conditions/gripper_released'):.0f} "
                f"place={_log_value(last_info, 'Conditions/place_pose_ready'):.0f}"
            )
        if max_release_ang_speed > 2.0 or max_release_lin_speed > 0.10:
            raise SanityFailure(
                "RELEASE_IMPACT_TOO_LARGE "
                f"max_omega={max_release_ang_speed:.3f}rad/s "
                f"max_speed={max_release_lin_speed:.3f}m/s"
            )

        print(
            f"[SANITY:PASS] PICK_PLACE completed_steps={completed_steps} "
            f"success_hold={_log_value(last_info, 'Conditions/success_held'):.0f} "
            f"max_omega={max(max_object_ang_speed, max_release_ang_speed):.3f}rad/s "
            f"carry_max_omega_step={max_object_ang_speed_step} "
            f"release_max_omega={max_release_ang_speed:.3f}rad/s "
            f"release_max_speed={max_release_lin_speed:.3f}m/s",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SanityFailure as exc:
        exit_code = 2
        print(f"[SANITY:FAIL] {exc}", flush=True)
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
