from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for source_dir in ("isaaclab", "isaaclab_rl", "isaaclab_tasks"):
    source_path = PROJECT_ROOT / "source" / source_dir
    if source_path.is_dir() and str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test the DOFBOT reach environment.")
parser.add_argument("--task", type=str, default=None, help="Gym task id. Defaults to the joint-delta reach task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=4, help="Number of random actions to step.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--actuator_test",
    action="store_true",
    help="Command each DOFBOT_V2 driven joint separately and verify measured motion plus finger mimic.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not 1 <= args_cli.num_envs <= 2048:
    parser.error(f"--num_envs must be in the safe range 1..2048 (got {args_cli.num_envs})")
if args_cli.num_steps < 1:
    parser.error(f"--num_steps must be positive (got {args_cli.num_steps})")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from dofbot_rl.tasks import (
    ENV_ID,
    IK_ENV_ID,
    PICK_PLACE_ENV_ID,
    PICK_PLACE_LIFT_ENV_ID,
    PICK_PLACE_REACH_ENV_ID,
)
from dofbot_rl.tasks.dofbot_pick_place_cfg import (
    DofbotPickPlaceEnvCfg,
    DofbotPickPlaceLiftEnvCfg,
    DofbotPickPlaceReachEnvCfg,
)
from dofbot_rl.tasks.dofbot_reach_cfg import DofbotReachEnvCfg, DofbotReachIKEnvCfg
import dofbot_rl.tasks  # noqa: F401


def run_pick_place_actuator_test(env) -> None:
    """Prove that the five robot joints and right-finger driver respond independently."""
    unwrapped = env.unwrapped
    expected_lower = torch.full((5,), -0.5 * torch.pi, device=unwrapped.device)
    expected_upper = torch.full((5,), 0.5 * torch.pi, device=unwrapped.device)
    if not torch.allclose(unwrapped._controlled_lower, expected_lower, atol=1.0e-5) or not torch.allclose(
        unwrapped._controlled_upper, expected_upper, atol=1.0e-5
    ):
        raise RuntimeError(
            "joint1 through wrist must all use the physical -90 to +90 degree range"
        )
    print("[JOINT LIMIT] joint1, joint2, joint3, joint4, wrist: -90.0 to +90.0 deg", flush=True)

    driver_limits = unwrapped.robot.data.soft_joint_pos_limits[0, unwrapped._gripper_driver_joint_id]
    expected_driver_limits = torch.deg2rad(
        torch.tensor([-57.0, 33.0], device=unwrapped.device)
    )
    if not torch.allclose(driver_limits, expected_driver_limits, atol=1.0e-5):
        raise RuntimeError(
            f"Finger_Right_01 range must be -57..+33 deg, got {driver_limits.tolist()}"
        )
    print("[JOINT LIMIT] Finger_Right_01 driver: -57.0 to +33.0 deg (90.0 deg total)", flush=True)

    # Vertical alignment alone cannot detect a twist around the wrist Z axis.
    # Prove that a +/-90-degree wrist cannot advance even an otherwise perfect
    # pre-grasp sample, then restore the required all-zero reset state.
    wrist_test_pos = unwrapped.robot.data.joint_pos.clone()
    wrist_test_pos[:, unwrapped._wrist_joint_id] = 0.5 * torch.pi
    unwrapped.robot.write_joint_state_to_sim(
        wrist_test_pos, torch.zeros_like(wrist_test_pos)
    )
    unwrapped._joint_targets[:] = wrist_test_pos
    unwrapped._task_phase.zero_()
    unwrapped._phase_gate_hold_count.zero_()
    zeros = torch.zeros((unwrapped.num_envs,), device=unwrapped.device)
    ones = torch.ones_like(zeros)
    unwrapped._update_task_phase(
        zeros,
        zeros,
        zeros,
        ones,
        ones,
        ones,
        ones,
        zeros,
        ones,
        zeros.bool(),
    )
    if not bool(torch.all(unwrapped._task_phase == 0)):
        raise RuntimeError("A 90-degree wrist twist advanced the pre-grasp phase")
    print("[WRIST GATE] +90 deg cannot advance pre-grasp: ok", flush=True)
    env.reset()

    unwrapped._task_phase.zero_()
    unwrapped._gripper_command.zero_()
    unwrapped._update_task_phase(
        zeros,
        zeros,
        zeros,
        ones,
        ones,
        ones,
        ones,
        zeros,
        ones,
        zeros.bool(),
    )
    unwrapped._gripper_command.fill_(0.8)
    unwrapped._update_task_phase(
        ones,
        ones,
        ones,
        zeros,
        ones,
        ones,
        ones,
        zeros,
        ones,
        zeros.bool(),
    )
    if not bool(torch.all(unwrapped._task_phase == 1)):
        raise RuntimeError("A gripper command without measured finger travel advanced the grasp phase")
    unwrapped._update_task_phase(
        ones,
        ones,
        ones,
        zeros,
        ones,
        ones,
        ones,
        zeros,
        ones,
        ones.bool(),
    )
    unwrapped._update_task_phase(
        ones, ones, ones, zeros, ones, ones, ones,
        ones * (unwrapped.cfg.lift_height + 0.01), ones, ones.bool(),
    )
    unwrapped._update_task_phase(
        ones, ones, ones, zeros, zeros, zeros, zeros,
        ones * (unwrapped.cfg.lift_height + 0.01), ones, ones.bool(),
    )
    expected_phase = {"reach": 2, "lift": 3, "pick_place": 4}[unwrapped.cfg.curriculum_stage]
    if not bool(torch.all(unwrapped._task_phase == expected_phase)):
        raise RuntimeError(f"Pick–Place phase transition failed: {unwrapped._task_phase}")
    print(
        f"[PHASE] stage={unwrapped.cfg.curriculum_stage} stops at phase {expected_phase}: ok",
        flush=True,
    )

    driven_names = [*unwrapped.cfg.arm_joint_names, unwrapped.cfg.wrist_joint_name]
    driven_ids = unwrapped._controlled_joint_ids

    for action_index, (joint_name, joint_id) in enumerate(zip(driven_names, driven_ids, strict=True)):
        env.reset()
        before_all = unwrapped.robot.data.joint_pos[:, joint_id].clone()
        action = torch.zeros((unwrapped.num_envs, unwrapped.cfg.action_space), device=unwrapped.device)
        action[:, action_index] = 1.0
        for _ in range(10):
            env.step(action)
        after_all = unwrapped.robot.data.joint_pos[:, joint_id]
        delta_all = after_all - before_all
        min_delta = float(torch.min(torch.abs(delta_all)).item())
        print(
            f"[ACTUATOR] {joint_name}: before={before_all[0].item():+.5f} "
            f"after={after_all[0].item():+.5f} delta={delta_all[0].item():+.5f} "
            f"batch_abs_delta=[{min_delta:.5f}, {torch.max(torch.abs(delta_all)).item():.5f}]",
            flush=True,
        )
        if min_delta < 0.01:
            raise RuntimeError(
                f"DOFBOT_V2 driven joint did not move in every environment: "
                f"{joint_name}, min_abs_delta={min_delta}"
            )

    env.reset()
    right_id = unwrapped._gripper_driver_joint_id
    left_id = unwrapped._gripper_mimic_joint_id
    joint_pos = unwrapped.robot.data.default_joint_pos.clone()
    joint_pos[:, unwrapped._arm_joint_ids] = torch.tensor(
        unwrapped.cfg.initial_arm_positions_rad, device=unwrapped.device
    )
    joint_pos[:, unwrapped._wrist_joint_id] = 0.0
    joint_pos[:, right_id] = unwrapped.cfg.gripper_driver_open_target
    joint_pos[:, left_id] = unwrapped.cfg.gripper_mimic_open_position
    joint_vel = torch.zeros_like(joint_pos)
    unwrapped.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    unwrapped._joint_targets[:] = joint_pos
    object_state = unwrapped.object.data.root_state_w.clone()
    object_state[:, :3] = unwrapped.scene.env_origins + torch.tensor(
        [0.25, 0.35, 0.20], device=unwrapped.device
    )
    object_state[:, 7:] = 0.0
    unwrapped.object.write_root_pose_to_sim(object_state[:, :7])
    unwrapped.object.write_root_velocity_to_sim(object_state[:, 7:])
    unwrapped._object_initial_pos_w[:] = object_state[:, :3]
    open_action = torch.zeros(
        (unwrapped.num_envs, unwrapped.cfg.action_space), device=unwrapped.device
    )
    open_action[:, 5] = -1.0
    # Refresh Isaac Lab's articulation cache and let the closed-loop linkage
    # settle in its constraint-consistent open pose before measuring motion.
    for _ in range(10):
        env.step(open_action)
    right_before_all = unwrapped.robot.data.joint_pos[:, right_id].clone()
    left_before_all = unwrapped.robot.data.joint_pos[:, left_id].clone()
    fingertip_ids = unwrapped._fingertip_body_ids
    gap_before_all = torch.linalg.norm(
        unwrapped.robot.data.body_pos_w.torch[:, fingertip_ids[0]]
        - unwrapped.robot.data.body_pos_w.torch[:, fingertip_ids[1]],
        dim=-1,
    )
    close_action = torch.zeros((unwrapped.num_envs, unwrapped.cfg.action_space), device=unwrapped.device)
    close_action[:, 5] = 1.0
    # Phase 0 intentionally forces the gripper open. Phase 2 retains a commanded
    # grasp and therefore isolates the physical close/mimic actuator test.
    for _ in range(60):
        unwrapped._task_phase.fill_(2)
        unwrapped._grasp_loss_steps.zero_()
        unwrapped._task_failed.zero_()
        env.step(close_action)
    right_delta_all = unwrapped.robot.data.joint_pos[:, right_id] - right_before_all
    left_delta_all = unwrapped.robot.data.joint_pos[:, left_id] - left_before_all
    gap_after_all = torch.linalg.norm(
        unwrapped.robot.data.body_pos_w.torch[:, fingertip_ids[0]]
        - unwrapped.robot.data.body_pos_w.torch[:, fingertip_ids[1]],
        dim=-1,
    )
    right_delta = float(right_delta_all[0].item())
    left_delta = float(left_delta_all[0].item())
    gap_before = float(gap_before_all[0].item())
    gap_after = float(gap_after_all[0].item())
    closing = (
        (unwrapped.robot.data.joint_pos[:, right_id] >= unwrapped.cfg.gripper_driver_grasp_min_position)
        & (left_delta_all < -0.10)
        & (gap_after_all <= unwrapped.cfg.gripper_grasp_max_gap)
    )
    print(
        f"[ACTUATOR] right-finger driver delta={right_delta:+.5f}, "
        f"left-finger mimic delta={left_delta:+.5f}, "
        f"fingertip gap={gap_before:.5f}->{gap_after:.5f} m, "
        f"batch_close_rate={closing.float().mean().item():.3f}, "
        f"right_delta_range=[{right_delta_all.min().item():+.5f}, {right_delta_all.max().item():+.5f}]",
        flush=True,
    )
    if not bool(torch.all(closing)):
        raise RuntimeError(
            "DOFBOT_V2 physical gripper closure failed: "
            f"right_delta={right_delta}, left_delta={left_delta}, "
            f"gap_before={gap_before}, gap_after={gap_after}"
        )


def main():
    task_id = args_cli.task or ENV_ID
    if task_id == IK_ENV_ID:
        env_cfg = DofbotReachIKEnvCfg()
    elif task_id == ENV_ID:
        env_cfg = DofbotReachEnvCfg()
    elif task_id == PICK_PLACE_REACH_ENV_ID:
        env_cfg = DofbotPickPlaceReachEnvCfg()
    elif task_id == PICK_PLACE_LIFT_ENV_ID:
        env_cfg = DofbotPickPlaceLiftEnvCfg()
    elif task_id == PICK_PLACE_ENV_ID:
        env_cfg = DofbotPickPlaceEnvCfg()
    else:
        raise ValueError(f"Unsupported DOFBOT task id: {task_id}")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.actuator_test and hasattr(env_cfg, "phase_transition_hold_steps"):
        # The synthetic state-machine check below deliberately evaluates one
        # sample per gate; physical hold timing is covered by deterministic sanity.
        env_cfg.phase_transition_hold_steps = (1, 1, 1, 1)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    print(f"[SMOKE] reset ok: obs={tuple(obs['policy'].shape)}", flush=True)
    if not bool(torch.isfinite(obs["policy"]).all()):
        raise RuntimeError("Non-finite observation detected immediately after reset")
    if task_id in (PICK_PLACE_REACH_ENV_ID, PICK_PLACE_LIFT_ENV_ID, PICK_PLACE_ENV_ID):
        unwrapped = env.unwrapped
        base_id = unwrapped.robot.find_bodies("base_link")[0][0]
        base_local = unwrapped.robot.data.body_pos_w.torch[:, base_id] - unwrapped.scene.env_origins
        max_base_error = float(torch.max(torch.abs(base_local)).item())
        print(
            f"[SMOKE] base_link local={base_local[0].detach().cpu().tolist()} "
            f"max_error={max_base_error:.8f}",
            flush=True,
        )
        if max_base_error > 1.0e-5:
            raise RuntimeError(f"DOFBOT_V2 base_link is not at local origin: max error={max_base_error}")
        initial_driven = unwrapped.robot.data.joint_pos[:, unwrapped._controlled_joint_ids]
        initial_driver = unwrapped.robot.data.joint_pos[:, unwrapped._gripper_driver_joint_id]
        max_initial_error = float(
            torch.max(torch.abs(torch.cat((initial_driven, initial_driver.unsqueeze(-1)), dim=-1))).item()
        )
        print(
            f"[SMOKE] initial driven joints={initial_driven[0].detach().cpu().tolist()} "
            f"right_finger={initial_driver[0].item():+.6f} max_error={max_initial_error:.8f}",
            flush=True,
        )
        if max_initial_error > 1.0e-5:
            raise RuntimeError(
                "DOFBOT controlled joint reset is not exactly (0, 0, 0, 0, 0, 0): "
                f"max error={max_initial_error}"
            )
        if args_cli.actuator_test:
            run_pick_place_actuator_test(env)

    for step_idx in range(args_cli.num_steps):
        actions = 2.0 * torch.rand((env.unwrapped.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device) - 1.0
        obs, reward, terminated, truncated, _ = env.step(actions)
        if not bool(torch.isfinite(obs["policy"]).all()):
            raise RuntimeError(f"Non-finite observation detected at step {step_idx + 1}")
        if not bool(torch.isfinite(reward).all()):
            raise RuntimeError(f"Non-finite reward detected at step {step_idx + 1}")
        print(
            f"[SMOKE] step {step_idx + 1}: "
            f"obs={tuple(obs['policy'].shape)} "
            f"reward_mean={reward.mean().item():.4f} "
            f"done={int((terminated | truncated).sum().item())}",
            flush=True,
        )

    env.close()
    print(
        f"[SMOKE:PASS] task={task_id} envs={args_cli.num_envs} steps={args_cli.num_steps}",
        flush=True,
    )
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
