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

    zeros = torch.zeros((unwrapped.num_envs,), device=unwrapped.device)
    ones = torch.ones_like(zeros)
    unwrapped._task_phase.zero_()
    unwrapped._gripper_command.zero_()
    unwrapped._update_task_phase(zeros, ones, ones, ones, zeros, ones, zeros.bool())
    unwrapped._gripper_command.fill_(0.8)
    unwrapped._update_task_phase(ones, zeros, ones, ones, zeros, ones, ones.bool())
    unwrapped._update_task_phase(
        ones, zeros, ones, ones, ones * (unwrapped.cfg.lift_height + 0.01), ones, ones.bool()
    )
    unwrapped._update_task_phase(ones, zeros, zeros, zeros, ones, ones, ones.bool())
    if not bool(torch.all(unwrapped._task_phase == 4)):
        raise RuntimeError(f"Pick–Place phase transition failed: {unwrapped._task_phase}")
    print("[PHASE] 0 approach -> 1 grasp -> 2 lift -> 3 transport -> 4 place/release: ok", flush=True)

    driven_names = [*unwrapped.cfg.arm_joint_names, unwrapped.cfg.wrist_joint_name]
    driven_ids = unwrapped._controlled_joint_ids

    for action_index, (joint_name, joint_id) in enumerate(zip(driven_names, driven_ids, strict=True)):
        env.reset()
        before = float(unwrapped.robot.data.joint_pos[0, joint_id].item())
        action = torch.zeros((unwrapped.num_envs, unwrapped.cfg.action_space), device=unwrapped.device)
        action[:, action_index] = 1.0
        for _ in range(10):
            env.step(action)
        after = float(unwrapped.robot.data.joint_pos[0, joint_id].item())
        delta = after - before
        print(f"[ACTUATOR] {joint_name}: before={before:+.5f} after={after:+.5f} delta={delta:+.5f}", flush=True)
        if abs(delta) < 0.01:
            raise RuntimeError(f"DOFBOT_V2 driven joint did not move: {joint_name}, delta={delta}")

    env.reset()
    right_id = unwrapped._gripper_driver_joint_id
    left_id = unwrapped._gripper_mimic_joint_id
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
    for _ in range(20):
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
    closing = (right_delta_all > 0.045) & (left_delta_all < -0.045) & (gap_after_all < gap_before_all)
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
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    print(f"[SMOKE] reset ok: obs={tuple(obs['policy'].shape)}", flush=True)
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
        object_local = unwrapped.object.data.root_pos_w.torch - unwrapped._base_pos_w()
        expected_joint1 = -torch.atan2(object_local[:, 0], object_local[:, 1])
        actual_joint1 = unwrapped.robot.data.joint_pos[:, unwrapped._arm_joint_ids[0]]
        max_heading_error = float(torch.max(torch.abs(actual_joint1 - expected_joint1)).item())
        print(
            f"[SMOKE] cube heading: expected_joint1={expected_joint1[0].item():+.5f} "
            f"actual_joint1={actual_joint1[0].item():+.5f} error={max_heading_error:.5f}",
            flush=True,
        )
        if max_heading_error > unwrapped.cfg.initial_joint_noise_rad + 1.0e-3:
            raise RuntimeError(f"joint1 does not face the sampled cube: error={max_heading_error}")
        if args_cli.actuator_test:
            run_pick_place_actuator_test(env)

    for step_idx in range(args_cli.num_steps):
        actions = 2.0 * torch.rand((env.unwrapped.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device) - 1.0
        obs, reward, terminated, truncated, _ = env.step(actions)
        print(
            f"[SMOKE] step {step_idx + 1}: "
            f"obs={tuple(obs['policy'].shape)} "
            f"reward_mean={reward.mean().item():.4f} "
            f"done={int((terminated | truncated).sum().item())}",
            flush=True,
        )

    env.close()
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
