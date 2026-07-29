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

parser = argparse.ArgumentParser(description="Inspect DOFBOT reach end-effector reference frames.")
parser.add_argument("--task", type=str, default=None, help="Gym task id. Defaults to the joint-delta reach task.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dofbot_rl.tasks  # noqa: F401
from dofbot_rl.tasks import ENV_ID, IK_ENV_ID
from dofbot_rl.tasks.dofbot_reach_cfg import DofbotReachEnvCfg, DofbotReachIKEnvCfg


def _fmt_tensor(tensor: torch.Tensor) -> list[float]:
    return [round(float(value), 8) for value in tensor.detach().cpu().flatten()]


def main():
    task_id = args_cli.task or ENV_ID
    if task_id == IK_ENV_ID:
        env_cfg = DofbotReachIKEnvCfg()
    elif task_id == ENV_ID:
        env_cfg = DofbotReachEnvCfg()
    else:
        raise ValueError(f"Unsupported DOFBOT task id: {task_id}")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    target = unwrapped.target

    body_state = getattr(robot.data, "body_state_w", None)
    body_com_state = getattr(robot.data, "body_com_state_w", None)
    if body_state is None:
        raise RuntimeError("robot.data.body_state_w is not available in this Isaac Lab build.")

    ee_body_id = unwrapped._ee_body_id
    ee_state = body_state[:, ee_body_id]
    ee_tip_pos = unwrapped._get_end_effector_position()
    target_pos = target.data.root_pos_w[:, :3]
    link5_pos = ee_state[:, :3]
    link5_quat = ee_state[:, 3:7]
    zero_offset_dist = torch.linalg.norm(target_pos - link5_pos, dim=-1)
    tip_offset_dist = torch.linalg.norm(target_pos - ee_tip_pos, dim=-1)

    print("[EE-CHECK] asset ee_body_name:", unwrapped._ee_body_name, flush=True)
    print("[EE-CHECK] cfg ee_tip_offset:", _fmt_tensor(unwrapped._ee_tip_offset), flush=True)
    print("[EE-CHECK] body names:", robot.data.body_names, flush=True)
    print("[EE-CHECK] joint names:", robot.data.joint_names, flush=True)
    print("[EE-CHECK] obs shape:", tuple(obs["policy"].shape), flush=True)

    for env_id in range(unwrapped.num_envs):
        print(f"[EE-CHECK] env {env_id}", flush=True)
        print("  link5 body pos_w        :", _fmt_tensor(link5_pos[env_id]), flush=True)
        print("  link5 body quat_w       :", _fmt_tensor(link5_quat[env_id]), flush=True)
        if body_com_state is not None:
            com_pos = body_com_state[:, ee_body_id, :3]
            print("  link5 com pos_w         :", _fmt_tensor(com_pos[env_id]), flush=True)
            print("  body_pos - com_pos      :", _fmt_tensor(link5_pos[env_id] - com_pos[env_id]), flush=True)
        print("  computed ee tip pos_w   :", _fmt_tensor(ee_tip_pos[env_id]), flush=True)
        print("  target pos_w            :", _fmt_tensor(target_pos[env_id]), flush=True)
        print("  target - ee tip         :", _fmt_tensor(target_pos[env_id] - ee_tip_pos[env_id]), flush=True)
        print("  dist(target, link5 body):", round(float(zero_offset_dist[env_id]), 8), flush=True)
        print("  dist(target, ee tip)    :", round(float(tip_offset_dist[env_id]), 8), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
