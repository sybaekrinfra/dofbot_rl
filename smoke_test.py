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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from dofbot_rl.tasks import ENV_ID, IK_ENV_ID
from dofbot_rl.tasks.dofbot_reach_cfg import DofbotReachEnvCfg, DofbotReachIKEnvCfg
import dofbot_rl.tasks  # noqa: F401


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
    print(f"[SMOKE] reset ok: obs={tuple(obs['policy'].shape)}", flush=True)

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
