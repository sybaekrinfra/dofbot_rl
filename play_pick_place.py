from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained DOFBOT_V2 Pick–Place policy.")
parser.add_argument("--task", type=str, default="Dofbot-V2-PickPlace-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=0, help="Use 0 to run until the app is closed.")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--debug_interval",
    type=int,
    default=0,
    help="Print grasp phase, distance, close action and physical finger positions every N steps.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from dofbot_rl.tasks import (
    PICK_PLACE_ENV_ID,
    PICK_PLACE_LIFT_ENV_ID,
    PICK_PLACE_REACH_ENV_ID,
)
from dofbot_rl.tasks.agents.rsl_rl_ppo_cfg import DofbotPickPlacePPORunnerCfg
from dofbot_rl.tasks.dofbot_pick_place_cfg import (
    DofbotPickPlaceEnvCfg,
    DofbotPickPlaceLiftEnvCfg,
    DofbotPickPlaceReachEnvCfg,
)
import dofbot_rl.tasks  # noqa: F401


def get_env_cfg(task_id: str) -> DofbotPickPlaceEnvCfg:
    if task_id == PICK_PLACE_REACH_ENV_ID:
        return DofbotPickPlaceReachEnvCfg()
    if task_id == PICK_PLACE_LIFT_ENV_ID:
        return DofbotPickPlaceLiftEnvCfg()
    if task_id == PICK_PLACE_ENV_ID:
        return DofbotPickPlaceEnvCfg()
    raise ValueError(f"Unsupported DOFBOT_V2 Pick–Place task: {task_id}")


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg = get_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    agent_cfg = DofbotPickPlacePPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        print(f"[INFO] Loading DOFBOT_V2 Pick–Place checkpoint: {checkpoint}", flush=True)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = env.get_observations()
        step_count = 0
        dt = env.unwrapped.step_dt
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step_count += 1
            if args_cli.debug_interval > 0 and step_count % args_cli.debug_interval == 0:
                unwrapped = env.unwrapped
                _, _, _, reach_dist, _, _, _ = unwrapped._task_state()
                right_pos = unwrapped.robot.data.joint_pos[
                    :, unwrapped._gripper_driver_joint_id
                ]
                left_pos = unwrapped.robot.data.joint_pos[
                    :, unwrapped._gripper_mimic_joint_id
                ]
                print(
                    f"[GRASP DEBUG] step={step_count} phase={int(unwrapped._task_phase[0].item())} "
                    f"distance={reach_dist[0].item():.4f} action={actions[0, 5].item():+.3f} "
                    f"command={unwrapped._gripper_command[0].item():.3f} "
                    f"right={right_pos[0].item():+.3f} left={left_pos[0].item():+.3f}",
                    flush=True,
                )
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
