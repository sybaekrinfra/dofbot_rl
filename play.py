from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for source_dir in ("isaaclab", "isaaclab_rl", "isaaclab_tasks"):
    source_path = PROJECT_ROOT / "source" / source_dir
    if source_path.is_dir() and str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a trained DOFBOT reach policy in Isaac Lab.")
parser.add_argument("--task", type=str, default=None, help="Gym task id. Defaults to the joint-delta reach task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained RL-Games checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from dofbot_rl.tasks import ENV_ID, IK_ENV_ID  # noqa: E402
from dofbot_rl.tasks.dofbot_reach_cfg import DofbotReachEnvCfg, DofbotReachIKEnvCfg  # noqa: E402
import dofbot_rl.tasks  # noqa: E402,F401 - ensures gym registration happens


def find_latest_checkpoint() -> str | None:
    root = Path("logs") / "rl_games"
    if not root.exists():
        return None
    candidates = sorted(root.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def build_agent_cfg(num_envs: int, experiment_name: str) -> dict:
    rl_device = args_cli.device if args_cli.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    return {
        "params": {
            "seed": args_cli.seed,
            "env": {
                "clip_observations": 5.0,
                "clip_actions": 1.0,
            },
            "algo": {
                "name": "a2c_continuous",
            },
            "model": {
                "name": "continuous_a2c_logstd",
            },
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": None,
                        "sigma_activation": "softplus",
                    }
                },
                "mu_init": {
                    "name": "orthogonal_initializer",
                    "gain": 0.0141421356237,
                },
                "sigma_init": {
                    "name": "const_initializer",
                    "val": -0.5,
                    "fixed_sigma": False,
                },
                "mlp": {
                    "units": [256, 256],
                    "activation": "elu",
                    "initializer": {
                        "name": "orthogonal_initializer",
                    },
                    "regularizer": {
                        "name": None,
                    },
                },
            },
            "config": {
                "name": "DofbotReach",
                "full_experiment_name": experiment_name,
                "env_name": "rlgpu",
                "train_dir": str(Path("logs") / "rl_games"),
                "max_epochs": 1,
                "save_frequency": 1,
                "save_best_after": 1,
                "print_interval": 1,
                "num_actors": num_envs,
                "reward_shaper": {
                    "scale_value": 1.0,
                },
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 3e-4,
                "lr_schedule": "adaptive",
                "kl_threshold": 0.01,
                "entropy_coef": 0.0,
                "e_clip": 0.2,
                "horizon_length": 32,
                "minibatch_size": 1,
                "mini_epochs": 1,
                "critic_coef": 2.0,
                "grad_norm": 1.0,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "normalize_advantage": True,
                "value_bootstrap": True,
                "truncate_grads": True,
                "print_stats": True,
                "device": rl_device,
                "device_name": rl_device,
                "multi_gpu": False,
                "score_to_win": 20000,
            },
        }
    }


def make_env_cfg(task_id: str) -> DofbotReachEnvCfg:
    if task_id == IK_ENV_ID:
        return DofbotReachIKEnvCfg()
    if task_id == ENV_ID:
        return DofbotReachEnvCfg()
    raise ValueError(f"Unsupported DOFBOT task id: {task_id}")


def main():
    task_id = args_cli.task or ENV_ID
    env_cfg = make_env_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 1.75
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    experiment_name = datetime.now().strftime("dofbot_reach_play_%Y-%m-%d_%H-%M-%S")
    agent_cfg = build_agent_cfg(1, experiment_name)

    env = gym.make(task_id, cfg=env_cfg, render_mode="human")
    rl_device = agent_cfg["params"]["config"]["device"]
    env = RlGamesVecEnvWrapper(env, rl_device, 5.0, 1.0)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "IsaacRlgWrapper",
            "env_creator": lambda **kwargs: env,
        },
    )

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    checkpoint = args_cli.checkpoint or find_latest_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError(
            "No checkpoint was provided and no .pth files were found under logs/rl_games."
        )

    runner.run({"train": False, "play": True, "checkpoint": checkpoint})
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
