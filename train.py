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

parser = argparse.ArgumentParser(description="Train a DOFBOT reach policy with Isaac Lab and RL-Games.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum PPO iterations.")
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

from dofbot_rl.tasks import ENV_ID  # noqa: E402
from dofbot_rl.tasks.dofbot_reach_cfg import DofbotReachEnvCfg  # noqa: E402
import dofbot_rl.tasks  # noqa: E402,F401 - ensures gym registration happens


def build_agent_cfg(num_envs: int, max_iterations: int, experiment_name: str) -> dict:
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
                "max_epochs": max_iterations,
                "save_frequency": 50,
                "save_best_after": 50,
                "print_interval": 20,
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
                "minibatch_size": 512,
                "mini_epochs": 5,
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


def main():
    env_cfg = DofbotReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 1.75
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    experiment_name = datetime.now().strftime("dofbot_reach_%Y-%m-%d_%H-%M-%S")
    agent_cfg = build_agent_cfg(args_cli.num_envs, args_cli.max_iterations, experiment_name)

    env = gym.make(ENV_ID, cfg=env_cfg, render_mode=None)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"]["clip_observations"]
    clip_actions = agent_cfg["params"]["env"]["clip_actions"]
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

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
    runner.run({"train": True, "play": False})

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
