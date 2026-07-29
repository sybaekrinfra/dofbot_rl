from __future__ import annotations

import gymnasium as gym

from .dofbot_reach_cfg import DofbotReachEnvCfg, DofbotReachIKEnvCfg

ENV_ID = "Dofbot-Reach-Direct-v0"
IK_ENV_ID = "Dofbot-Reach-IK-Direct-v0"

# Register on import so train/play scripts can create the env via gym.make().
gym.register(
    id=ENV_ID,
    entry_point="dofbot_rl.tasks.dofbot_reach_env:DofbotReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DofbotReachEnvCfg,
        "rsl_rl_cfg_entry_point": "dofbot_rl.tasks.agents.rsl_rl_ppo_cfg:DofbotReachPPORunnerCfg",
    },
)

gym.register(
    id=IK_ENV_ID,
    entry_point="dofbot_rl.tasks.dofbot_reach_env:DofbotReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DofbotReachIKEnvCfg,
        "rsl_rl_cfg_entry_point": "dofbot_rl.tasks.agents.rsl_rl_ppo_cfg:DofbotReachIKPPORunnerCfg",
    },
)


def register_tasks() -> list[str]:
    """External callback used by Isaac Lab launchers to register DOFBOT tasks."""
    return []
