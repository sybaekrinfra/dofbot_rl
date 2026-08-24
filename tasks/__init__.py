from __future__ import annotations

import gymnasium as gym

from .dofbot_reach_cfg import DofbotReachEnvCfg, DofbotReachIKEnvCfg
from .dofbot_pick_place_cfg import (
    DofbotPickPlaceEnvCfg,
    DofbotPickPlaceLiftEnvCfg,
    DofbotPickPlaceReachEnvCfg,
)

ENV_ID = "Dofbot-Reach-Direct-v0"
IK_ENV_ID = "Dofbot-Reach-IK-Direct-v0"
PICK_PLACE_REACH_ENV_ID = "Dofbot-V2-PickPlace-Reach-Direct-v0"
PICK_PLACE_LIFT_ENV_ID = "Dofbot-V2-PickPlace-Lift-Direct-v0"
PICK_PLACE_ENV_ID = "Dofbot-V2-PickPlace-Direct-v0"

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

for task_id, env_cfg in (
    (PICK_PLACE_REACH_ENV_ID, DofbotPickPlaceReachEnvCfg),
    (PICK_PLACE_LIFT_ENV_ID, DofbotPickPlaceLiftEnvCfg),
    (PICK_PLACE_ENV_ID, DofbotPickPlaceEnvCfg),
):
    gym.register(
        id=task_id,
        entry_point="dofbot_rl.tasks.dofbot_pick_place_env:DofbotPickPlaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": (
                "dofbot_rl.tasks.agents.rsl_rl_ppo_cfg:DofbotPickPlacePPORunnerCfg"
            ),
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
