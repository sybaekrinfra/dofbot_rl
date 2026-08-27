from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DofbotReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 50
    experiment_name = "dofbot_reach_direct_refine"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        # rsl_rl 3.0 raises adaptive LR by 1.5x up to 1e-2 whenever measured
        # KL is below half the target.  The recorded runs reached that ceiling,
        # so keep the verified conservative rate instead.
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DofbotReachIKPPORunnerCfg(DofbotReachPPORunnerCfg):
    """RSL-RL runner config for the Cartesian-IK DOFBOT reach variant."""

    experiment_name = "dofbot_reach_ik_direct_refine"


@configclass
class DofbotPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration shared by all DOFBOT_V2 Pick–Place curriculum stages."""

    num_steps_per_env = 48
    max_iterations = 4000
    save_interval = 100
    experiment_name = "dofbot_v2_pick_place"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 2048 parallel environments already provide broad exploration.  A
        # smaller entropy bonus lets Reach converge to millimetre-scale XY/Z.
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        # Do not let low early-stage KL inflate 3e-4 to rsl_rl's 1e-2 cap.
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
