import ray.rllib.agents.ppo as ppo


def init_default_ppo_config(train_config) -> dict:
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = train_config["num_gpus"]
    config["log_level"] = train_config["log_level"]
    config["framework"] = train_config["framework"]
    config["env"] = train_config["env"]
    config["env_config"] = train_config["env_config"]
    config["model"] = train_config["model"]
    config["rollout_fragment_length"] = train_config["rollout_fragment_length"]
    config["entropy_coeff"] = train_config["entropy_coeff"]
    config["lambda"] = train_config["lambda"]
    config["vf_loss_coeff"] = train_config["vf_loss_coeff"]
    return config
