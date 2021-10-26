import os
import shutil
import argparse
import json

import ray
import ray.rllib.agents.ppo as ppo
import wandb
from ray import tune

from config import CONFIG_FILE
from config.ppo_config_init import init_default_ppo_config
from ppo import ModifiedDungeon
from config.wandb_init import wandb_init
from util import save_image, save_gif, log_results


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-i', '--input_config_path',
                        type=str,
                        default=CONFIG_FILE.parent / 'config.json',
                        help='Path to json configuration file. ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(args.input_config_path))

    wandb_init(config)

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("ModifiedDungeon", lambda config: ModifiedDungeon(**config))
    shutil.rmtree(config["global"]["checkpoint_directory_path"], ignore_errors=True, onerror=None)
    ray_results = os.getenv("HOME") + "/ray_results1/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    default_config = init_default_ppo_config(config["train"])
    agent = ppo.PPOTrainer(default_config)

    iterations_count = config["global"]["n_iterations"]

    for n in range(iterations_count):
        result = agent.train()
        file_name = agent.save(f'{config["global"]["checkpoint_directory_path"]}')
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(f'{file_name}')
        wandb.log_artifact(artifact)
        log_results(result)

        if (n + 1) % config["global"]["log_each_n_iter"] == 0:
            env = ModifiedDungeon(**config["train"]["env_config"])
            observation = env.reset()
            data = env._map.render(env._agent)
            save_image(data)
            gif_name = f'{config["global"]["gif_name"]}iter_{n + 1}.gif'
            save_gif(config["global"]["n_actions_to_save_image"], gif_name, agent, env, observation)
