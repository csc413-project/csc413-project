# main.py
import os
import wandb
from ppo import PPO, PPOConfig, init_wandb
from envs import DMCEnv  # Assuming this is a standardized environment class.
from data import Collector  # Handles data collection from the environment.

def create_env(config):
    return DMCEnv(
        domain_name=config.env_name,
        task_name=config.task_name,
        obs_image_size=config.obs_image_shape,
        action_repeat=config.action_repeats,
        render_kwargs=config.render_kwargs
    )

def main():
    config = PPOConfig(
        env_name="cheetah",
        action_shape=(6,),
        obs_image_shape=(3, 64, 64),
        model_dir="/path/to/models",
        data_dir="/path/to/data",
        training_epochs=1100
    )

    init_wandb(config)

    env = create_env(config)
    ppo = PPO(config, env)
    ppo.train()

    # Assuming there's a testing method or additional functionality to test the model:
    # test_model(ppo.model, env)

    wandb.finish()

if __name__ == "__main__":
    main()
