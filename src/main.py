import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import torch
import wandb
from torch.utils.data.dataset import Dataset
from tqdm import trange

from data import TrajectoryDataset, Collector
from dreamer import Dreamer
from envs import DMCEnv
from models.agent import AgentModel


@dataclass
class DreamerConfig:
    # env setting
    domain_name: str = "cartpole"
    task_name: str = "swingup"
    obs_image_size: Tuple = (64, 64)

    base_dir = f"/home/scott/tmp/dreamer/{domain_name}_{task_name}/"
    data_dir: str = os.path.join(base_dir, "episodes")  # where to store trajectories
    model_dir: str = os.path.join(base_dir, "models")  # where to store models

    prefill_episodes = 0  # number of episodes to prefill the dataset
    batch_size: int = 50  # batch size for training
    batch_length: int = 50  # sequence length of each training batch
    training_steps: int = 100  # number of training steps
    training_device = "cuda"  # training device

    collector_device = "cuda"  # collector device


def init_dirs(config: DreamerConfig):
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    return config


def main():
    config = DreamerConfig()
    config = init_dirs(config)

    # wandb.login(key=os.getenv("WANDB_KEY"))
    wandb.init(
        project="csc413-proj",
        config=asdict(config),
        name=f"dreamer-{config.domain_name}_{config.task_name}",
        entity="scott-reseach",
    )
    wandb.define_metric("env_steps")
    wandb.define_metric("training_return", step_metric="env_steps")
    wandb.define_metric("training_steps")
    wandb.define_metric("train/*", step_metric="training_steps")

    env = DMCEnv(config.domain_name, config.task_name, config.obs_image_size)
    action_spec = env.action_space

    # init dreamer
    dreamer = Dreamer(
        AgentModel(action_shape=action_spec.shape), device=config.training_device
    )

    # prefill dataset with 5 random trajectories
    total_env_steps = 0
    collector = Collector(env, dreamer.agent, config.collector_device)
    if config.prefill_episodes > 0:
        prefill_data, _, (_, total_env_steps) = collector.collect(
            target_num_episodes=config.prefill_episodes, random_action=True
        )
        for i, data in enumerate(prefill_data):
            np.savez_compressed(os.path.join(config.data_dir, f"pre_{i}.npz"), **data)

    for i in trange(1000, desc="Training Epochs"):
        # train
        dataset = TrajectoryDataset(config.data_dir, config.batch_length)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, num_workers=10
        )
        data_iter = iter(dataloader)
        dreamer.agent.train()
        for _ in trange(config.training_steps, desc="Training Steps"):
            try:
                obs, action, reward = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                obs, action, reward = next(data_iter)
            obs, action, reward = (
                obs.to(config.training_device),
                action.to(config.training_device),
                reward.to(config.training_device),
            )
            dreamer.update(obs, action, reward)

        # collect
        collector.reset_agent(dreamer.agent)
        data, _, (_, env_steps) = collector.collect(target_num_episodes=1)
        total_env_steps += env_steps
        data = data[0]
        np.savez_compressed(os.path.join(config.data_dir, f"{i}.npz"), **data)
        wandb.log(
            {
                "env_steps": total_env_steps,
                "training_return": sum(data["reward"]),
            }
        )

        if i % 25 == 0:
            torch.save(
                dreamer.agent.state_dict(),
                os.path.join(config.model_dir, f"{total_env_steps}.pt"),
            )


if __name__ == "__main__":
    main()
