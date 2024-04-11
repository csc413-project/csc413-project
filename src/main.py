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
from utils import denormalize_images


@dataclass
class DreamerConfig:
    # env setting
    domain_name: str = "cartpole"
    task_name: str = "swingup"
    obs_image_size: Tuple = (64, 64)
    action_repeats: int = 2
    # general setting
    base_dir = f"/home/scott/tmp/dreamer/{domain_name}_{task_name}/1/"
    data_dir: str = os.path.join(base_dir, "episodes")  # where to store trajectories
    model_dir: str = os.path.join(base_dir, "models")  # where to store models
    # training setting
    prefill_episodes = 0  # number of episodes to prefill the dataset
    batch_size: int = 50  # batch size for training
    batch_length: int = 50  # sequence length of each training batch
    training_steps: int = 100  # number of training steps
    training_device = "cuda"  # training device
    # testing setting
    test_every: int = 25  # test (and save model) every n episodes
    # collector setting
    collector_device = "cuda"  # collector device

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


def main():
    config = DreamerConfig()

    # wandb.login(key=os.getenv("WANDB_KEY"))
    wandb.init(
        project="csc413-proj",
        config=asdict(config),
        name=f"dreamer-{config.domain_name}_{config.task_name}",
        entity="scott-reseach",
    )
    wandb.define_metric("env_steps")
    wandb.define_metric("agent/*", step_metric="env_steps")
    wandb.define_metric("training_steps")
    wandb.define_metric("train/*", step_metric="training_steps")

    env = DMCEnv(
        config.domain_name,
        config.task_name,
        config.obs_image_size,
        action_repeat=config.action_repeats,
    )
    action_spec = env.action_space

    # init dreamer
    dreamer = Dreamer(
        AgentModel(action_shape=action_spec.shape), device=config.training_device
    )

    # prefill dataset with 5 random trajectories
    total_env_steps = 0
    train_collector = Collector(env, dreamer.agent, True, config.collector_device)
    if config.prefill_episodes > 0:
        prefill_data, _, (_, total_env_steps) = train_collector.collect(
            target_num_episodes=config.prefill_episodes, random_action=True
        )
        for i, data in enumerate(prefill_data):
            np.savez_compressed(os.path.join(config.data_dir, f"pre_{i}.npz"), **data)
    test_collector = Collector(env, dreamer.agent, False, config.collector_device)

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
        train_collector.reset_agent(dreamer.agent)
        data, _, (_, env_steps) = train_collector.collect(target_num_episodes=1)
        total_env_steps += env_steps
        data = data[0]
        np.savez_compressed(os.path.join(config.data_dir, f"{i}.npz"), **data)
        wandb.log(
            {
                "env_steps": total_env_steps * config.action_repeats,
                "agent/training_return": sum(data["reward"]),
            }
        )

        # test
        if i % config.test_every == 0:
            print("Testing...")
            torch.save(
                dreamer.agent.state_dict(),
                os.path.join(config.model_dir, f"{total_env_steps}.pt"),
            )
            test_collector.reset_agent(dreamer.agent)
            data, _, _ = test_collector.collect(target_num_episodes=1)
            data = data[0]
            observations = denormalize_images(np.array(data["obs"]))
            wandb.log(
                {
                    "agent/test_return": sum(data["reward"]),
                    "agent/test_video": wandb.Video(observations, fps=30, format="mp4")
                }
            )


if __name__ == "__main__":
    main()
