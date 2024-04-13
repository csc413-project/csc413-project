import os
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import wandb
from tqdm import trange

from data import Collector, ExperienceReplayDataset, VectorCollector, VectorDMCEnv
from dreamer import Dreamer
from envs import DMCEnv
from models.agent import AgentModel
from utils import denormalize_images, count_env_steps

os.putenv("MUJOCO_GL", "osmesa")

ENV_SETTINGS = {
    "acrobot": ["swingup"],
    "cartpole": ["swingup", "swingup_sparse"],
    "cheetah": ["run"],
    "finger": ["turn_easy", "turn_hard", "spin"],
    "hopper": ["hop"],
}

ENV_PREFERRED_CAMERA = {
    ("acrobot", "swingup"): 0,
    ("cartpole", "swingup"): None,
    ("cartpole", "swingup_sparse"): None,
    ("cheetah", "run"): 0,
    ("finger", "turn_easy"): 1,
    ("finger", "turn_hard"): 1,
    ("finger", "spin"): 1,
    ("hopper", "hop"): 0,
}


@dataclass
class DreamerConfig:
    # env setting
    domain_name: str = "cheetah"
    task_name: str = "run"
    obs_image_size: Tuple = (64, 64)
    action_repeats: int = 2
    camera_id: int = ENV_PREFERRED_CAMERA[(domain_name, task_name)]
    render_kwargs: Dict = None
    # general setting
    base_dir = f"/home/scott/tmp/dreamer/{domain_name}_{task_name}/2/"
    data_dir: str = os.path.join(base_dir, "episodes")  # where to store trajectories
    model_dir: str = os.path.join(base_dir, "models")  # where to store models
    load_model_path: Optional[str] = None
    debug: bool = False  # if True, then wandb will be disabled
    # training setting
    training_epochs: int = 1100  # number of training episodes
    prefill_episodes = 5  # number of episodes to prefill the dataset
    batch_size: int = 50  # batch size for training
    batch_length: int = 50  # sequence length of each training batch
    training_steps: int = 100  # number of training steps
    training_device = "cuda"  # training device
    # testing setting
    test_every: int = 8  # test (and save model) every n episodes
    test_num_envs: int = 5  # number of parallel test environments
    test_env_starting_seed: int = 1108  # starting seed for test environments
    # collector setting
    collector_device = "cuda"  # collector device

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        if self.camera_id is None:
            self.render_kwargs = {}
        else:
            self.render_kwargs = {
                "camera_id": self.camera_id,
            }


def main():
    config = DreamerConfig()

    # wandb.login(key=os.getenv("WANDB_KEY"))
    wandb.init(
        project="csc413-proj",
        config=asdict(config),
        name=f"dreamer-{config.domain_name}_{config.task_name}",
        entity="scott-reseach",
        mode="disabled" if config.debug else "online",
    )
    wandb.define_metric("env_steps")
    wandb.define_metric("agent/*", step_metric="env_steps")
    wandb.define_metric("dataset/*", step_metric="env_steps")
    wandb.define_metric("training_steps")
    wandb.define_metric("train/*", step_metric="training_steps")

    # init env
    env = DMCEnv(
        config.domain_name,
        config.task_name,
        config.obs_image_size,
        action_repeat=config.action_repeats,
        render_kwargs=config.render_kwargs,
    )  # for training
    action_spec = env.action_space

    def create_env(seed: int = 0):  # for testing and initial data collection
        return DMCEnv(
            domain_name=config.domain_name,
            task_name=config.task_name,
            seed=seed,
            render_kwargs=config.render_kwargs,
        )

    # init dreamer
    agent = AgentModel(action_shape=action_spec.shape)
    if config.load_model_path is not None:
        agent.load_state_dict(torch.load(config.load_model_path))
    dreamer = Dreamer(agent, device=config.training_device)

    # init buffer
    buffer = ExperienceReplayDataset(
        config.data_dir,
        buffer_size=100,
        max_episode_length=1000 // config.action_repeats,
        obs_shape=env.observation_space.shape,
        action_size=np.prod(action_spec.shape).item(),
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        device=config.training_device,
    )

    # init collectors
    train_collector = Collector(env, dreamer.agent, True, config.collector_device)
    test_collector = VectorCollector(
        VectorDMCEnv(create_env, config.test_num_envs, config.test_env_starting_seed),
        dreamer.agent,
        False,
        config.collector_device,
    )

    # prefill dataset with 5 random trajectories, if needed
    total_env_steps, episodes = count_env_steps(config.data_dir, config.action_repeats)
    if config.prefill_episodes - episodes > 0:
        prefill_collector = VectorCollector(
            VectorDMCEnv(
                create_env,
                config.prefill_episodes - episodes,
                config.test_env_starting_seed,
            ),
            dreamer.agent,
            False,
            config.collector_device,
        )
        prefill_data, _, (_, env_steps) = prefill_collector.collect(
            target_num_episodes=1, random_action=True
        )
        prefill_collector.close()
        prefill_data = {k: np.array(v) for k, v in prefill_data[0].items()}
        total_env_steps += env_steps * config.action_repeats * len(prefill_data)
        prefill_data = [
            {
                "obs": prefill_data["obs"][:, i],
                "action": prefill_data["action"][:, i],
                "reward": prefill_data["reward"][:, i],
            }
            for i in range(config.prefill_episodes - episodes)
        ]
        for i, data in enumerate(prefill_data):
            np.savez_compressed(
                os.path.join(config.data_dir, f"pre_{i + episodes}.npz"), **data
            )
            buffer.add_trajectory(data)

    for i in trange(config.training_epochs, desc="Training Epochs"):
        dreamer.agent.train()
        for _ in trange(config.training_steps, desc="Training Steps"):
            obs, action, reward = next(buffer)
            dreamer.update(obs, action, reward)

        # collect
        train_collector.reset_agent(dreamer.agent)
        data, _, (_, env_steps) = train_collector.collect(target_num_episodes=1)
        total_env_steps += env_steps * config.action_repeats
        data = data[0]  # only 1 trajectory
        buffer.add_trajectory(data)
        np.savez_compressed(os.path.join(config.data_dir, f"{i}.npz"), **data)
        wandb.log(
            {
                "env_steps": total_env_steps,
                "agent/training_return": sum(data["reward"]),
            }
        )

        # test
        if i % config.test_every == 0 or i == config.training_epochs - 1:
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
                    "agent/test_return": np.mean(np.sum(data["reward"], axis=0)),
                    "agent/test_video": [
                        wandb.Video(observations[:, j], fps=30, format="mp4")
                        for j in range(config.test_num_envs)
                    ],
                }
            )

    test_collector.close()


if __name__ == "__main__":
    main()
