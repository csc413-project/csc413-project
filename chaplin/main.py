import os
import threading
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import wandb
from tqdm import trange

from chaplin import Chaplin
from data import (
    ExperienceReplayDataset,
    VectorDMCEnv,
    VectorCollector,
    raw_data_to_trajectories,
    merge_raw_data,
)
from envs import DMCEnv
from models.agent import ChaplinAgent
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
class ChaplinConfig:
    # env setting
    domain_name: str = "cartpole"
    task_name: str = "swingup"
    obs_image_size: Tuple = (64, 64)
    action_repeats: int = 2
    camera_id: int = ENV_PREFERRED_CAMERA[(domain_name, task_name)]
    render_kwargs: Dict = None
    # general setting
    algorithm: str = "chaplin"
    # algorithm: str = "ppo"
    base_dir = f"/home/scott/tmp/{algorithm}/{domain_name}_{task_name}/0/"
    data_dir: str = os.path.join(base_dir, "episodes")  # where to store trajectories
    model_dir: str = os.path.join(base_dir, "models")  # where to store models
    load_model_path: Optional[str] = None
    debug: bool = False  # if True, then wandb will be disabled
    # training setting
    ppo_only: bool = algorithm == "ppo"  # if True, then only PPO is trained
    train_num_envs: int = 8  # number of parallel training environments
    iterations: int = 2000  # number of training episodes
    training_device = "cuda"  # training device
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95
    # dreamer setting
    prefill_episodes = 5  # number of episodes to prefill the dataset
    dreamer_batch_size: int = 100  # batch size for training
    dreamer_batch_length: int = 50  # sequence length of each training batch
    dreamer_training_steps: int = 100  # number of training steps
    model_lr: float = 1e-3  # learning rate for the world model
    # ppo setting
    ppo_T: int = 200  # number of steps to collect in each iteration for each env
    ppo_minibatch_size: int = 100
    ppo_minibatch_length: int = 50
    ppo_epochs: int = 3  # sample reuse
    ppo_training_steps: int = 8  # number of training steps
    ppo_epsilon: float = 0.2  # clip ratio
    ppo_value_loss_coef: float = 1.0
    ppo_entropy_loss_coef: float = 0.02
    policy_lr: float = 8e-5  # learning rate for the policy
    value_lr: float = 5e-4
    # testing setting
    test_every: int = 10  # test (and save model) every n iterations
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

    def step(self, i: int):
        """
        Specify parameter decay schedule here.
        :param i: current iteration
        """
        if i >= 100:
            self.dreamer_training_steps = 50


def main():
    config = ChaplinConfig()
    # wandb.login(key=os.getenv("WANDB_KEY"))
    wandb.init(
        project="csc413-proj",
        config=asdict(config),
        name=f"{config.algorithm}-{config.domain_name}_{config.task_name}",
        entity="scott-reseach",
        mode="disabled" if config.debug else "online",
    )
    wandb.define_metric("env_steps")
    wandb.define_metric("agent/*", step_metric="env_steps")
    wandb.define_metric("dataset/*", step_metric="env_steps")
    wandb.define_metric("ppo_training_steps")
    wandb.define_metric("ppo/*", step_metric="ppo_training_steps")
    wandb.define_metric("dreamer_training_steps")
    wandb.define_metric("dreamer/*", step_metric="dreamer_training_steps")

    # env for getting action space and observation space
    env = DMCEnv(
        config.domain_name,
        config.task_name,
        config.obs_image_size,
        action_repeat=config.action_repeats,
        render_kwargs=config.render_kwargs,
    )
    action_space = env.action_space
    action_size = np.prod(action_space.shape).item()
    obs_space = env.observation_space

    def create_env(seed: int = 0):
        return DMCEnv(
            domain_name=config.domain_name,
            task_name=config.task_name,
            seed=seed,
            render_kwargs=config.render_kwargs,
        )

    # init agent
    agent = ChaplinAgent(action_shape=action_space.shape)
    if config.load_model_path is not None:
        agent.load_state_dict(torch.load(config.load_model_path))
    chaplin = Chaplin(
        agent,
        model_lr=config.model_lr,
        action_lr=config.policy_lr,
        value_lr=config.value_lr,
        discount=config.gamma,
        discount_lambda=config.gae_lambda,
        ppo_epsilon=config.ppo_epsilon,
        ppo_value_loss_coef=config.ppo_value_loss_coef,
        ppo_entropy_coef=config.ppo_entropy_loss_coef,
        device=config.training_device
    )

    # init buffer
    buffer = ExperienceReplayDataset(
        config.data_dir,
        buffer_size=100,
        max_episode_length=1000 // config.action_repeats,
        obs_shape=obs_space.shape,
        action_size=action_size,
        batch_size=config.dreamer_batch_size,
        batch_length=config.dreamer_batch_length,
        device=config.training_device,
    )

    # init collectors
    train_collector = VectorCollector(
        VectorDMCEnv(create_env, config.train_num_envs),
        chaplin.agent,
        True,
        config.collector_device,
    )
    test_collector = VectorCollector(
        VectorDMCEnv(create_env, config.test_num_envs, config.test_env_starting_seed),
        chaplin.agent,
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
            ),
            chaplin.agent,
            False,
            config.collector_device,
        )
        prefill_data, _, (_, env_steps) = prefill_collector.collect(
            target_num_episodes=1, random_action=True
        )
        prefill_collector.close()
        prefill_data = prefill_data[0]
        taus = raw_data_to_trajectories(
            prefill_data, True, config.gamma, config.gae_lambda
        )
        for i, tau in enumerate(taus):
            buffer.add_trajectory(tau)
            tau.save(os.path.join(config.data_dir, f"pre_{i}.npz"))

    # pretrain dreamer
    if config.load_model_path is None and not config.ppo_only:
        chaplin.agent.train()
        for _ in trange(config.dreamer_training_steps, desc="Dreamer Training Steps"):
            obs, action, _, _, reward = next(buffer)[:5]
            chaplin.update_dreamer(obs, action, reward)

    current_raw_tau = None
    prev_obs, prev_action, prev_state = None, None, None
    for i in trange(config.iterations, desc="Training Iterations"):
        chaplin.agent.train()

        # collect T samples and do PPO update
        train_collector.reset_agent(chaplin.agent)
        raw, (prev_obs, prev_action, prev_state), (_, env_steps) = (
            train_collector.collect(
                target_num_steps=config.ppo_T,
                prev_obs=prev_obs,
                prev_action=prev_action,
                prev_state=prev_state,
            )
        )
        total_env_steps += env_steps * config.action_repeats * config.train_num_envs
        done = len(raw) == 2
        raw = raw[0]
        if done:
            assert current_raw_tau is not None
            current_raw_tau = merge_raw_data([current_raw_tau, raw])
            wandb.log(
                {
                    "agent/training_return": np.mean(
                        np.sum(current_raw_tau["reward"], axis=0)
                    ),
                }
            )
            finished_taus = raw_data_to_trajectories(
                current_raw_tau, True, config.gamma, config.gae_lambda
            )
            for j, tau in enumerate(finished_taus):
                buffer.add_trajectory(tau)
                threading.Thread(
                    target=lambda: tau.save(
                        os.path.join(config.data_dir, f"{i}_{j}.npz")
                    )
                ).start()
            current_raw_tau = None
        else:
            if current_raw_tau is None:
                current_raw_tau = raw
            else:
                current_raw_tau = merge_raw_data([current_raw_tau, raw])

        taus = raw_data_to_trajectories(raw, done, config.gamma, config.gae_lambda)
        dataloader = ExperienceReplayDataset(
            None,
            config.train_num_envs,
            config.ppo_T,
            obs_space.shape,
            action_size,
            config.ppo_minibatch_size,
            config.ppo_minibatch_length,
            config.training_device,
        )
        for tau in taus:
            dataloader.add_trajectory(tau)

        for _ in trange(config.ppo_epochs, desc="PPO Epochs"):
            for _ in range(config.ppo_training_steps):
                obs, action, log_prob, value, reward, rewards_to_go, advantages = next(
                    dataloader
                )
                chaplin.update_ppo(
                    obs, action, reward, value, log_prob, rewards_to_go, advantages
                )

        # dreamer update
        if not config.ppo_only:
            for _ in trange(
                config.dreamer_training_steps, desc="Dreamer Training Steps"
            ):
                obs, action, log_prob, value, reward = next(buffer)[:5]
                chaplin.update_dreamer(obs, action, reward)

        config.step(i)
        wandb.log(
            {
                "env_steps": total_env_steps,
            }
        )

        # test
        if i % config.test_every == 0 or i == config.iterations - 1:
            print("Testing...")
            torch.save(
                chaplin.agent.state_dict(),
                os.path.join(config.model_dir, f"{total_env_steps}.pt"),
            )
            test_collector.reset_agent(chaplin.agent)
            raw, _, _ = test_collector.collect(target_num_episodes=1)
            raw = raw[0]
            observations = denormalize_images(np.array(raw["obs"]))
            wandb.log(
                {
                    "agent/test_return": np.mean(np.sum(raw["reward"], axis=0)),
                    "agent/test_video": [
                        wandb.Video(observations[:, j], fps=30, format="mp4")
                        for j in range(config.test_num_envs)
                    ],
                }
            )

    train_collector.close()
    test_collector.close()


if __name__ == "__main__":
    main()
