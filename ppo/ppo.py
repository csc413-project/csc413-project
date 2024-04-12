import os
import time
from dataclasses import dataclass

import envpool
import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStack
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
from torch.utils.data import DataLoader

from data import BatchDataset, VectorSampler, Agent
from network import AbstractModel, CategoricalCNNModel
from utils import init_wandb


@dataclass
class PPOConfig:
    env_name: str

    sampler: VectorSampler
    init_model: AbstractModel
    optimizer: optim.Optimizer
    lr_scheduler: LRScheduler

    # sampler config
    T: int = 128  # horizon
    # rl hyperparameters
    discount_gamma: float = 0.95
    gae_lambda: float = 0.97
    # training hyperparameters
    lr: float = 3e-4
    lr_gamma: float = 0.98
    mini_batch_size: int = 5
    iterations: int = 200
    epochs: int = 1  # sample reuse
    device: str = "cuda"
    # PPO specific
    clip_epsilon: float = 0.2  # clip ratio
    entropy_beta: float = 0.01
    value_coeff: float = 1.0

    def to_dict(self):
        return {
            "env_name": self.env_name,
            "steps_per_iter": self.T,
            "gamma": self.discount_gamma,
            "lambda_": self.gae_lambda,
            "policy_lr": self.lr,
            "mini_batch_size": self.mini_batch_size,
            "iterations": self.iterations,
            "epochs": self.epochs,
            "clip_epsilon": self.clip_epsilon,
            "entropy_beta": self.entropy_beta,
            "value_coeff": self.value_coeff,
        }


class PPO:

    def __init__(self, config: PPOConfig):
        self.config = config
        self.sampler = config.sampler
        self.model = config.init_model.to(config.device)
        self.optimizer = config.optimizer
        self.scheduler = config.lr_scheduler

    def train(self):
        self.model.train()
        wandb.define_metric("iteration")
        wandb.define_metric("update_steps")
        wandb.define_metric("train/*", step_metric="update_steps")
        wandb.define_metric("agent/*", step_metric="iteration")
        update_steps, total_env_steps = 0, 0
        policy_losses, entropy, kl_divergences, value_losses = [], [], [], []
        prev_sample = None
        for i in range(self.config.iterations):
            start_time = time.time()
            print(f"========= Iteration {i} =========")
            # collect trajectories using the current policy
            batch, prev_sample = self.sampler.sample(self.config.T, iteration=i, prev=prev_sample)
            total_env_steps += len(batch)
            wandb.log({
                # "agent/episode_return": batch.compute_episode_return(),
                # "agent/episode_length": batch.compute_episode_length(),
                "agent/batch_size": len(batch),
                "agent/total_env_steps": total_env_steps,
            }, step=i)
            dataset = BatchDataset(batch, self.config.device)
            loader = DataLoader(dataset, batch_size=self.config.mini_batch_size, shuffle=True)
            print(f"Collected batch in {time.time() - start_time:.2f}s")
            start_time = time.time()

            for e in range(self.config.epochs):
                for obses, actions, log_probs_old, rewards, values, reward_to_go, advantages in loader:
                    # optimize policy
                    self.optimizer.zero_grad()
                    pi, log_pi_a = self.model.forward_policy(obses, actions)
                    kl_divergences.append((log_probs_old - log_pi_a).mean().item())
                    ratio = torch.exp(log_pi_a - log_probs_old)
                    clipped_ratio = torch.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                    pi_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
                    policy_losses.append(pi_loss.item())
                    entropy_bonus = pi.entropy().mean()
                    entropy.append(entropy_bonus.item())
                    # optimize value
                    v_loss = ((reward_to_go - self.model.forward_value(obses)) ** 2).mean()
                    value_losses.append(v_loss.item())

                    total_loss = (pi_loss
                                  + self.config.value_coeff * v_loss
                                  - self.config.entropy_beta * entropy_bonus)
                    total_loss.backward()
                    self.optimizer.step()

                    update_steps += 1
                    wandb.log({
                        "update_steps": update_steps,
                        "iteration": i,
                        "train/policy_loss": pi_loss.item(),
                        "train/entropy": entropy_bonus.item(),
                        "train/value_loss": v_loss.item(),
                        "train/kl_divergence": kl_divergences[-1],
                        "train/lr": self.scheduler.get_last_lr()[0],
                    }, commit=False)
            print(f"Optimized in {time.time() - start_time:.2f}s")
            start_time = time.time()
            # update learning rate
            self.scheduler.step()
            # sync model to sampler
            self.sampler.reset_model(self.model)
            # save model and test
            if i % 10 == 0:
                os.makedirs(f"models/{self.config.env_name}", exist_ok=True)
                os.makedirs(f"videos/{self.config.env_name}", exist_ok=True)
                torch.save(self.model.state_dict(), f"models/{self.config.env_name}/ppo_{i}.pt")
                test_taus = self.sampler.sample_episodes(2, f"videos/{self.config.env_name}/{i}.mp4")
                wandb.log({
                    "agent/episode_return": np.mean([tau.compute_episode_return() for tau in test_taus]),
                    "agent/episode_length": np.mean([len(tau) for tau in test_taus]),
                }, step=i)
                print(f"Saved model and Tested in {time.time() - start_time:.2f}s")


def main():
    env_name = "Breakout-v5"
    # env_name = "Pong-v5"
    # env_name = "BattleZone-v5"
    # env_name = "SpaceInvaders-v5"
    env = envpool.make(env_name, "gymnasium")
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.n
    print(f"Environment: {env_name} | obs_shape: {obs_shape} | act_shape: {act_shape.item()}")

    model = CategoricalCNNModel(4, act_shape.item())
    model.load_state_dict(torch.load("models/Breakout-v5/ppo_1990.pt"))

    policy_lr = 2.5e-4
    optimizer = optim.Adam(model.parameters(), lr=policy_lr)
    lr_gamma = 0.9995
    scheduler = ExponentialLR(optimizer, lr_gamma)

    def recorder_env_fn():
        return RecordVideo(
            FrameStack(
                AtariPreprocessing(
                    gym.make("ALE/" + env_name, render_mode="rgb_array", frameskip=1),
                    grayscale_obs=True),
                num_stack=4
            ),
            video_folder=f"videos/{env_name}", episode_trigger=lambda e: True, disable_logger=True)

    recorder = Agent(
        recorder_env_fn,
        model,
        device="cpu"
    )

    discount_gamma = 0.99
    gae_lambda = 0.95
    sampler = VectorSampler(
        env_name,
        model,
        discount_gamma,
        gae_lambda,
        # video_recorder=recorder,
        num_envs=8,
        num_parallel=8,
    )

    config = PPOConfig(
        env_name=env_name,
        sampler=sampler,
        init_model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        T=128,
        discount_gamma=discount_gamma,
        gae_lambda=gae_lambda,
        lr=policy_lr,
        lr_gamma=lr_gamma,
        mini_batch_size=32 * 8,
        iterations=10_000,
        epochs=3,
        clip_epsilon=0.1,
        entropy_beta=0.01,
    )
    init_wandb(f"ppo-{env_name}", **config.to_dict())

    ppo = PPO(config)
    ppo.train()

    wandb.finish()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
