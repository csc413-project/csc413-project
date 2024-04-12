import os
import time
import torch
import numpy as np
import wandb
from dataclasses import dataclass, asdict
import envpool
import gymnasium as gym
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from PPOAgent import PPOAgentWithRSSM, PPOAgent  # Assume we've stored our model here
from data import BatchDataset, VectorSampler, Agent
from utils import init_wandb, FrameStack, AtariPreprocessing, RecordVideo

@dataclass
class PPOConfig:
    env_name: str
    sampler: VectorSampler
    init_model: PPOAgentWithRSSM
    optimizer: optim.Optimizer
    lr_scheduler: optim.lr_scheduler.LRScheduler

    # Sampler configuration
    T: int = 128  # Horizon for trajectory rollout
    # RL hyperparameters
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    # Training hyperparameters
    lr: float = 3e-4
    lr_gamma: float = 0.98
    mini_batch_size: int = 64
    iterations: int = 1000
    epochs: int = 3  # Sample reuse
    device: str = "cuda"
    # PPO specific
    clip_epsilon: float = 0.2
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
        for i in range(self.config.iterations):
            start_time = time.time()
            print(f"========= Iteration {i} =========")
            # Collect trajectories using the current policy
            batch, prev_sample = self.sampler.sample(self.config.T, iteration=i, prev=prev_sample)
            total_env_steps += len(batch)
            wandb.log({
                "agent/batch_size": len(batch),
                "agent/total_env_steps": total_env_steps,
            }, step=i)
            dataset = BatchDataset(batch, self.config.device)
            loader = DataLoader(dataset, batch_size=self.config.mini_batch_size, shuffle=True)
            print(f"Collected batch in {time.time() - start_time:.2f}s")
            start_time = time.time()

            for e in range(self.config.epochs):
                for obses, actions, log_probs_old, rewards, values, reward_to_go, advantages in loader:
                    # Optimize policy
                    self.optimizer.zero_grad()
                    # Adapt these API calls to use the evaluate_actions method from the agent
                    pi, log_pi_a, value_preds, _ = self.model.evaluate_actions(obses, actions)
                    kl_divergences = (log_probs_old - log_pi_a).mean().item()
                    ratio = torch.exp(log_pi_a - log_probs_old)
                    clipped_ratio = torch.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)

                    pi_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
                    v_loss = ((reward_to_go - value_preds) ** 2).mean()

                    total_loss = (pi_loss
                                  + self.config.value_coeff * v_loss
                                  - self.config.entropy_beta * pi.entropy().mean())
                    
                    
                    total_loss.backward()
                    self.optimizer.step()
                    update_steps += 1
                    wandb.log({
                        "update_steps": update_steps,
                        "iteration": i,
                        "train/policy_loss": pi_loss.item(),
                        "train/entropy": pi.entropy().mean().item(),
                        "train/value_loss": v_loss.item(),
                        "train/kl_divergence": kl_divergences,
                        "train/lr": self.scheduler.get_last_lr()[0],
                    }, commit=False)
            print(f"Optimized in {time.time() - start_time:.2f}s")
            start_time = time.time()
            # Update learning rate
            self.scheduler.step()
            # Sync model to sampler
            self.sampler.reset_model(self.model)
            # Save model and test
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


def init_wandb(config):
    wandb.init(project="ppo-project", config=asdict(config), entity="user-entity")