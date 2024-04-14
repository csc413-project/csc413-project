import copy
import os
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import wandb
from gymnasium.spaces import Box
from tqdm import tqdm

from envs import DMCEnv, VectorDMCEnv
from models.agent import ChaplinAgent
from utils import discounted_cumulative_sum


def merge_raw_data(raw_data: List[Dict]) -> Dict:
    merged = {}
    for key in raw_data[0].keys():
        merged[key] = np.concatenate([data[key] for data in raw_data])
    return merged


class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    done: bool

    rewards_to_go: Optional[np.ndarray]
    advantages: Optional[np.ndarray]

    def __init__(self, tau: Dict, done: bool):
        self.done = done

        self.observations = np.asarray(tau["obs"])
        self.actions = np.asarray(tau["action"])
        # self.log_probs = np.asarray(tau["log_prob"])
        self.log_probs = np.asarray(tau["log_prob"]).flatten()
        self.rewards = np.asarray(tau["reward"]).flatten()
        self.values = np.asarray(tau["value"]).flatten()
        if "rewards_to_go" in tau:
            self.rewards_to_go = np.asarray(tau["rewards_to_go"]).flatten()
        else:
            self.rewards_to_go = None
        if "advantages" in tau:
            self.advantages = np.asarray(tau["advantages"]).flatten()
        else:
            self.advantages = None
        # sanity check: they should have the same length
            
        assert (
            len(self.observations)
            == len(self.actions)
            == len(self.log_probs)
            == len(self.rewards)
            == len(self.values)
        )
 
    def compute_advantages(self, gamma: float, gae_lambda: float) -> None:
        if self.advantages is not None and self.rewards_to_go is not None:
            return
        if self.done:
            self.rewards_to_go = discounted_cumulative_sum(self.rewards, gamma)
            # note that we append 0 here and this might introduce bias
            values = np.append(self.values, 0)
        else:
            # bootstrap the rewards-to-go to include the value of the last state
            self.rewards_to_go = discounted_cumulative_sum(
                np.append(self.rewards, self.values[-1]), gamma
            )[:-1]
            values = np.append(
                self.values, self.values[-1]
            )  # bootstrap the value of the last state

        deltas = self.rewards + gamma * values[1:] - values[:-1]
        self.advantages = discounted_cumulative_sum(deltas, gamma * gae_lambda)
        self.advantages = (self.advantages - np.mean(self.advantages)) / (
            np.std(self.advantages) + 1e-10
        )

    def compute_episode_return(self) -> float:
        return np.sum(self.rewards).item()

    def __len__(self):
        return self.observations.shape[0]

    def save(self, data_path: str):
        np.savez_compressed(
            data_path,
            obs=self.observations,
            action=self.actions,
            value=self.values,
            log_prob=self.log_probs,
            reward=self.rewards,
            rewards_to_go=self.rewards_to_go,
            advantages=self.advantages,
        )


def load_trajectory_data(data_path: str) -> Trajectory:
    data = np.load(data_path)
    return Trajectory(
        {
            "obs": data["obs"],
            "action": data["action"],
            "value": data["value"],
            "log_prob": data["log_prob"],
            "reward": data["reward"],
            "rewards_to_go": data["rewards_to_go"],
            "advantages": data["advantages"],
        },
        True,
    )


def raw_data_to_trajectories(
    raw: Dict, done: bool, gamma: float, gae_lambda: float
) -> List[Trajectory]:
    num_traj = raw["obs"][0].shape[0]
    for key in raw:
        raw[key] = np.asarray(raw[key])
    taus = []
    for i in range(num_traj):
        tau = Trajectory(
            {
                "obs": raw["obs"][:, i],
                "action": raw["action"][:, i],
                "value": raw["value"][:, i],
                "log_prob": raw["log_prob"][:, i],
                "reward": raw["reward"][:, i],
            },
            done,
        )
        tau.compute_advantages(gamma, gae_lambda)
        taus.append(tau)
    return taus


class ExperienceReplayDataset:
    """
    A dataset to store and sample trajectories for training.

    100 episodes of 1000 steps each will take up approximately 9GB of memory.
    """

    def __init__(
        self,
        episode_dir: Optional[str],
        buffer_size: int,
        max_episode_length: int,
        obs_shape: Tuple[int, ...],
        action_size: int,
        batch_size: int,
        batch_length: int,
        device: str = "cuda",
    ):
        self.episode_dir = episode_dir
        self.buffer_size = buffer_size
        self.max_episode_length = max_episode_length
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.device = device

        self.num_episodes = 0
        self.oldest_index = 0
        self.episode_lengths = np.zeros(buffer_size, dtype=np.int32)
        self.observations = np.empty(
            (buffer_size, max_episode_length, *obs_shape), dtype=np.float32
        )
        self.actions = np.empty(
            (buffer_size, max_episode_length, action_size), dtype=np.float32
        )
        self.values = np.empty((buffer_size, max_episode_length), dtype=np.float32)
        # self.log_prob = np.empty((buffer_size, max_episode_length, action_size), dtype=np.float32)
        self.log_prob = np.empty((buffer_size, max_episode_length), dtype=np.float32)
        self.rewards = np.empty((buffer_size, max_episode_length), dtype=np.float32)
        self.rewards_to_go = np.empty(
            (buffer_size, max_episode_length), dtype=np.float32
        )
        self.advantages = np.empty((buffer_size, max_episode_length), dtype=np.float32)

        # load all episodes if exists
        # list all files, sorted by modification time, oldest first
        if episode_dir is not None:
            files = sorted(
                [
                    os.path.join(episode_dir, f)
                    for f in os.listdir(episode_dir)
                    if f.endswith(".npz")
                ],
                key=lambda x: os.path.getmtime(x),
            )
            for file in files[-buffer_size:]:
                self.add_trajectory(load_trajectory_data(file))

    def add_trajectory(self, tau: Trajectory):
        tau_length = len(tau)
        assert tau_length <= self.max_episode_length
        assert tau.rewards_to_go is not None and tau.advantages is not None

        self.episode_lengths[self.oldest_index] = tau_length
        self.observations[self.oldest_index, :tau_length] = tau.observations
        self.actions[self.oldest_index, :tau_length] = tau.actions
        self.rewards[self.oldest_index, :tau_length] = tau.rewards
        self.values[self.oldest_index, :tau_length] = tau.values
        self.log_prob[self.oldest_index, :tau_length] = tau.log_probs
        self.rewards_to_go[self.oldest_index, :tau_length] = tau.rewards_to_go
        self.advantages[self.oldest_index, :tau_length] = tau.advantages

        self.oldest_index = (self.oldest_index + 1) % self.buffer_size
        self.num_episodes = min(self.num_episodes + 1, self.buffer_size)

        wandb.log(
            {
                "dataset/num_episodes": self.num_episodes,
                "dataset/oldest_index": self.oldest_index,
            }
        )

    def __iter__(self):
        return self

    def __next__(self):
        # sample batch_size trajectory indices
        i1 = np.random.randint(0, self.num_episodes, self.batch_size)
        lengths = (self.episode_lengths[i1] - self.batch_length).astype(int)
        # sample batch_size start indices
        starts = np.random.randint(0, lengths)

        obs = np.zeros(
            (self.batch_size, self.batch_length, *self.obs_shape), dtype=np.float32
        )
        actions = np.zeros(
            (self.batch_size, self.batch_length, self.action_size), dtype=np.float32
        )
        log_prob = np.zeros(
            (self.batch_size, self.batch_length), dtype=np.float32
        )
        values = np.zeros((self.batch_size, self.batch_length), dtype=np.float32)
        rewards = np.zeros((self.batch_size, self.batch_length), dtype=np.float32)
        rewards_to_go = np.zeros((self.batch_size, self.batch_length), dtype=np.float32)
        advantages = np.zeros((self.batch_size, self.batch_length), dtype=np.float32)

        for i, episode_index in enumerate(i1):
            start = starts[i]
            end = start + self.batch_length
            obs[i] = self.observations[episode_index, start:end]
            actions[i] = self.actions[episode_index, start:end]
            log_prob[i] = self.log_prob[episode_index, start:end]
            values[i] = self.values[episode_index, start:end]
            rewards[i] = self.rewards[episode_index, start:end]
            rewards_to_go[i] = self.rewards_to_go[episode_index, start:end]
            advantages[i] = self.advantages[episode_index, start:end]

        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self.device),
            torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            torch.as_tensor(log_prob, dtype=torch.float32, device=self.device),
            torch.as_tensor(values, dtype=torch.float32, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(rewards_to_go, dtype=torch.float32, device=self.device),
            torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
        )


def get_random_action(action_spec: Box):
    minimum, maximum = action_spec.low, action_spec.high
    shape = action_spec.shape
    return np.random.uniform(minimum, maximum, shape)


class Collector:
    def __init__(self, env: DMCEnv, agent: ChaplinAgent, explore: bool, device="cpu"):
        self.env = env
        self.agent = None
        self.explore = explore
        self.device = device

        self.reset_agent(agent)

    @torch.no_grad()
    def collect(
        self,
        target_num_episodes: int = 1,
        target_num_steps: int = -1,
        obs=None,
        prev_action=None,
        prev_state=None,
        random_action=False,
    ):
        assert (target_num_episodes > 0 or target_num_steps > 0) and (
            target_num_episodes < 0 or target_num_steps < 0
        ), "Only one of num_episodes or num_steps should be greater than 0"
        assert self.agent.training is False, "Agent should be in eval mode"
        assert self.agent.explore is self.explore

        env = self.env
        agent = self.agent
        data = [dict(obs=[], action=[], value=[], log_prob=[], reward=[])]

        if obs is None:
            obs = env.reset()
        num_episodes, num_steps = 0, 0
        pbar = tqdm(desc="Collecting Data", leave=True)
        while num_episodes < target_num_episodes or num_steps < target_num_steps:
            if random_action:
                action = get_random_action(env.action_space)
                log_prob = None
                state = None
                action_tensor = action
                value = 0
            else:
                obs_tensor = torch.unsqueeze(
                    torch.tensor(obs.copy(), dtype=torch.float32, device=self.device),
                    0,
                )  # assume normalized obs
                action_tensor, action_dist, value, i_reward, state = agent(
                    obs_tensor, prev_action, prev_state
                )
                action = action_tensor.cpu().flatten().numpy()
                action_clipped = torch.clamp(
                    action_tensor, min=-0.99, max=0.99
                )  # Adjust the range based on your specific action space

                # Calculate log probabilities safely
                log_prob = action_dist.log_prob(action_clipped).cpu().numpy()

                value = value.mean.cpu().numpy()[0][0]

            next_obs, reward, done, _ = env.step(action)

            if done:
                next_obs = env.reset()
                num_episodes += 1
                data.append(dict(obs=[], action=[], value=[], log_prob=[], reward=[]))
            else:
                data[num_episodes]["obs"].append(obs)
                data[num_episodes]["action"].append(action)
                data[num_episodes]["log_prob"].append(log_prob)
                data[num_episodes]["value"].append(value)
                data[num_episodes]["reward"].append(reward)

            obs = next_obs
            prev_action = action_tensor
            prev_state = state
            num_steps += 1
            pbar.update(1)
        pbar.close()

        if not data[-1]["obs"]:
            data.pop(-1)
        # return additional information for resuming
        return data, (obs, prev_action, prev_state), (num_episodes, num_steps)

    def reset_agent(self, new_agent: ChaplinAgent):
        self.agent = copy.deepcopy(new_agent)
        self.agent = self.agent.to(self.device)
        self.agent.eval()
        self.agent.explore = self.explore


class VectorCollector:
    def __init__(
        self, env: VectorDMCEnv, agent: ChaplinAgent, explore: bool, device="cpu"
    ):
        self.envs = env
        self.agent = None
        self.explore = explore
        self.device = device

        self.reset_agent(agent)

    @torch.no_grad()
    def collect(
        self,
        target_num_episodes: int = -1,
        target_num_steps: int = -1,
        prev_obs=None,
        prev_action=None,
        prev_state=None,
        random_action=False,
    ):
        """
        Collects data from the environment using the agent.
        :param target_num_episodes:
        :param target_num_steps:
        :param obs:
        :param prev_action:
        :param prev_state:
        :param random_action:
        :return: the num_episodes and num_steps are for single env
        """
        assert (target_num_episodes > 0 or target_num_steps > 0) and (
            target_num_episodes < 0 or target_num_steps < 0
        ), "Only one of num_episodes or num_steps should be greater than 0"
        assert self.agent.training is False, "Agent should be in eval mode"

        envs = self.envs
        agent = self.agent
        data = [dict(obs=[], action=[], log_prob=[], value=[], reward=[])]

        if prev_obs is None:
            obs = envs.reset()
        else:
            obs = prev_obs

        num_episodes, num_steps = 0, 0
        pbar = tqdm(desc="Collecting Data", leave=False)
        while num_episodes < target_num_episodes or num_steps < target_num_steps:
            if random_action:
                # Get a random action for each environment
                action = np.random.uniform(
                    envs.action_space.low,
                    envs.action_space.high,
                    (envs.num_envs,) + envs.action_space.shape,
                )

                log_prob = np.zeros(
                    (envs.num_envs, envs.action_space.shape[0])
                )  # Properly shaped as two-dimensional
                value = np.zeros((envs.num_envs,))  # Also two-dimensional

                state = None
                action_tensor = action
            else:
                obs_tensor = torch.tensor(
                    obs.copy(), dtype=torch.float32, device=self.device
                )
                action_tensor, action_dist, value, i_reward, state = agent(
                    obs_tensor, prev_action, prev_state
                )

                log_prob = action_dist.log_prob(action_tensor).cpu().numpy()
                action = action_tensor.cpu().numpy()
                value = value.cpu().numpy()

            next_obs, reward, done, _ = envs.step(action)

            if done.all():  # assume all envs are done at the same time
                next_obs = envs.reset()
                num_episodes += 1
                data.append(dict(obs=[], action=[], value=[], log_prob=[], reward=[]))

            else:
                data[num_episodes]["obs"].append(obs)
                data[num_episodes]["action"].append(action)
                data[num_episodes]["log_prob"].append(log_prob)
                data[num_episodes]["value"].append(value)
                data[num_episodes]["reward"].append(reward)

            obs = next_obs
            prev_action = action_tensor
            prev_state = state
            num_steps += 1
            pbar.update(1)
        pbar.close()

        # return additional information for resuming
        return data, (obs, prev_action, prev_state), (num_episodes, num_steps)

    def reset_agent(self, new_agent: ChaplinAgent):
        self.agent = copy.deepcopy(new_agent)
        self.agent = self.agent.to(self.device)
        self.agent.eval()
        self.agent.explore = self.explore

    def close(self):
        self.envs.close()
