import copy
import os
from typing import Dict, Tuple

import numpy as np
import torch
import wandb
from gymnasium.spaces import Box
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from envs import DMCEnv, VectorDMCEnv
from models.agent import AgentModel


def load_trajectory_data(data_path: str) -> Dict:
    data = np.load(data_path)
    return {
        "obs": data["obs"],
        "action": data["action"],
        "reward": data["reward"],
    }


class TrajectoryIndexDataset(Dataset):
    def __init__(self, data_dir, batch_length, device="cpu"):
        self.data_dir = data_dir
        self.batch_length = batch_length
        self.device = device
        self.cache = {}

        # Load all file names
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ]
        self.indexes = self._create_indexes()

    def _create_indexes(self):
        # Creates a list of tuples (file_index, start_pos) for sampling
        indexes = []
        for file_index, file_path in enumerate(self.files):
            if file_index < 10:
                self.cache[file_index] = load_trajectory_data(file_path)
                data = self.cache[file_index]
            else:
                data = load_trajectory_data(file_path)
            num_samples = len(data["obs"])
            if num_samples >= self.batch_length:
                for start_pos in range(num_samples - self.batch_length + 1):
                    indexes.append((file_index, start_pos))
        return indexes

    def __len__(self):
        # Returns the total number of possible segments
        return len(self.indexes)

    def __getitem__(self, idx):
        # Fetches a random segment from the dataset
        file_index, start_pos = self.indexes[idx]
        if file_index not in self.cache:
            self.cache[file_index] = load_trajectory_data(self.files[file_index])
            if len(self.cache) > 10:
                self.cache.pop(list(self.cache.keys())[0])
        data = self.cache[file_index]
        end_pos = start_pos + self.batch_length
        obs = data["obs"][start_pos:end_pos]
        actions = data["action"][start_pos:end_pos]
        rewards = data["reward"][start_pos:end_pos]
        return (
            torch.tensor(obs, dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.float32, device=self.device).view(
                self.batch_length, -1
            ),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
        )


class ExperienceReplayDataset:
    """
    A dataset to store and sample trajectories for training.

    100 episodes of 1000 steps each will take up approximately 9GB of memory.
    """

    def __init__(
        self,
        episode_dir: str,
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
        self.rewards = np.empty((buffer_size, max_episode_length), dtype=np.float32)

        # load all episodes if exists
        # list all files, sorted by modification time, oldest first
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

    def add_trajectory(self, data: Dict):
        obs, actions, rewards = data["obs"], data["action"], data["reward"]
        episode_length = len(obs)
        assert episode_length == len(actions) == len(rewards)
        assert episode_length <= self.max_episode_length

        self.episode_lengths[self.oldest_index] = episode_length
        self.observations[self.oldest_index, :episode_length] = np.asarray(obs)
        self.actions[self.oldest_index, :episode_length] = np.asarray(actions)
        self.rewards[self.oldest_index, :episode_length] = np.asarray(rewards)
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
        rewards = np.zeros((self.batch_size, self.batch_length), dtype=np.float32)

        for i, episode_index in enumerate(i1):
            start = starts[i]
            end = start + self.batch_length
            obs[i] = self.observations[episode_index, start:end]
            actions[i] = self.actions[episode_index, start:end]
            rewards[i] = self.rewards[episode_index, start:end]

        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self.device),
            torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
        )


def get_random_action(action_spec: Box):
    minimum, maximum = action_spec.low, action_spec.high
    shape = action_spec.shape
    return np.random.uniform(minimum, maximum, shape)


class Collector:
    def __init__(self, env: DMCEnv, agent: AgentModel, explore: bool, device="cpu"):
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
        data = [dict(obs=[], action=[], reward=[])]

        if obs is None:
            obs = env.reset()
        num_episodes, num_steps = 0, 0
        pbar = tqdm(desc="Collecting Data", leave=True)
        while num_episodes < target_num_episodes or num_steps < target_num_steps:
            if random_action:
                action = get_random_action(env.action_space)
                state = None
                action_tensor = action
            else:
                obs_tensor = torch.unsqueeze(
                    torch.tensor(obs.copy(), dtype=torch.float32, device=self.device),
                    0,
                )  # assume normalized obs
                action_tensor, action_dist, value, i_reward, state = agent(
                    obs_tensor, prev_action, prev_state
                )
                action = action_tensor.cpu().flatten().numpy()

            next_obs, reward, done, _ = env.step(action)

            if done:
                next_obs = env.reset()
                num_episodes += 1
                data.append(dict(obs=[], action=[], reward=[]))
            else:
                data[num_episodes]["obs"].append(obs)
                data[num_episodes]["action"].append(action)
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

    def reset_agent(self, new_agent: AgentModel):
        self.agent = copy.deepcopy(new_agent)
        self.agent = self.agent.to(self.device)
        self.agent.eval()
        self.agent.explore = self.explore


class VectorCollector:
    def __init__(
        self, env: VectorDMCEnv, agent: AgentModel, explore: bool, device="cpu"
    ):
        self.envs = env
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
        data = [dict(obs=[], action=[], reward=[])]

        if obs is None:
            obs = envs.reset()

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

                state = None
                action_tensor = action
            else:
                obs_tensor = torch.tensor(
                    obs.copy(), dtype=torch.float32, device=self.device
                )
                action_tensor, action_dist, value, i_reward, state = agent(
                    obs_tensor, prev_action, prev_state
                )
                action = action_tensor.cpu().numpy()

            next_obs, reward, done, _ = envs.step(action)

            if done.all():
                next_obs = envs.reset()
                num_episodes += 1
                data.append(dict(obs=[], action=[], reward=[]))

            else:
                data[num_episodes]["obs"].append(obs)
                data[num_episodes]["action"].append(action)
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

    def reset_agent(self, new_agent: AgentModel):
        self.agent = copy.deepcopy(new_agent)
        self.agent = self.agent.to(self.device)
        self.agent.eval()
        self.agent.explore = self.explore

    def close(self):
        self.envs.close()
