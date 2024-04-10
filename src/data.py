import copy
import os
from typing import Dict

import numpy as np
import torch
from gymnasium.spaces import Box
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from src.envs import DMCEnv
from src.models.agent import AgentModel


def load_trajectory_data(data_path: str) -> Dict:
    data = np.load(data_path)
    return {
        "obs": data["obs"],
        "action": data["action"],
        "reward": data["reward"],
    }


class TrajectoryDataset(Dataset):
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


def get_random_action(action_spec: Box):
    minimum, maximum = action_spec.low, action_spec.high
    shape = action_spec.shape
    return np.random.uniform(minimum, maximum, shape)


class Collector:

    def __init__(self, env: DMCEnv, agent: AgentModel, device="cpu"):
        self.env = env
        self.agent = None
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
                action = action_tensor.cpu().numpy()

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
