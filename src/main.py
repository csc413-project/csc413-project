import os
from dataclasses import dataclass

import dm_control.suite as suite
import numpy as np
import torch
from dm_control.suite.wrappers import pixels
from torch.utils.data.dataset import Dataset
from tqdm import tqdm, trange

from dreamer import Dreamer
from models.agent import AgentModel


@dataclass
class DreamerConfig:
    data_dir: str = "/home/scott/tmp/dreamer/"  # where to store trajectories

    batch_size: int = 50  # batch size for training
    batch_length: int = 50  # sequence length of each training batch

    training_steps: int = 100  # number of training steps


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, batch_length, device="cuda"):
        self.data_dir = data_dir
        self.batch_length = batch_length
        self.device = device

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
            data = np.load(file_path)
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
        data = np.load(self.files[file_index])
        end_pos = start_pos + self.batch_length
        obs = data["obs"][start_pos:end_pos]
        actions = data["action"][start_pos:end_pos]
        rewards = data["reward"][start_pos:end_pos]

        return (
            torch.transpose(
                torch.tensor(obs, dtype=torch.float32, device=self.device), 1, 3
            ),
            torch.tensor(actions, dtype=torch.float32, device=self.device).view(
                self.batch_length, -1
            ),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
        )


def get_random_action(action_spec):
    minimum, maximum = action_spec.minimum, action_spec.maximum
    shape = action_spec.shape
    return np.random.uniform(minimum, maximum, shape)


def main():
    config = DreamerConfig()
    env = suite.load(domain_name="pendulum", task_name="swingup")
    env = pixels.Wrapper(env, render_kwargs=dict(height=64, width=64))
    action_spec = env.action_spec()
    # init dreamer
    agent = AgentModel(action_shape=action_spec.shape)
    agent = agent.to("cuda")
    dreamer = Dreamer(agent)
    for i in range(100):
        # collect
        data = dict(obs=[], action=[], reward=[])
        time_step = env.reset()
        step_type, reward, discount, obs = time_step
        obs = obs["pixels"]
        num_episodes, env_steps = 0, 0
        pbar = tqdm(leave=False)
        prev_action, prev_state = None, None
        while num_episodes < 1 and env_steps < 2000:
            with torch.no_grad():
                obs_tensor = torch.unsqueeze(
                    torch.transpose(
                        torch.tensor(obs.copy(), dtype=torch.float32, device="cuda"),
                        0,
                        2,
                    ),
                    0,
                )
                action, action_dist, value, i_reward, state = dreamer.agent(
                    obs_tensor, prev_action, prev_state
                )
            time_step = env.step(action.cpu())
            step_type, reward, discount, next_obs = time_step

            if time_step.last():
                step_type, reward, discount, next_obs = env.reset()
                num_episodes += 1
            else:
                data["obs"].append(obs)
                data["action"].append(action.cpu().numpy())
                data["reward"].append(reward)

            obs = next_obs["pixels"]
            prev_action = action
            prev_state = state
            env_steps += 1
            pbar.update(1)
        pbar.close()
        np.savez_compressed(os.path.join(config.data_dir, f"{i}.npz"), **data)
        print("Episode Return: ", sum(data["reward"]))

        # train
        dataset = TrajectoryDataset(config.data_dir, config.batch_length)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )
        data_iter = iter(dataloader)
        for _ in trange(config.training_steps):
            try:
                obs, action, reward = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                obs, action, reward = next(data_iter)
            dreamer.update(obs, action, reward)


def debug():
    config = DreamerConfig()
    env = suite.load(domain_name="reacher", task_name="easy")
    env = pixels.Wrapper(env, render_kwargs=dict(height=64, width=64))
    action_spec = env.action_spec()
    # init dreamer
    agent = AgentModel(action_shape=action_spec.shape)
    dreamer = Dreamer(agent)

    dataset = TrajectoryDataset(config.data_dir, config.batch_length)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )
    data_iter = iter(dataloader)
    for _ in range(config.training_steps):
        obs, action, reward = next(data_iter)
        dreamer.update(obs, action, reward)


if __name__ == "__main__":
    main()
    # debug()
