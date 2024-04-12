import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Callable, Optional

import cv2
import envpool
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from torch.utils.data import Dataset
from tqdm import trange, tqdm

from network import AbstractModel
from utils import discounted_cumulative_sum


class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    done: bool

    rewards_to_go: Optional[np.ndarray]
    advantages: Optional[np.ndarray]

    def __init__(self, tau: List[Tuple]):
        """
        :param tau: contains data of this trajectory
            - t[0]: observation
            - t[1]: action
            - t[2]: log_prob of the action
            - t[3]: reward
            - t[4]: value
        """
        self.done = tau[-1][1] is None  # last action is None -> done
        if self.done:
            tau.pop()  # drop the last sample
        tau_len = len(tau)
        obs_shape = tau[0][0].shape
        self.observations = np.zeros((tau_len, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((tau_len,), dtype=np.float32)
        self.log_probs = np.zeros((tau_len,), dtype=np.float32)
        self.rewards = np.zeros((tau_len,), dtype=np.float32)
        self.values = np.zeros((tau_len,), dtype=np.float32)
        for t, (obs, action, log_prob, reward, value) in enumerate(tau):
            self.observations[t] = obs
            self.actions[t] = action
            self.log_probs[t] = log_prob
            self.rewards[t] = reward
            self.values[t] = value
        self.rewards_to_go = None
        self.advantages = None
        # sanity check: they should have the same length
        assert (len(self.observations)
                == len(self.actions)
                == len(self.log_probs)
                == len(self.rewards)
                == len(self.values))

    def compute_advantages(self, gamma: float, lambda_: float) -> None:
        if self.done:
            self.rewards_to_go = discounted_cumulative_sum(self.rewards, gamma)
            # note that we append 0 here and this might introduce bias
            values = np.append(self.values, 0)
        else:
            # bootstrap the rewards-to-go to include the value of the last state
            self.rewards_to_go = discounted_cumulative_sum(
                np.append(self.rewards, self.values[-1]), gamma)[:-1]
            values = np.append(self.values, self.values[-1])  # bootstrap the value of the last state

        deltas = self.rewards + gamma * values[1:] - values[:-1]
        self.advantages = discounted_cumulative_sum(deltas, gamma * lambda_)
        self.advantages = (self.advantages - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-10)

    def compute_episode_return(self) -> float:
        return np.sum(self.rewards).item()

    def record_video(self, filename: str) -> None:
        observations = np.expand_dims(self.observations[:, 1, :, :], axis=-1)
        save_video(observations, filename)

    def __len__(self):
        return self.observations.shape[0]


class Batch:
    taus: List[Trajectory]

    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    rewards_to_go: np.ndarray
    advantages: np.ndarray

    def __init__(self, taus: List[Trajectory]):
        # sanity check: rewards to go and advantages should be computed
        for tau in taus:
            assert tau.rewards_to_go is not None
            assert tau.advantages is not None
        self.taus = taus

        # normalize the observations to [0, 1]
        self.observations = np.concatenate([tau.observations for tau in taus], axis=0) / 255.0
        assert self.observations.min() >= 0 and self.observations.max() <= 1
        self.actions = np.concatenate([tau.actions for tau in taus], axis=0)
        self.log_probs = np.concatenate([tau.log_probs for tau in taus], axis=0)
        self.rewards = np.concatenate([tau.rewards for tau in taus], axis=0)
        self.values = np.concatenate([tau.values for tau in taus], axis=0)
        self.rewards_to_go = np.concatenate([tau.rewards_to_go for tau in taus], axis=0)
        self.advantages = np.concatenate([tau.advantages for tau in taus], axis=0)
        # sanity check: they should have the same length
        batch_size = sum(len(tau) for tau in taus)
        assert (batch_size
                == len(self.observations)
                == len(self.actions)
                == len(self.log_probs)
                == len(self.rewards)
                == len(self.values)
                == len(self.rewards_to_go)
                == len(self.advantages))

    def __len__(self):
        return self.observations.shape[0]

    def compute_episode_return(self) -> float:
        return np.mean([np.sum(tau.rewards) for tau in self.taus]).item()

    def compute_episode_length(self) -> float:
        return np.mean([len(tau) for tau in self.taus]).item()


class BatchDataset(Dataset):

    def __init__(self, batch: Batch, device: str = "cuda"):
        self.device = device
        # convert batch to tensors
        self.observations = torch.as_tensor(batch.observations)
        self.actions = torch.as_tensor(batch.actions)
        self.log_probs = torch.as_tensor(batch.log_probs)
        self.rewards = torch.as_tensor(batch.rewards)
        self.values = torch.as_tensor(batch.values)
        self.rewards_to_go = torch.as_tensor(batch.rewards_to_go)
        self.advantages = torch.as_tensor(batch.advantages)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return (
            self.observations[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.log_probs[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.values[idx].to(self.device),
            self.rewards_to_go[idx].to(self.device),
            self.advantages[idx].to(self.device)
        )


class Agent:
    env_func: Callable[[], gym.Env]
    env: gym.Env
    model: AbstractModel
    device: str

    def __init__(self, env_func: Callable[[], gym.Env], model: AbstractModel, device: str = "cuda"):
        self.env_func = env_func
        self.model = model
        self.device = device

        self.env = env_func()

    def run(self) -> Trajectory:
        self.model.eval()
        # self.model = self.model.to(self.device)
        env = self.env
        tau = []
        obs, info = env.reset()
        while True:
            obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
            action, value, log_pi_a = self.model.step(obs)
            action, value, log_pi_a = action.item(), value.item(), log_pi_a.item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            tau.append((obs, action, log_pi_a, reward, value))
            obs = next_obs
            if terminated or truncated:
                break
        return Trajectory(tau)

    def reset_model(self, model: AbstractModel) -> None:
        self.model = model

    def reset_env_func(self, env_func: Callable[[], gym.Env]) -> None:
        self.env_func = env_func


class VectorSampler:

    def __init__(self,
                 env_name: str,
                 model: AbstractModel,
                 gamma: float,
                 lambda_: float,
                 video_recorder: Optional[Agent] = None,
                 device: str = "cpu",
                 **kwargs):
        self.env_name = env_name
        self.num_envs = kwargs.get("num_envs", 1)
        self.num_parallel = kwargs.get("num_parallel", 1)
        kwargs.pop("num_parallel", None)
        kwargs["batch_size"] = self.num_parallel
        self.env_func = lambda: envpool.make(env_name, "gymnasium", seed=int(time.time() // 1000), **kwargs)
        self.env = self.env_func()  # training envs
        self.model = model
        self.gamma = gamma
        self.lambda_ = lambda_
        self.video_recorder = video_recorder
        self.device = device

        self.reset_model(model)
        self.model.eval()

        self.env.async_reset()  # send the initial reset signal to all envs

    def sample(self, T: int, iteration: int = -1,
               prev: Optional[Tuple[np.ndarray, ...]] = None) -> Tuple[Batch, Tuple[np.ndarray, ...]]:
        env = self.env
        env_batches = [[[]] for _ in range(self.num_envs)]
        if prev is None:
            # get the very first observation
            obs, reward, terminated, truncated, info = env.recv()
            env_ids = info["env_id"]
        else:
            obs, env_ids = prev
        for _ in trange(int(T * (self.num_envs / self.num_parallel)),
                        desc=f"Collecting {T} steps ({self.num_envs} envs | {self.num_parallel} agents)"):
            # plot_obs(obs[0][-1])
            action, value, log_pi_a = self.model.step(obs)
            # action, value, log_pi_a = np.random.randint(0, env.action_space.n, (self.num_envs,)), np.random.randn(self.num_envs, ), np.random.randn(self.num_envs, )
            # send the action to the env
            env.send(action, env_ids)
            # get the next observation
            next_obs, reward, terminated, truncated, info = env.recv()
            env_ids = info["env_id"]
            done = terminated | truncated
            # store the data in the corresponding env batch
            for i, env_id in enumerate(env_ids):
                env_batches[env_id][-1].append((obs[i], action[i], log_pi_a[i], reward[i], value[i]))
                if done[i]:
                    env_batches[env_id][-1].append((next_obs[i], None, None, 0, 0))  # utilize the last obs
                    env_batches[env_id].append([])
            # update the observation
            obs = next_obs
        # post-process the data
        # drop the first sample in every envs and trajectories, except for the first trajectories
        for env_batch in env_batches:
            for i in range(1, len(env_batch)):
                if not env_batch[i]:
                    continue
                env_batch[i].pop(0)
        # also drop the last trajectory in every envs if not terminated
        # for env_batch in env_batches:
        #     if not env_batch[-1] or env_batch[-1][-1][1] is not None:
        #         env_batch.pop()
        if iteration % 10 == 0 and self.video_recorder is not None:
            test_tau = self.video_recorder.run()
            # wandb.log({
            #     "agent/episode_return": test_tau.compute_episode_return(),
            #     "agent/episode_length": len(test_tau),
            # }, step=iteration)
        # convert the data to trajectory and batch
        trajectories = []
        for env_batch in env_batches:
            for raw_tau in env_batch:
                if not raw_tau:
                    continue
                # raw_tau.pop()
                tau = Trajectory(raw_tau)
                tau.compute_advantages(self.gamma, self.lambda_)
                trajectories.append(tau)
        return Batch(trajectories), (obs, env_ids)

    def sample_episodes(self, num_episodes: int, video_name: str = None) -> List[Trajectory]:
        env = envpool.make(self.env_name, "gymnasium", num_envs=2, batch_size=2, seed=int(time.time() // 1000), )  # testing envs
        env.async_reset()
        env_batches = [[[]] for _ in range(self.num_envs)]
        completed_episodes = 0

        # get the very first observation
        obs, reward, terminated, truncated, info = env.recv()
        env_ids = info["env_id"]
        pbar = tqdm(desc="Steps processed", leave=True)
        while completed_episodes < num_episodes:
            action, value, log_pi_a = self.model.step(obs)
            # send the action to the env
            env.send(action, env_ids)
            # get the next observation
            next_obs, reward, terminated, truncated, info = env.recv()
            env_ids = info["env_id"]
            done = terminated | truncated

            # store the data in the corresponding trajectory
            for i, env_id in enumerate(env_ids):
                env_batches[env_id][-1].append((obs[i], action[i], log_pi_a[i], reward[i], value[i]))
                if done[i]:
                    env_batches[env_id][-1].append((next_obs[i], None, None, 0, 0))  # utilize the last obs
                    env_batches[env_id].append([])
                    completed_episodes += 1
                    pbar.set_description(f"Steps processed. Completed episodes: {completed_episodes}/{num_episodes}")

            obs = next_obs
            pbar.update(1)
        pbar.close()

        # post-process the trajectories
        for env_batch in env_batches:
            for i in range(1, len(env_batch)):
                if not env_batch[i]:
                    continue
                env_batch[i].pop(0)
        # also drop the last trajectory in every envs if not terminated
        for env_batch in env_batches:
            if not env_batch[-1] or env_batch[-1][-1][1] is not None:
                env_batch.pop()
        trajectories = []
        for env_batch in env_batches:
            for raw_tau in env_batch:
                if not raw_tau:
                    continue
                tau = Trajectory(raw_tau)
                tau.compute_advantages(self.gamma, self.lambda_)
                trajectories.append(tau)

        if video_name is not None:
            trajectories.sort(key=lambda t: t.compute_episode_return(), reverse=True)
            best_tau = trajectories[0]
            best_tau.record_video(video_name)

        return trajectories

    def reset_model(self, model: AbstractModel):
        self.model = copy.deepcopy(model).to(self.device)
        self.model.device = self.device
        self.model.eval()
        if self.video_recorder is not None:
            self.video_recorder.reset_model(self.model)
            self.video_recorder.device = self.device


def plot_obs(obs: np.ndarray):
    # grayscale
    plt.imshow(obs, cmap="gray")
    plt.show()


def save_video(obses: np.ndarray, filename: str):
    """
    Save a video from a sequence of observations.
    :param obses: shape (video_length, height, width, channels = 1 or 3)
    :param filename: the filename of the video
    :return: None
    """
    obses = obses.astype(np.uint8)
    # Get the video dimensions
    video_length, height, width, channels = obses.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

    # Iterate over the observations and write each frame to the video
    for obs in obses:
        # Convert the observation to the appropriate color format
        if channels == 1:
            obs = cv2.cvtColor(obs, cv2.COLOR_GRAY2BGR)
        elif channels == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        # Write the frame to the video
        out.write(obs)

    # Release the VideoWriter object
    out.release()


class Sampler:
    env_func: Callable[[], gym.Env]
    num_agents: int
    model: AbstractModel
    batch_size: int
    gamma: float
    lambda_: float
    current_iter: int  # how many sampling has been done

    agents: List[Agent]
    record_video_agent: Optional[Agent]

    def __init__(self,
                 env_func: Callable[[], gym.Env],
                 num_agents: int,
                 model: AbstractModel,
                 batch_size: int,
                 gamma: float,
                 lambda_: float,
                 record_video: bool = True):
        self.env_func = env_func
        self.num_agents = num_agents
        self.model = model
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        if record_video:
            self.record_video_agent = Agent(lambda: RecordVideo(
                env_func(), "videos", episode_trigger=lambda e: e % 10 == 0, disable_logger=True), model)
        else:
            self.record_video_agent = None

        self.agents = [Agent(env_func, model) for _ in range(num_agents)]
        self.current_iter = 0

    def sample(self) -> Batch:
        self.current_iter += 1
        taus = []
        with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
            futures = []
            if self.record_video_agent is not None:
                futures.append(executor.submit(self.record_video_agent.run))
            while self.compute_batch_size(taus) < self.batch_size:
                for agent in self.agents:
                    futures.append(executor.submit(agent.run))
                for future in as_completed(futures):
                    taus.append(future.result())
                futures.clear()  # TODO: design a more clever way to collect trajectories in parallel
        for tau in taus:
            tau.compute_advantages(self.gamma, self.lambda_)
        return Batch(taus)

    @staticmethod
    def compute_batch_size(taus: List[Trajectory]) -> int:
        return sum(len(tau) for tau in taus)

    def reset_model(self, model: AbstractModel) -> None:
        self.model = model
        for agent in self.agents:
            agent.reset_model(model)
