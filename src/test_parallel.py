import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import trange

from data import TrajectoryDataset, Collector, VectorCollector
from envs import DMCEnv, VectorDMCEnv, SequentialDMCEnv
from models.agent import AgentModel

def create_env():
    return DMCEnv(domain_name="cartpole", task_name="swingup")

if __name__ == "__main__":
    """
    Wrapping the main function in main is IMPORTANT
    If you don't do this, Multiprocess will complain and it will crash
    """

    num_envs = 3

    import timeit

    # # Check the amount of time needed to do this
    # start = timeit.default_timer()

    # with VectorDMCEnv(create_env, num_envs=num_envs) as vector_env:
    #     # Do stuff with vector_env
    #     agent = AgentModel(action_shape=(1, ))

    #     # prefill dataset with 5 random trajectories
    #     total_env_steps = 0
    #     collector = VectorCollector(vector_env, agent, 'cpu')

    #     prefill_data, _, (_, total_env_steps) = collector.collect(
    #         target_num_episodes=1, random_action=True
    #     )

    #     # Demo of how you can access information
    #     last_data = prefill_data[-1]
    #     print(last_data['obs'][0].shape)

    #     # prefill_data is now a list of dictionaries
    #     # The index of the prefill_data is the index of the episode
    #     # Then, the key obs, rewards, actions, etc. are the keys of the dictionary
    #     # Then, there is one more index to get the data from one step in the episode

    #     # Then, the shape is (num_envs, 3, 64, 64)

    #     stop = timeit.default_timer()

    # print('Time (Parallel): ', stop - start)

    start = timeit.default_timer()
    # Run the thing in serial

    # You can optionally provide a seed
    env = SequentialDMCEnv(domain_name="cartpole", task_name="swingup", num_envs=num_envs, seed=0)

    # init agent
    agent = AgentModel(action_shape=(1, ))

    total_env_steps = 0
    collector = VectorCollector(env, agent, 'cpu')

    prefill_data, _, (_, total_env_steps) = collector.collect(
        target_num_episodes=1, random_action=True
    )

    # Demo of how you can access information
    last_data = prefill_data[-1]
    print(last_data['obs'][0].shape)

    # prefill_data is now a list of dictionaries
    # The index of the prefill_data is the index of the episode
    # Then, the key obs, rewards, actions, etc. are the keys of the dictionary
    # Then, there is one more index to get the data from one step in the episode
    # Then the shape is (num_envs, 3, 64, 64)

    stop = timeit.default_timer()
    print('Time (Serial): ', stop - start)
