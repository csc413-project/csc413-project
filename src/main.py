import os
from dataclasses import dataclass

import dm_control.suite as suite
import numpy as np
import torch
from dm_control.suite.wrappers import pixels
from tqdm import tqdm

from dreamer import Dreamer
from models.agent import AgentModel


@dataclass
class DreamerConfig:
    data_dir = "/home/scott/tmp/dreamer/"  # where to store trajectories


def get_random_action(action_spec):
    minimum, maximum = action_spec.minimum, action_spec.maximum
    shape = action_spec.shape
    return np.random.uniform(minimum, maximum, shape)


def main():
    config = DreamerConfig()
    env = suite.load(domain_name="reacher", task_name="easy")
    env = pixels.Wrapper(env, render_kwargs=dict(height=64, width=64))
    action_spec = env.action_spec()
    # init dreamer
    agent = AgentModel(action_shape=action_spec.shape)
    dreamer = Dreamer(agent)
    # collect
    data = dict(obs=[], action=[], reward=[])
    time_step = env.reset()
    step_type, reward, discount, obs = time_step
    obs = obs["pixels"]
    num_episodes, env_steps = 0, 0
    pbar = tqdm(leave=False)
    prev_action, prev_state = None, None
    while num_episodes < 1 and env_steps < 50:
        with torch.no_grad():
            obs_tensor = torch.unsqueeze(
                torch.transpose(torch.tensor(obs.copy(), dtype=torch.float32), 0, 2), 0
            )
            action, action_dist, value, i_reward, state = dreamer.agent(
                obs_tensor, prev_action, prev_state
            )
        time_step = env.step(action)
        step_type, reward, discount, next_obs = time_step

        if time_step.last():
            step_type, reward, discount, next_obs = env.reset()
            num_episodes += 1
        else:
            data["obs"].append(obs)
            data["action"].append(action.numpy())
            data["reward"].append(reward)

        obs = next_obs["pixels"]
        prev_action = action
        prev_state = state
        env_steps += 1
        pbar.update(1)
    pbar.close()
    np.savez_compressed(os.path.join(config.data_dir, "debug_data.npz"), data["action"])

    # train


if __name__ == "__main__":
    main()
