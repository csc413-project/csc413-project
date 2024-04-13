import os
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import wandb
from tqdm import trange

from data import Collector, ExperienceReplayDataset, VectorDMCEnv, VectorCollector
from chaplin import Chaplin
from envs import DMCEnv
from models.agent import ChaplinAgent
from utils import denormalize_images, count_env_steps

ENV_SETTINGS = {
    "cartpole": ["swingup"],
    "cheetah": ["run"],
    "finger": ["turn_hard"],
    "hopper": ["hop"],
}

ENV_PREFERRED_CAMERA = {
    ("cartpole", "swingup"): None,
    ("cheetah", "run"): 0,
    ("finger", "turn_hard"): 0,
    ("hopper", "hop"): 0,
}

def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    input:
        vector x: [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """
    return np.array([sum(x[j] * (discount ** j) for j in range(i, len(x)))
                     for i in range(len(x))])


@dataclass
class DreamerConfig:
    # env setting
    domain_name: str = "cheetah"
    task_name: str = "run"
    obs_image_size: Tuple = (64, 64)
    action_repeats: int = 2
    camera_id: int = ENV_PREFERRED_CAMERA[(domain_name, task_name)]
    render_kwargs: Dict = None
    # general setting
    base_dir = f"/home/blair/tmp/chaplin/{domain_name}_{task_name}/2/"
    data_dir: str = os.path.join(base_dir, "episodes")  # where to store trajectories
    model_dir: str = os.path.join(base_dir, "models")  # where to store models
    load_model_path: Optional[str] = None
    # training setting
    training_epochs: int = 1100  # number of training episodes
    prefill_episodes = 5  # number of episodes to prefill the dataset
    batch_size: int = 50  # batch size for training
    batch_length: int = 50  # sequence length of each training batch
    training_steps: int = 1  # number of training steps
    ppo_training_steps: int = 3
    training_device = "cuda"  # training device
    # testing setting
    test_every: int = 8  # test (and save model) every n episodes
    test_num_envs: int = 5  # number of parallel test environments
    test_env_starting_seed: int = 1108  # starting seed for test environments
    # collector setting
    collector_device = "cuda"  # collector device
    gamma = 0.95
    lambda_ = 0.95

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        if self.camera_id is None:
            self.render_kwargs = {}
        else:
            self.render_kwargs = {
                "camera_id": self.camera_id,
            }


def main():
    config = DreamerConfig()

    # wandb.login(key=os.getenv("WANDB_KEY"))
    wandb.init(
        project="csc413-proj",
        config=asdict(config),
        name=f"dreamer-{config.domain_name}_{config.task_name}",
        entity="scott-reseach",
        mode="disabled"
    )
    wandb.define_metric("env_steps")
    wandb.define_metric("agent/*", step_metric="env_steps")
    wandb.define_metric("dataset/*", step_metric="env_steps")
    wandb.define_metric("training_steps")
    wandb.define_metric("train/*", step_metric="training_steps")

    # init env
    env = DMCEnv(
        config.domain_name,
        config.task_name,
        config.obs_image_size,
        action_repeat=config.action_repeats,
        render_kwargs=config.render_kwargs,
    )  # for training
    action_spec = env.action_space

    def create_env(seed: int = 0):  # for testing and initial data collection
        return DMCEnv(
            domain_name=config.domain_name,
            task_name=config.task_name,
            seed=seed,
            render_kwargs=config.render_kwargs,
        )

    # init dreamer
    agent = ChaplinAgent(action_shape=action_spec.shape)
    if config.load_model_path is not None:
        agent.load_state_dict(torch.load(config.load_model_path))
    chaplin = Chaplin(agent, device=config.training_device)

    # init buffer
    buffer = ExperienceReplayDataset(
        config.data_dir,
        buffer_size=100,
        max_episode_length=1000 // config.action_repeats,
        obs_shape=env.observation_space.shape,
        action_size=np.prod(action_spec.shape).item(),
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        device=config.training_device,
    )

    # init collectors
    train_collector = Collector(env, chaplin.agent, True, config.collector_device)
    test_collector = VectorCollector(
        VectorDMCEnv(create_env, config.test_num_envs, config.test_env_starting_seed),
        chaplin.agent,
        False,
        config.collector_device,
    )

    # prefill dataset with 5 random trajectories, if needed
    total_env_steps, episodes = count_env_steps(config.data_dir, config.action_repeats)
    
    # Seems like PPO doesn't need prefill, but Dreamer does
    # Anyways, I keep it for now.
    
    # if config.prefill_episodes - episodes > 0:
        # prefill_collector = VectorCollector(
        #     VectorDMCEnv(
        #         create_env,
        #         config.prefill_episodes - episodes,
        #         config.test_env_starting_seed,
        #     ),
        #     chaplin.agent,
        #     False,
        #     config.collector_device,
        # )
        # prefill_data, _, (_, env_steps) = prefill_collector.collect(
        #     target_num_episodes=1, random_action=True
        # )
        # prefill_collector.close()
        
        # prefill_data = {k: np.array(v) for k, v in prefill_data[0].items()}
        
        # # print shape of reward and value
        # print(f'{prefill_data["reward"].shape = }')
        # print(f'{prefill_data["action"].shape = }')
        # print(f'{prefill_data["log_prob"].shape = }')
        # print(f'{prefill_data["value"].shape = }')

        
        # total_env_steps += env_steps * config.action_repeats * len(prefill_data)
        # prefill_data = [
        #     {
        #         "obs": prefill_data["obs"][:, i],
        #         "action": prefill_data["action"][:, i],
        #         "log_prob": prefill_data["log_prob"][:,i],
        #         "value": prefill_data["value"][:, i],
        #         "reward": prefill_data["reward"][:, i],
        #     }
        #     for i in range(config.prefill_episodes - episodes)
        # ]
        # for i, data in enumerate(prefill_data):
        #     np.savez_compressed(
        #         os.path.join(config.data_dir, f"pre_{i + episodes}.npz"), **data
        #     )
        #     buffer.add_trajectory(data)

    for i in trange(config.training_epochs, desc="Training Epochs"):
        chaplin.agent.train()
        
        # 1. interact with the environment to collect data, like normal PPO
        train_collector.reset_agent(chaplin.agent)
        data, _, (_, env_steps) = train_collector.collect(target_num_episodes=1)
        total_env_steps += env_steps * config.action_repeats
        data = data[0]  # only 1 trajectory
        buffer.add_trajectory(data)
        np.savez_compressed(os.path.join(config.data_dir, f"{i}.npz"), **data)
        
        no_samples_per_episode = 1
        
        # train_collector.collect(target_num_episodes=no_samples_per_episode)
        
        def 我是猪鼻():
                    # for _ in range(no_samples_per_episode):
            #     obs = env.reset()
            #     done = False
            #     prev_action, prev_hidden = None, None
            #     step = 0
            #     while not done:
            #         action, value, hidden = chaplin.agent.ppo_forward(1,
            #                                                           obs,
            #                                                           prev_action, 
            #                                                           prev_hidden)
                    
            #         next_obs, reward, done, _ = env.step(action.detach().cpu().numpy())
            #         buffer.add(obs, action, value, reward)
            #         obs = next_obs
            #         prev_action = action
            #         prev_hidden = hidden
            #         step += 1
            pass

        # 2. Do representation training
        for _ in trange(config.training_steps, desc="Training Steps"):
            obs, action, log_prob, value, reward = next(buffer)
            # dreamer.imitation_update()
            chaplin.update_dreamer(obs, action, reward)
            
            
        # 3. Do policy training
        online_train = True
        if online_train:
            obs, action, log_prob, value, reward = next(buffer)
            
            # Print shapes for debugging
            print("Shapes of buffered data:")
            print(f"Obs: {obs.shape}, Action: {action.shape}, Log_prob: {log_prob.shape}, Value: {value.shape}, Reward: {reward.shape}")
            print('-' * 50)

            # Convert tensors to numpy if they are not already
            value_np = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            reward_np = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward

            # Append a zero bootstrap value at the end of each sequence in the batch
            # This ensures that values_plus has an extra time step at the end for bootstrap
            values_plus = np.concatenate([value_np, np.zeros((value_np.shape[0], 1))], axis=1)

            # Calculate TD Residuals using broadcasting
            deltas = reward_np + config.gamma * values_plus[:, 1:] - values_plus[:, :-1]
            advantages = discount_cumsum(deltas, config.gamma * config.lambda_)

            # Convert observations, actions, and log probabilities to numpy if needed
            obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
            log_prob_np = log_prob.cpu().numpy() if isinstance(log_prob, torch.Tensor) else log_prob

            # Update PPO policy

            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
            action_tensor = torch.tensor(action_np, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
            reward_tensor = torch.tensor(reward_np, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
            value_tensor = torch.tensor(value_np, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
            log_prob_tensor = torch.tensor(log_prob_np, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

            # Now call the ppo_update with tensors
            chaplin.update_ppo(obs_tensor, action_tensor, reward_tensor, value_tensor, log_prob_tensor, advantages_tensor)

                        

        wandb.log(
            {
                "env_steps": total_env_steps,
                "agent/training_return": sum(data["reward"]),
            }
        )

        # test
        if i % config.test_every == 0 or i == config.training_epochs - 1:
            print("Testing...")
            torch.save(
                chaplin.agent.state_dict(),
                os.path.join(config.model_dir, f"{total_env_steps}.pt"),
            )
            test_collector.reset_agent(chaplin.agent)
            data, _, _ = test_collector.collect(target_num_episodes=1)
            data = data[0]
            observations = denormalize_images(np.array(data["obs"]))
            wandb.log(
                {
                    "agent/test_return": np.mean(np.sum(data["reward"], axis=0)),
                    "agent/test_video": [
                        wandb.Video(observations[:, j], fps=30, format="mp4")
                        for j in range(config.test_num_envs)
                    ],
                }
            )

    test_collector.close()



if __name__ == "__main__":
    main()
