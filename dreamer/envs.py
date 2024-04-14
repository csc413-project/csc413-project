import os
from multiprocessing import Process, Pipe

import numpy as np
from dm_control import suite
from gymnasium.spaces import Box


class DMCEnv:
    """
    Only provides pixel observations.
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        image_size=(64, 64),
        normalize_obs: bool = True,
        action_repeat: int = 2,
        seed: int = 0,
        render_kwargs=None,
    ):
        self.env = suite.load(
            domain_name=domain_name, task_name=task_name, task_kwargs={"random": seed}
        )
        self.image_size = image_size
        self.normalize_obs = normalize_obs
        self.action_repeat = action_repeat
        self.render_kwargs = render_kwargs if render_kwargs is not None else {}

    @property
    def observation_space(self):
        # spaces = {}
        # for key, value in self.env.observation_spec().items():
        #     spaces[key] = Box(
        #         value.minimum, value.maximum, value.shape, dtype=value.dtype
        #     )
        # spaces["image"] = Box(0, 255, (3,) + self.image_size, dtype=np.uint8)
        # return Dict(spaces)
        if self.normalize_obs:
            return Box(-0.5, 0.5, (3,) + self.image_size, dtype=np.float32)
        else:
            return Box(0, 255, (3,) + self.image_size, dtype=np.uint8)

    @property
    def action_space(self):
        spec = self.env.action_spec()
        return Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        done = False
        info = {}
        total_reward = 0
        for _ in range(self.action_repeat):
            _, reward, done, info = self._step(action)
            total_reward += reward
            if done:
                break
        obs = np.transpose(self.render(), (2, 0, 1))
        if self.normalize_obs:
            obs = obs.astype(np.float32) / 255.0 - 0.5
        return obs, total_reward, done, info

    def _step(self, action):
        time_step = self.env.step(action)
        # obs = dict(time_step.observation)
        # obs["image"] = self.render()
        obs = None
        reward = time_step.reward
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        # time_step = self.env.reset()
        # obs = dict(time_step.observation)
        # obs["image"] = self.render()
        # return obs
        self.env.reset()
        obs = np.transpose(self.render(), (2, 0, 1))
        if self.normalize_obs:
            obs = obs.astype(np.float32) / 255.0 - 0.5
        return obs

    def render(self):
        return self.env.physics.render(*self.image_size, **self.render_kwargs)


def worker(conn, env_fn, seed=0):
    """
    Worker process that handles stepping through an environment.
    """
    if os.name != "nt":  # TODO: investigate more
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ["MUJOCO_GL"] = "osmesa"
    env = env_fn(seed)  # Initialize the environment for this worker.
    while True:
        cmd, action = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(action)
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "close":
            conn.close()
            break
        else:
            raise ValueError("Unknown command received by worker.")


class VectorDMCEnv:
    """
    Creates a bunch of DMCEnvs and runs them in parallel

    You will have to initialize the environment as follows:
    env_fn = lambda: DMCEnv(domain_name='cheetah', task_name='run')
    with VectorDMCEnv(env_fn, num_envs=4) as vector_env:
        # Do stuff with vector_env
    """

    def __init__(self, env_fn, num_envs, seed=0):
        self.num_envs = num_envs
        self.env_fns = [env_fn for _ in range(num_envs)]
        self.parents, self.workers = zip(*[Pipe() for _ in range(num_envs)])
        self.procs = [
            Process(target=worker, args=(child, env_fn, i))
            for i, child, env_fn in zip(
                range(seed, seed + self.num_envs), self.workers, self.env_fns
            )
        ]

        for proc in self.procs:
            proc.start()

        # Used to get the observation space and action space
        self.env = env_fn()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, actions):
        """
        Steps through all environments in parallel.
        """
        for parent, action in zip(self.parents, actions):
            parent.send(("step", action))
        results = [parent.recv() for parent in self.parents]
        obs, rewards, dones, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), infos

    def reset(self):
        """
        Resets all environments.
        """
        for parent in self.parents:
            parent.send(("reset", None))
        observations = [parent.recv() for parent in self.parents]
        return np.array(observations)

    def close(self):
        # Send a close signal to each environment process
        for parent in self.parents:
            parent.send(("close", None))

        # Close all parent ends of the pipes
        for parent in self.parents:
            parent.close()

        # Wait for all processes to finish
        for proc in self.procs:
            proc.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
