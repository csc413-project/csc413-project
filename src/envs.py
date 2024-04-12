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
    ):
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.image_size = image_size
        self.normalize_obs = normalize_obs
        self.action_repeat = action_repeat

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
        return self.env.physics.render(*self.image_size, camera_id=0)
