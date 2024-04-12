from typing import Optional, Union, Tuple, Any, Type, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution


def create_mlp(sizes: List[int], activation: Type[nn.Module],
               output_activation: Type[nn.Module] = nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class AbstractModel(nn.Module):
    device: str = "cuda"

    def forward_policy(self, batch_obs: torch.Tensor,
                       actions: Optional[torch.Tensor] = None) -> Union[Categorical, Tuple[Categorical, torch.Tensor]]:
        raise NotImplementedError

    def forward_value(self, batch_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        raise NotImplementedError

    @staticmethod
    def get_log_prob(distribution: Distribution, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CategoricalMLPModel(AbstractModel):

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden_sizes: List[int],
                 activation: Type[nn.Module] = nn.Tanh):
        super().__init__()
        self.pi_logits_net = create_mlp([obs_dim] + hidden_sizes + [act_dim], activation)
        self.v_net = create_mlp([obs_dim] + hidden_sizes + [1], activation)

    def forward_policy(self, batch_obs: torch.Tensor,
                       actions: Optional[torch.Tensor] = None) -> Union[Categorical, Tuple[Categorical, torch.Tensor]]:
        pi = self.get_distribution(batch_obs)
        if actions is None:
            return pi
        return pi, pi.log_prob(actions)

    def forward_value(self, batch_obs: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.v_net(batch_obs), -1)

    def step(self, obs: np.ndarray) -> Tuple[Any, float, float]:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        """Use for CPU to get actions, values, ..."""
        with torch.no_grad():  # to support multi-process
            pi = self.get_distribution(obs)
            action = pi.sample()
            log_pi_a = self.get_log_prob(pi, action)
            v = self.forward_value(obs)
        return action.item(), v.item(), log_pi_a.item()

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.pi_logits_net(obs)
        return Categorical(logits=logits)

    @staticmethod
    def get_log_prob(distribution: Categorical, action: torch.Tensor) -> torch.Tensor:
        return distribution.log_prob(action)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # adjust the residual connection for changes in dimensions or channels
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = torch.relu(out)

        return out


class CategoricalCNNModel(AbstractModel):

    def __init__(self, in_channels: int, action_dim: int, device="cuda"):
        super().__init__()
        self.device = device
        # input frame stack: (4, 84, 84)
        # self.policy_net = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=4, padding=3),
        #     nn.ReLU(),
        #     ResidualBlock(32, 32),
        #     ResidualBlock(32, 64, stride=2),
        #     ResidualBlock(64, 64, stride=2),
        #     ResidualBlock(64, 64, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(64 * 7 * 5, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, action_dim)
        # )
        # self.value_net = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=4, padding=3),
        #     nn.ReLU(),
        #     ResidualBlock(32, 32),
        #     ResidualBlock(32, 64, stride=2),
        #     ResidualBlock(64, 64, stride=2),
        #     ResidualBlock(64, 64, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(64 * 7 * 5, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            feature_dim = int(np.prod(self.feature_net(torch.zeros(1, in_channels, 84, 84)).shape[1:]))
        print(f"Feature Dimension: {feature_dim}")
        self.policy_net = nn.Sequential(
            self.feature_net,
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )
        self.value_net = nn.Sequential(
            self.feature_net,
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward_policy(self, batch_obs: torch.Tensor,
                       actions: Optional[torch.Tensor] = None) -> Union[Categorical, Tuple[Categorical, torch.Tensor]]:
        # for 5d tensor, needs to be flattened to 4d and then reshape back
        s = None
        if batch_obs.ndim == 5:
            s = batch_obs.shape[:2]
            batch_obs = batch_obs.view(-1, *batch_obs.shape[2:])  # flatten the first two dimensions
        pi = self.get_distribution(batch_obs, new_shape=s)
        if actions is None:
            return pi
        return pi, pi.log_prob(actions)

    def forward_value(self, batch_obs: torch.Tensor) -> torch.Tensor:
        if batch_obs.ndim == 5:
            s = batch_obs.shape[:2]
            batch_obs = batch_obs.view(-1, *batch_obs.shape[2:])
            return self.value_net(batch_obs).view(s)
        else:
            return torch.squeeze(self.value_net(batch_obs), -1)

    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use for CPU to get actions, values, ..."""
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs /= 255
        with torch.no_grad():
            pi = self.get_distribution(obs)
            action = pi.sample()
            log_pi_a = self.get_log_prob(pi, action)
            v = self.forward_value(obs)
        return action.cpu().numpy(), v.cpu().numpy(), log_pi_a.cpu().numpy()

    def get_distribution(self, obs: torch.Tensor,
                         new_shape: Optional[Tuple[int, int]] = None) -> Categorical:
        logits = self.policy_net(obs)
        if new_shape is not None:
            logits = logits.view(*new_shape, -1)
        return Categorical(logits=logits)

    @staticmethod
    def get_log_prob(distribution: Categorical, action: torch.Tensor) -> torch.Tensor:
        return distribution.log_prob(action)
