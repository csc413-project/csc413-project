import torch
import torch.distributions as td
import torch.nn as nn


class ObservationEncoder(nn.Module):
    def __init__(self, obs_shape=(3, 64, 64), depth=32, stride=2, activation=nn.ReLU):
        super().__init__()
        self.shape = obs_shape
        self.stride = stride
        self.depth = depth

        self.convolutions = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_shape[0],
                out_channels=1 * depth,
                kernel_size=4,
                stride=stride,
            ),
            activation(),
            nn.Conv2d(
                in_channels=1 * depth,
                out_channels=2 * depth,
                kernel_size=4,
                stride=stride,
            ),
            activation(),
            nn.Conv2d(
                in_channels=2 * depth,
                out_channels=4 * depth,
                kernel_size=4,
                stride=stride,
            ),
            activation(),
            nn.Conv2d(
                in_channels=4 * depth,
                out_channels=8 * depth,
                kernel_size=4,
                stride=stride,
            ),
            activation(),
        )

        with torch.no_grad():
            self.embed_shape = self.convolutions(torch.zeros(1, *obs_shape)).shape

    def forward(self, obs: torch.Tensor):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed


class ObservationDecoder(nn.Module):

    def __init__(
        self, feature_size: int, obs_shape=(3, 64, 64), depth=32, activation=nn.ReLU
    ):
        super().__init__()
        self.embed_size = feature_size
        self.obs_shape = obs_shape
        self.depth = depth

        self.dense = nn.Linear(in_features=feature_size, out_features=32 * depth)

        self.deconvolutions = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32 * depth,
                out_channels=4 * depth,
                kernel_size=5,
                stride=2,
            ),
            activation(),
            nn.ConvTranspose2d(
                in_channels=4 * depth,
                out_channels=2 * depth,
                kernel_size=5,
                stride=2,
            ),
            activation(),
            nn.ConvTranspose2d(
                in_channels=2 * depth,
                out_channels=1 * depth,
                kernel_size=6,
                stride=2,
            ),
            activation(),
            nn.ConvTranspose2d(
                in_channels=1 * depth,
                out_channels=obs_shape[0],
                kernel_size=6,
                stride=2,
            ),
        )

    def forward(self, features):
        batch_shape = features.shape[:-1]
        x = self.dense(features)
        x = x.view(-1, 32 * self.depth, 1, 1)
        x = self.deconvolutions(x)
        mean = x.view(*batch_shape, *self.obs_shape)
        # each pixel is a Normal distribution with a standard deviation of 1 for simplicity
        return td.Independent(td.Normal(mean, 1), len(self.obs_shape))
