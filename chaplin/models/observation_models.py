import torch
import torch.distributions as td
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=64, img_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Sequential(
            # Using a convolution to implement patch embedding
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)  # Flattening height and width into "patch" dimension
        )

        # Positional embedding
        self.positions = nn.Parameter(torch.randn(num_patches, emb_size))

    def forward(self, x):
        x = self.projection(x)  # [B, E, N] B=batch size, E=embedding dim, N=num_patches
        x = x.permute(0, 2, 1)  # [B, N, E]
        return x + self.positions

class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, emb_size=64, num_heads=4, num_layers=4, feature_size=512):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=emb_size, img_size=img_size)

        layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.to_cls_token = nn.Identity()
        
        self.embed_shape = feature_size

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, self.embed_shape)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_token = self.to_cls_token(torch.mean(x, dim=1))  # Using mean pooling as representation
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pooling over the sequence dimension
        x = self.mlp_head(x + cls_token)
        return x


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
