from collections import namedtuple
from typing import Tuple, List

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf


class ObservationEncoder(nn.Module):
    def __init__(self, obs_shape=(3, 64, 64), depth=32, stride=2, activation=nn.ReLU):
        super().__init__()
        self.shape = obs_shape
        self.stride = stride
        self.depth = depth

        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=obs_shape[0], out_channels=1 * depth, kernel_size=4, stride=stride),
            activation(),
            nn.Conv2d(in_channels=1 * depth, out_channels=2 * depth, kernel_size=4, stride=stride),
            activation(),
            nn.Conv2d(in_channels=2 * depth, out_channels=4 * depth, kernel_size=4, stride=stride),
            activation(),
            nn.Conv2d(in_channels=4 * depth, out_channels=8 * depth, kernel_size=4, stride=stride),
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

    def __init__(self, embed_shape: Tuple, obs_shape=(3, 64, 64), depth=32, activation=nn.ReLU):
        super().__init__()
        self.embed_shape = embed_shape
        self.obs_shape = obs_shape
        self.depth = depth

        embed_size = np.prod(embed_shape).item()
        self.dense = nn.Linear(in_features=embed_size, out_features=32 * depth)

        self.deconvolutions = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * depth, out_channels=4 * depth, kernel_size=5),
            activation(),
            nn.ConvTranspose2d(in_channels=4 * depth, out_channels=2 * depth, kernel_size=5),
            activation(),
            nn.ConvTranspose2d(in_channels=2 * depth, out_channels=1 * depth, kernel_size=6),
            activation(),
            nn.ConvTranspose2d(in_channels=1 * depth, out_channels=obs_shape[0], kernel_size=6, stride=2),
        )

    def forward(self, obs_embed):
        batch_shape = obs_embed.shape[:-1]
        x = self.dense(obs_embed)
        x = x.view(-1, 32 * self.depth, 1, 1)
        x = self.deconvolutions(x)
        mean = x.view(*batch_shape, *self.shape)
        # each pixel is a Normal distribution with a standard deviation of 1 for simplicity
        return td.Independent(td.Normal(mean, 1), len(self.shape))


RSSMState = namedtuple("RSSMState", ["mean", "std", "stoch", "deter"])


def stack_states(rssm_states: List, dim: int):
    return RSSMState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )


class RSSMTransition(nn.Module):
    """
    p(s_t | s_{t-1}, a_{t-1})
    """

    def __init__(
            self,
            action_size,
            stochastic_size=30,
            deterministic_size=200,
            hidden_size=200,
            activation=nn.ELU,
            distribution=td.Normal,
    ):
        super().__init__()
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.deter_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.cell = nn.GRUCell(hidden_size, deterministic_size)
        self.rnn_input_model = nn.Sequential(
            nn.Linear(self.action_size + self.stoch_size, self.hidden_size),
            self.activation(),
        )
        self.stochastic_prior_model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            self.activation(),
            nn.Linear(self.hidden_size, 2 * self.stoch_size)
        )
        self.dist = distribution

    def initial_state(self, batch_size, **kwargs):
        return RSSMState(
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.deter_size, **kwargs),
        )

    def forward(self, prev_action: torch.Tensor, prev_state: RSSMState):
        rnn_input = self.rnn_input_model(
            torch.cat([prev_action, prev_state.stoch], dim=-1))
        deter_state = self.cell(rnn_input, prev_state.deter)
        mean, std = torch.chunk(self.stochastic_prior_model(deter_state), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self.dist(mean, std)
        stoch_state = dist.rsample()
        return RSSMState(mean, std, stoch_state, deter_state)


class RSSMRepresentation(nn.Module):
    """
    p(s_t | s_{t-1}, a_{t-1}, o_t)
    """

    def __init__(
            self,
            transition_model: RSSMTransition,
            obs_embed_size,
            action_size,
            stochastic_size=30,
            deterministic_size=200,
            hidden_size=200,
            activation=nn.ELU,
            distribution=td.Normal,
    ):
        super().__init__()
        self.transition_model = transition_model
        self.obs_embed_size = obs_embed_size
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.deter_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.dist = distribution
        self.stochastic_posterior_model = nn.Sequential(
            nn.Linear(self.deter_size + self.obs_embed_size, self.hidden_size),
            self.activation(),
            nn.Linear(self.hidden_size, 2 * self.stoch_size)
        )

    def initial_state(self, batch_size, **kwargs):
        return RSSMState(
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.deter_size, **kwargs),
        )

    def forward(self,
                obs_embed: torch.Tensor,
                prev_action: torch.Tensor,
                prev_state: RSSMState):
        prior_state = self.transition_model(prev_action, prev_state)
        x = torch.cat([prior_state.deter, obs_embed], -1)
        mean, std = torch.chunk(self.stochastic_posterior_model(x), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self.dist(mean, std)
        stoch_state = dist.rsample()
        posterior_state = RSSMState(mean, std, stoch_state, prior_state.deter)
        return prior_state, posterior_state


class DenseModel(nn.Module):
    """
    For model reward q(r_t | s_t) and value v(s_t)
    """

    def __init__(
        self,
        feature_size: int,
        output_shape: Tuple,
        layers: int,
        hidden_size: int,
        dist: str = "normal",
        activation=nn.ELU,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.layers = layers
        self.hidden_size = hidden_size
        self.dist = dist
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self.feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self.feature_size, self.hidden_size)]
        model += [self.activation()]
        for i in range(self.layers - 1):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self.hidden_size, int(np.prod(self.output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs = torch.reshape(
            dist_inputs, features.shape[:-1] + self.output_shape
        )
        if self.dist == "normal":
            return td.independent.Independent(
                td.Normal(reshaped_inputs, 1), len(self.output_shape)
            )
        if self.dist == "binary":
            return td.independent.Independent(
                td.Bernoulli(logits=reshaped_inputs), len(self.output_shape)
            )
        raise NotImplementedError(self.dist)


class SampleDist:

    def __init__(self, dist: td.Distribution, samples=100):
        self.dist = dist
        self.samples = samples

    def __getattr__(self, name):
        return getattr(self.dist, name)

    def mean(self):
        dist = self.dist.expand((self.samples, *self.dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self.dist.expand((self.samples, *self.dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self.dist.expand((self.samples, *self.dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self.dist.sample()


class ActionDecoder(nn.Module):
    """
    q(a_t | s_t)
    """
    
    def __init__(
        self,
        action_size,
        feature_size,
        hidden_size,
        layers,
        dist: str = "tanh_normal",
        activation=nn.ELU,
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
    ):
        super().__init__()
        self.action_size = action_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dist = dist
        self.activation = activation
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.feedforward_model = self.build_model()
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def build_model(self):
        model = [nn.Linear(self.feature_size, self.hidden_size)]
        model += [self.activation()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        if self.dist == "tanh_normal":
            model += [nn.Linear(self.hidden_size, self.action_size * 2)]
        elif self.dist == "one_hot" or self.dist == "relaxed_one_hot":
            model += [nn.Linear(self.hidden_size, self.action_size)]
        else:
            raise NotImplementedError(f"{self.dist} not implemented")
        return nn.Sequential(*model)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        dist = None
        if self.dist == "tanh_normal":
            mean, std = torch.chunk(x, 2, -1)
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = tf.softplus(std + self.raw_init_std) + self.min_std
            dist = td.Normal(mean, std)
            dist = td.TransformedDistribution(dist, td.TanhTransform())
            dist = td.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self.dist == "one_hot":
            dist = torch.distributions.OneHotCategorical(logits=x)
        elif self.dist == "relaxed_one_hot":
            dist = torch.distributions.RelaxedOneHotCategorical(0.1, logits=x)
        else:
            raise NotImplementedError(f"{self.dist} not implemented")
        return dist
