from typing import Tuple

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf


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
        
        # print to debug
        # print(f'{dist_inputs.shape = }')
        # print(f'{features.shape = }')
        # print(f'{self.output_shape = }')

        # Correct way to form the new shape tuple
        # new_shape = features.shape[:-1] + (self.output_shape,)
        # reshaped_inputs = torch.reshape(dist_inputs, new_shape)

        reshaped_inputs = torch.reshape(
            dist_inputs, features.shape[:-1] + self.output_shape
        )

        if self.dist == "normal":
            return td.Independent(
                td.Normal(reshaped_inputs, 1), len(self.output_shape)
            )
        if self.dist == "binary":
            return td.Independent(
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
    
    def std(self):
        dist = self.dist.expand((self.samples, *self.dist.batch_shape))
        sample = dist.rsample()
        return torch.std(sample, 0)

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
        if self.dist == "tanh_normal":
            mean, std = torch.chunk(x, 2, -1)
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = tf.softplus(std + self.raw_init_std) + self.min_std
            dist = td.Normal(mean, std)
            dist = td.TransformedDistribution(dist, td.TanhTransform())
            dist = td.Independent(dist, 1)
            dist = SampleDist(dist)
            # it looks like SampleDist methods are never used
        elif self.dist == "one_hot":
            dist = td.OneHotCategorical(logits=x)
        elif self.dist == "relaxed_one_hot":
            dist = td.RelaxedOneHotCategorical(0.1, logits=x)
        else:
            raise NotImplementedError(f"{self.dist} not implemented")
        return dist
