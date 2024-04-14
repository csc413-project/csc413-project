from collections import namedtuple
from typing import List, Callable

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf

RSSMState = namedtuple("RSSMState", ["mean", "std", "stoch", "deter"])


def apply_states(rssm_state: RSSMState, fn: Callable[[torch.Tensor], torch.Tensor]):
    """
    Apply a function to all the components of a state.
    """
    return RSSMState(
        fn(rssm_state.mean),
        fn(rssm_state.std),
        fn(rssm_state.stoch),
        fn(rssm_state.deter),
    )


def stack_states(rssm_states: List, dim: int):
    return RSSMState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )


def get_feat(rssm_state: RSSMState):
    return torch.cat((rssm_state.stoch, rssm_state.deter), dim=-1)


def get_dist(rssm_state: RSSMState):
    return td.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)


class RSSM(nn.Module):

    def __init__(
        self,
        obs_embed_size,
        action_size,
        stochastic_size=30,  # size of the stochastic state
        deterministic_size=200,  # size of the deterministic state
        hidden_size=200,  # size of the hidden state for linear layers
        activation=nn.ELU,
        distribution=td.Normal,
    ):
        super().__init__()
        self.obs_embed_size = obs_embed_size
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.deter_size = deterministic_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.dist = distribution

        # transition model
        self.cell = nn.GRUCell(hidden_size, deterministic_size)
        self.rnn_input_model = nn.Sequential(
            nn.Linear(self.action_size + self.stoch_size, self.hidden_size),
            self.activation(),
        )
        self.stochastic_prior_model = nn.Sequential(
            nn.Linear(self.deter_size, self.hidden_size),
            self.activation(),
            nn.Linear(self.hidden_size, 2 * self.stoch_size),
        )
        self.stochastic_posterior_model = nn.Sequential(
            nn.Linear(self.deter_size + self.obs_embed_size, self.hidden_size),
            self.activation(),
            nn.Linear(self.hidden_size, 2 * self.stoch_size),
        )

    def forward(
        self,
        steps: int,
        obs_embed: torch.Tensor,
        action: torch.Tensor,
        prev_state: RSSMState,
    ):
        return self.observe(steps, obs_embed, action, prev_state)

    def observe(
        self,
        steps: int,
        obs_embed: torch.Tensor,
        action: torch.Tensor,
        prev_state: RSSMState,
    ):
        """
        Roll out the model with actions and observations from data.
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, embedding_size)
        :param action: size(time_steps, batch_size, action_size)
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: prior, posterior states. size(time_steps, batch_size, state_size)
        """
        priors = []
        posteriors = []
        for t in range(steps):
            prior_state = self.get_prior(action[t], prev_state)
            posterior_state = self.get_posterior(obs_embed[t], prior_state)
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state
        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior, post

    def follow(self, steps: int, action: torch.Tensor, prev_state: RSSMState):
        """
        Roll out the model with actions from data.
        :param steps: number of steps to roll out
        :param action: size(time_steps, batch_size, action_size)
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: prior states. size(time_steps, batch_size, state_size)
        """
        priors = []
        state = prev_state
        for t in range(steps):
            state = self.get_prior(action[t], state)
            priors.append(state)
        return stack_states(priors, dim=0)

    def imagine(
        self,
        steps: int,
        policy: Callable[[RSSMState], torch.Tensor],
        prev_state: RSSMState,
    ):
        """
        Roll out the model with a policy function.
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
        state = prev_state
        next_states = []
        actions = []
        for t in range(steps):
            action, _ = policy(apply_states(state, lambda x: x.detach()))
            state = self.get_prior(action, state)
            next_states.append(state)
            actions.append(action)
        next_states = stack_states(next_states, dim=0)
        actions = torch.stack(actions, dim=0)
        return next_states, actions

    def get_prior(self, prev_action: torch.Tensor, prev_state: RSSMState):
        rnn_input = self.rnn_input_model(
            torch.cat([prev_action, prev_state.stoch], dim=-1)
        )
        deter_state = self.cell(rnn_input, prev_state.deter)
        mean, std = torch.chunk(self.stochastic_prior_model(deter_state), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self.dist(mean, std)
        stoch_state = dist.rsample()
        prior_state = RSSMState(mean, std, stoch_state, deter_state)
        return prior_state

    def get_posterior(self, obs_embed: torch.Tensor, prior_state: RSSMState):
        x = torch.cat([prior_state.deter, obs_embed], -1)
        mean, std = torch.chunk(self.stochastic_posterior_model(x), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self.dist(mean, std)
        stoch_state = dist.rsample()
        posterior_state = RSSMState(mean, std, stoch_state, prior_state.deter)
        return posterior_state

    def create_initial_state(self, batch_size, **kwargs):
        return RSSMState(
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.deter_size, **kwargs),
        )
