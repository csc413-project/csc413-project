from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .behavior_models import DenseModel, ActionDecoder
from .observation_models import ObservationEncoder, ObservationDecoder
from .rssm import (
    get_feat,
    RSSMState,
    RSSM,
)


class AgentModel(nn.Module):
    def __init__(
        self,
        action_shape: Tuple[int, ...],
        obs_image_shape=(3, 64, 64),
        # RSSM parameters
        stochastic_size=30,
        deterministic_size=200,
        hidden_size=200,
        # action decoder parameters
        action_hidden_size=200,
        action_layers=3,
        action_dist="tanh_normal",
        # reward model parameters
        reward_shape=(1,),
        reward_layers=3,
        reward_hidden=300,
        # value model parameters
        value_shape=(1,),
        value_layers=3,
        value_hidden=200,
        # pcont model parameters
        use_pcont=False,
        pcont_layers=3,
        pcont_hidden=200,
    ):
        super().__init__()
        feature_size = stochastic_size + deterministic_size
        # world model
        self.observation_encoder = ObservationEncoder(obs_shape=obs_image_shape)
        encoder_embed_size = np.prod(self.observation_encoder.embed_shape).item()
        self.observation_decoder = ObservationDecoder(
            feature_size=feature_size, obs_shape=obs_image_shape
        )
        self.action_size = np.prod(action_shape).item()
        self.rssm = RSSM(
            encoder_embed_size,
            self.action_size,
            stochastic_size,
            deterministic_size,
            hidden_size,
        )
        self.reward_model = DenseModel(
            feature_size, reward_shape, reward_layers, reward_hidden
        )
        # action decoder
        self.action_dist = action_dist
        self.action_decoder = ActionDecoder(
            self.action_size,
            feature_size,
            action_hidden_size,
            action_layers,
            action_dist,
        )
        # value model
        self.value_model = DenseModel(
            feature_size, value_shape, value_layers, value_hidden
        )
        if use_pcont:
            self.pcont = DenseModel(
                feature_size, (1,), pcont_layers, pcont_hidden, dist="binary"
            )

    def forward(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor = None,
        prev_state: RSSMState = None,
    ):
        state = self.get_state_representation(observation, prev_action, prev_state)
        action, action_dist = self.policy(state)
        value = self.value_model(get_feat(state))
        reward = self.reward_model(get_feat(state))
        return action, action_dist, value, reward, state

    def policy(self, state: RSSMState):
        feat = get_feat(state)
        action_dist = self.action_decoder(feat)
        if self.action_dist == "tanh_normal":
            if self.training:  # use agent.train(bool) or agent.eval()
                action = action_dist.rsample()
            else:
                action = action_dist.mode()
        elif self.action_dist == "one_hot":
            action = action_dist.sample()
            # This doesn't change the value, but gives us straight-through gradients
            action = action + action_dist.probs - action_dist.probs.detach()
        elif self.action_dist == "relaxed_one_hot":
            action = action_dist.rsample()
        else:
            action = action_dist.sample()
        return action, action_dist

    def get_state_representation(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor = None,
        prev_state: RSSMState = None,
    ):
        """
        :param observation: size(batch, channels, width, height)
        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        obs_embed = self.observation_encoder(observation)
        if prev_action is None:
            prev_action = torch.zeros(
                observation.size(0),
                self.action_size,
                device=observation.device,
                dtype=observation.dtype,
            )
        if prev_state is None:
            prev_state = self.rssm.create_initial_state(
                prev_action.size(0), device=prev_action.device, dtype=prev_action.dtype
            )
        prior = self.rssm.get_prior(prev_action, prev_state)
        posterior = self.rssm.get_posterior(obs_embed, prior)
        return posterior
