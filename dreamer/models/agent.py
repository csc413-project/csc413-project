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
        action_hidden_size=400,
        action_layers=4,
        action_dist="tanh_normal",
        explore: bool = True,
        # exploration parameters
        expl_type="additive_gaussian",
        train_noise=0.3,
        eval_noise=0.0,
        expl_decay=0.0,
        expl_min=0.0,
        # reward model parameters
        reward_shape=(1,),
        reward_layers=2,
        reward_hidden=400,
        # value model parameters
        value_shape=(1,),
        value_layers=3,
        value_hidden=400,
        # pcont model parameters
        use_pcont=False,
        pcont_layers=3,
        pcont_hidden=400,
        pcont_scale=10.0,
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
        self.explore = explore
        self.action_decoder = ActionDecoder(
            self.action_size,
            feature_size,
            action_hidden_size,
            action_layers,
            action_dist,
        )
        # exploration
        self.expl_type = expl_type
        self.train_noise = train_noise
        self.eval_noise = eval_noise
        self.expl_decay = expl_decay
        self.expl_min = expl_min
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
        action = self.exploration(action)
        feature = get_feat(state)
        value = self.value_model(feature)
        reward = self.reward_model(feature)
        return action, action_dist, value, reward, state

    def policy(self, state: RSSMState):
        feat = get_feat(state)
        action_dist = self.action_decoder(feat)
        if self.action_dist == "tanh_normal":
            if self.explore:
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

    def exploration(self, action: torch.Tensor) -> torch.Tensor:
        """
        :param action: action to take, shape (1,) (if categorical), or (action dim,) (if continuous)
        :return: action of the same shape passed in, augmented with some noise
        """
        if self.explore:
            expl_amount = self.train_noise
            # TODO: implement decay
            if self.expl_decay:
                raise NotImplementedError
                # expl_amount = expl_amount - self._itr / self.expl_decay
            # if self.expl_min:
            #     expl_amount = max(self.expl_min, expl_amount)
        else:
            expl_amount = self.eval_noise

        if self.expl_type == "additive_gaussian":  # For continuous actions
            noise = torch.randn(*action.shape, device=action.device) * expl_amount
            return torch.clamp(action + noise, -1, 1)
        raise NotImplementedError(self.expl_type)
        # TODO: implement other exploration types
        # if self.expl_type == "completely_random":  # For continuous actions
        #     if expl_amount == 0:
        #         return action
        #     else:
        #         return (
        #             torch.rand(*action.shape, device=action.device) * 2 - 1
        #         )  # scale to [-1, 1]
        # if self.expl_type == "epsilon_greedy":  # For discrete actions
        #     action_dim = self.env_model_kwargs["action_shape"][0]
        #     if np.random.uniform(0, 1) < expl_amount:
        #         index = torch.randint(
        #             0, action_dim, action.shape[:-1], device=action.device
        #         )
        #         action = torch.zeros_like(action)
        #         action[..., index] = 1
        #     return action
        # raise NotImplementedError(self.expl_type)

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
