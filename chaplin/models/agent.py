import numpy as np
import torch
import torch.nn as nn

from .behavior_models import DenseModel, ActionDecoder
from .observation_models import ObservationEncoder, ObservationDecoder
from .rssm import get_feat, RSSMState, RSSM, apply_states


class ChaplinAgent(nn.Module):
    def __init__(
        self,
        action_shape,
        obs_image_shape=(3, 64, 64),
        stochastic_size=30,
        deterministic_size=200,
        hidden_size=200,
        action_hidden_size=400,
        action_layers=4,
        action_dist="tanh_normal",
        reward_shape=(1,),
        reward_layers=2,
        reward_hidden=400,
        value_shape=(1,),
        value_layers=3,
        value_hidden=400,
        imitator_layers=3,
        imitator_hidden=400,
        explore=True,
        # exploration parameters
        expl_type="additive_gaussian",
        train_noise=0.3,
        eval_noise=0.0,
        expl_decay=0.0,
        expl_min=0.0,
    ):
        super().__init__()
        feature_size = stochastic_size + deterministic_size

        self.train_noise = train_noise
        # Shared encoder
        self.observation_encoder = ObservationEncoder(obs_shape=obs_image_shape)
        encoder_embed_size = np.prod(self.observation_encoder.embed_shape).item()
        self.rssm = RSSM(
            encoder_embed_size,
            np.prod(action_shape).item(),
            stochastic_size,
            deterministic_size,
            hidden_size,
        )

        # Policy model
        self.action_decoder = ActionDecoder(
            np.prod(action_shape).item(),
            feature_size,
            action_hidden_size,
            action_layers,
            action_dist,
        )

        # exploration
        self.explore = explore
        self.expl_type = expl_type
        self.train_noise = train_noise
        self.eval_noise = eval_noise
        self.expl_decay = expl_decay
        self.expl_min = expl_min

        # self.action_model = DenseModel(
        #     feature_size, action_shape, action_layers, action_hidden_size
        # )

        # Value model
        self.value_model = DenseModel(
            feature_size, value_shape, value_layers, value_hidden, dist=None
        )

        # Representation model
        self.observation_decoder = ObservationDecoder(
            feature_size=feature_size, obs_shape=obs_image_shape
        )
        self.reward_model = DenseModel(
            feature_size, reward_shape, reward_layers, reward_hidden
        )

        # Imitator models
        self.imitator_encoder = nn.GRU(
            input_size=feature_size,
            hidden_size=imitator_hidden,
            num_layers=imitator_layers,
            batch_first=True,
        )
        self.imitator_policy = ActionDecoder(
            np.prod(action_shape).item(),
            imitator_hidden,
            action_hidden_size,
            action_layers,
            action_dist,
        )
        self.imitator_value = DenseModel(
            imitator_hidden, value_shape, value_layers, value_hidden
        )

    def forward(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor = None,
        prev_state: RSSMState = None,
    ):
        obs_embedded = self.observation_encoder(observation)
        state = self.get_state_representation(obs_embedded, prev_action, prev_state)

        feature = get_feat(state)

        action, action_dist = self.policy(feature)
        # action = self.exploration(action)

        value = self.value_model(feature)
        reward = self.reward_model(feature)
        return action, action_dist, value, reward, state

    def policy(self, feat):
        action_dist = self.action_decoder(feat)
        if self.explore:
            action = action_dist.rsample()
        else:
            action = action_dist.mode()
        action = torch.clamp(action, min=-0.9999, max=0.9999)
        return action, action_dist

    def get_state_representation(self, obs_embed, prev_action, prev_state):
        if prev_action is None:
            prev_action = torch.zeros(
                obs_embed.shape[0],
                self.rssm.action_size,
                device=obs_embed.device,
                dtype=obs_embed.dtype,
            )
        if prev_state is None:
            prev_state = self.rssm.create_initial_state(
                prev_action.size(0), device=prev_action.device, dtype=prev_action.dtype
            )

        prior = self.rssm.get_prior(prev_action, prev_state)
        posterior = self.rssm.get_posterior(obs_embed, prior)
        return posterior

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
