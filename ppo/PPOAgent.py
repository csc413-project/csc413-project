from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Assuming these are imported correctly from your local modules
from .observation_models import ObservationEncoder
from .rssm import RSSMState, RSSM, get_feat
from .behavior_models import DenseModel, ActionDecoder

def create_mlp(sizes: List[int], activation: Type[nn.Module],
               output_activation: Type[nn.Module] = nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class PPOAgent(nn.Module):
    def __init__(self,
                 action_shape: Tuple[int, ...],
                 obs_image_shape=(3, 64, 64),
                 # Encoder parameters
                 encoder_hidden_sizes=[32, 64, 64],  # Modify as needed to match Dreamer's encoder settings
                 encoder_activation=nn.ReLU,
                 # PPO Policy and Value network parameters
                 hidden_sizes=[64, 64],
                 activation=nn.Tanh):
        super().__init__()

        self.observation_encoder = ObservationEncoder(obs_shape=obs_image_shape)
        encoder_output_size = np.prod(self.observation_encoder.embed_shape).item()

        # Create the MLP for the policy
        self.policy_net = create_mlp([encoder_output_size] + hidden_sizes + [np.prod(action_shape).item()], activation)
        # Create the MLP for the value function
        self.value_net = create_mlp([encoder_output_size] + hidden_sizes + [1], activation)

    def forward(self, observation: torch.Tensor):
        obs_embed = self.observation_encoder(observation)
        logits = self.policy_net(obs_embed)
        values = self.value_net(obs_embed)
        return Categorical(logits=logits), values.squeeze(-1)

    def act(self, observation: torch.Tensor):
        """ Generate action and value prediction from a single observation. """
        dist, value = self.forward(observation)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        """ Evaluate actions using current policy to obtain log probabilities and values for given observations and actions. """
        dist, value = self.forward(observations)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy


class PPOAgentRNN(nn.Module):
    def __init__(self,
                 action_shape: Tuple[int, ...],
                 obs_image_shape=(3, 64, 64),
                 stochastic_size=30,
                 deterministic_size=200,
                 hidden_size=200,
                 encoder_hidden_sizes=[32, 64, 64],
                 encoder_activation=nn.ReLU,
                 hidden_sizes=[64, 64],
                 activation=nn.Tanh):
        super().__init__()

        self.observation_encoder = ObservationEncoder(obs_shape=obs_image_shape)
        encoder_output_size = np.prod(self.observation_encoder.embed_shape).item()
        
        # Initialize RSSM components
        self.action_size = np.prod(action_shape).item()
        self.rssm = RSSM(
            encoder_output_size,
            self.action_size,
            stochastic_size,
            deterministic_size,
            hidden_size
        )
        
        # Create the MLP for the policy and value based on the RSSM's output feature size
        feature_size = stochastic_size + deterministic_size
        self.policy_net = create_mlp([feature_size] + hidden_sizes + [np.prod(action_shape).item()], activation)
        self.value_net = create_mlp([feature_size] + hidden_sizes + [1], activation)

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor, prev_state: RSSMState):
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
        
        # Get RSSM state transitions
        state = self.rssm.get_posterior(obs_embed, self.rssm.get_prior(prev_action, prev_state))
        feature = get_feat(state)

        # Compute policy and value outputs
        logits = self.policy_net(feature)
        values = self.value_net(feature)
        return Categorical(logits=logits), values.squeeze(-1), state

    def act(self, observation: torch.Tensor, prev_action: torch.Tensor, prev_state: RSSMState):
        """ Generate action, value prediction, and next state from a single observation. """
        dist, value, state = self.forward(observation, prev_action, prev_state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, state

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor, prev_actions: torch.Tensor, states: RSSMState):
        """ Evaluate actions using current policy to obtain log probabilities, values, and states for given observations and actions. """
        dist, value, new_states = self.forward(observations, prev_actions, states)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy, new_states
