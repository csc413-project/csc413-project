import numpy as np
import torch
import torch.nn as nn

from models.behavior_models import DenseModel, ActionDecoder
from models.observation_models import ObservationEncoder, ObservationDecoder
from models.rssm import get_feat, RSSMState, RSSM, apply_states


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
            feature_size, value_shape, value_layers, value_hidden
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
        action = self.exploration(action)

        value = self.value_model(feature)
        reward = self.reward_model(feature)
        return action, action_dist, value, reward, state

    def policy(self, feat):
        action_dist = self.action_decoder(feat)
        if self.explore:
            action = action_dist.sample()
        else:
            action = action_dist.mode()
        return action, action_dist
    
    def value(self, feat):
        # return  value
        return self.value_model(feat)
    
    def ppo_forward(self, step, obs, prev_action, prev_state, batch_size=1):
        device = 'cuda'
        # Convert numpy array to torch tensor and ensure it's on the correct device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)

        # Ensure obs is 2D (batch_size, feature_dim)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension if it's not there

        # Encode observations into embeddings
        obs_embed = self.observation_encoder(obs)
        
        # print(obs_embed.shape)
        # make it [1, 1024]
        obs_embed = obs_embed.view(1, -1)

        # If prev_action or prev_state are None, initialize them
        if prev_action is None:
            # Assuming actions are discrete and one-dimensional
            prev_action = torch.zeros((1, self.rssm.action_size), device=device)  # Ensure 2D tensor
            
        prev_action = prev_action.unsqueeze(0) if prev_action is None or prev_action.ndim == 1 else prev_action
        
        print(prev_action.shape)

        if prev_state is None:
            prev_state = self.rssm.create_initial_state(batch_size, device=device)

        # Single step forward in RSSM - since we are processing one step at a time
        prior, posterior = self.rssm.observe(
            1, obs_embed.unsqueeze(0), prev_action.unsqueeze(0), prev_state
        )  # Add time dimension

        # Get the last state from the outputs
        last_posterior = apply_states(posterior, lambda x: x.squeeze(0))  # Remove time dimension

        # Extract features for decision making
        feat = get_feat(last_posterior)

        # Compute policy (action probabilities) and value outputs
        action_dist = self.action_decoder(feat)  # Policy output
        value_estimation = self.value_model(feat)  # Value output
    
        if self.explore:
            action = action_dist.sample()
        else:
            action = action_dist.mode()
        
        # print(action_logits)
        print('-' * 50)
        
        print(action)
        
        print('-' * 50)

        return action, value_estimation, last_posterior

        
    def imitator_forward(self, obs):
        pass        

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