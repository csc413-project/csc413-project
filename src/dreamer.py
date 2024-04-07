from agent import AgentModel
from models import (get_feat, get_dist)
import torch
import torch.distributions as td

from src.utils import FreezeParameters


class Dreamer:

    def __init__(self, agent: AgentModel):
        self.agent = agent
        self.free_nats = ...
        self.kl_beta = ...

    def update(self):
        pass

    def calculate_loss(self, observations, actions, rewards):
        """
        Note: With observation[0], the agent took actions[0] and get rewards[0].
        Assume they are float32 and on the proper device.
        :param observations: (seq_len, batch_size, img_shape)
        :param actions: (seq_len, batch_size, act_shape)
        :param rewards: (seq_len, batch_size)
        :return: 
        """
        seq_len, batch_size = observations.shape[:2]
        img_shape = observations.shape[2:]
        act_shape = actions.shape[2:]
        
        # normalize obs images to make it center around 0
        observations = observations / 255.0 - 0.5
        obs_embed = self.agent.observation_encoder(observations)
        
        # init prev state
        prev_state = self.agent.representation.initial_state(batch_size)
        # get prior and posterior and initialize stuff
        prior, posterior = self.agent.rollout.rollout_representation(
            seq_len, obs_embed, actions, prev_state
        )  # (seq_len, batch_size, state_dim)
        prior_dist, posterior_dist = get_dist(prior), get_dist(posterior)
        features = get_feat(posterior)

        image_pred = self.agent.observation_decoder(features)
        image_loss = -torch.mean(image_pred.log_prob(observations))
        reward_pred = self.agent.reward_model(features)
        reward_loss = -torch.mean(reward_pred.log_prob(rewards))
        # TODO: also add pcont
        kl_div = torch.max(
            torch.mean(td.kl_divergence(posterior_dist, prior_dist)),
            self.free_nats  # to prevent penalize small KL divergence
        )
        model_loss = image_loss + reward_loss + self.kl_beta * kl_div
        
        pass
        
        
        
        
