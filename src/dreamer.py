import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim

from models.agent import AgentModel
from models.rssm import get_feat, get_dist, apply_states
from src.utils import FreezeParameters


class Dreamer:

    def __init__(
        self,
        agent: AgentModel,
        model_lr=1e-4,
        action_lr=1e-4,
        value_lr=1e-4,
        discount=0.99,
        discount_lambda=0.95,
        horizon=15,
        free_nats=3,
        kl_beta=1.0,
    ):
        self.agent = agent
        self.model_lr = model_lr
        self.action_lr = action_lr
        self.value_lr = value_lr
        self.discount = discount
        self.discount_lambda = discount_lambda
        self.horizon = horizon
        self.free_nats = free_nats
        self.kl_beta = kl_beta

        self.model_modules = nn.ModuleList(
            [
                self.agent.observation_encoder,
                self.agent.observation_decoder,
                self.agent.representation,
                self.agent.transition,
                self.agent.reward_model,
            ]
        )
        self.model_optimizer = optim.Adam(
            self.model_modules.parameters(),
            lr=self.model_lr,
        )
        self.action_optimizer = optim.Adam(
            self.agent.action_decoder.parameters(),
            lr=self.action_lr,
        )
        self.value_optimizer = optim.Adam(
            self.agent.value_model.parameters(),
            lr=self.value_lr,
        )

    def update(self, samples):
        """
        :param samples: (seq_len, batch_size, ...)
        :return:
        """
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
            self.free_nats,  # to prevent penalize small KL divergence
        )
        model_loss = image_loss + reward_loss + self.kl_beta * kl_div

        # produce a gradient-free posterior for action network
        with torch.no_grad():
            # TODO: handle pcont
            flat_post = apply_states(
                posterior, lambda x: x.reshape(seq_len + batch_size, -1)
            )
        with FreezeParameters(self.model_modules):
            imagine_states, _ = self.agent.rollout.rollout_policy(
                self.horizon, self.agent.policy, posterior
            )  # image_states of shape (horizon, seq_len * b, state_dim)
        imagine_features = get_feat(imagine_states)

        # get rewards from imagine features
        with FreezeParameters(self.model_modules + [self.agent.value_model]):
            imagine_reward_dists = self.agent.reward_model(imagine_features).mean
            imagine_value_pred = self.agent.value_model(imagine_features).mean
        # TODO: handle pcont
        # compute value estimate
        discount_arr = self.discount * torch.ones_like(imagine_reward_dists)
        value_estimates = self.compute_value_estimate(
            imagine_reward_dists[:-1],
            imagine_value_pred[:-1],
            discount_arr[:-1],
            bootstrap=imagine_value_pred[-1],
            lambda_=self.discount_lambda,
        )
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        actor_loss = -torch.mean(discount * value_estimates)

        with torch.no_grad():
            value_features = imagine_features[:-1].detach()
            value_discount = discount.detach()
            value_target = value_estimates.detach()
        value_pred = self.agent.value_model(value_features)
        log_prob = value_pred.log_prob(value_target)
        value_loss = -torch.mean(value_discount * log_prob.unsqueeze(2))

        # log
        ...
        return model_loss, actor_loss, value_loss

    def compute_value_estimate(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        discount: torch.Tensor,
        bootstrap: torch.Tensor,
        lambda_: float,
    ):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns
