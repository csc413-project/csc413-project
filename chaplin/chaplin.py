import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim
import wandb

from models.agent import ChaplinAgent
from models.rssm import get_feat, get_dist, apply_states
from utils import denormalize_images, merge_images_in_chunks, FreezeParameters


class Chaplin:

    def __init__(
        self,
        agent: ChaplinAgent,
        model_lr=6e-4,
        action_lr=8e-5,
        value_lr=5e-4,
        imitator_lr=8e-5,
        discount=0.99,
        discount_lambda=0.95,
        horizon=15,
        free_nats=3,
        kl_beta=1.0,
        ppo_epsilon=0.2,
        ppo_value_loss_coef=1.0,
        ppo_entropy_coef=0.01,
        device: str = "cuda",
    ):
        self.agent = agent.to(device)
        self.discount = discount
        self.discount_lambda = discount_lambda
        self.horizon = horizon
        self.free_nats = free_nats
        self.kl_beta = kl_beta
        self.device = device
        self.ppo_epsilon = ppo_epsilon
        self.ppo_value_loss_coef = ppo_value_loss_coef
        self.ppo_entropy_coef = ppo_entropy_coef

        self.ppo_training_steps = 0
        self.dreamer_training_steps = 0

        self.model_modules = nn.ModuleList(
            [
                self.agent.observation_encoder,
                self.agent.observation_decoder,
                self.agent.rssm,
                self.agent.reward_model,
                # self.agent.imitator_encoder,
                # self.agent.imitator_policy,
                # self.agent.imitator_value,
            ]
        )

        self.model_optimizer = optim.Adam(
            self.model_modules.parameters(),
            lr=model_lr,
        )
        self.action_optimizer = optim.Adam(
            self.agent.action_decoder.parameters(),
            lr=action_lr,
        )
        self.value_optimizer = optim.Adam(
            self.agent.value_model.parameters(),
            lr=value_lr,
        )

        self.imitator_optimizer = optim.Adam(
            [
                p
                for group in [
                    self.agent.imitator_policy.parameters(),
                    self.agent.imitator_value.parameters(),
                ]
                for p in group
            ],
            lr=imitator_lr,
        )

    def update_ppo(
        self,
        observations,
        actions,
        rewards,
        values,
        log_probs_old,
        rewards_to_go,
        advantages,
    ):
        """
        Inputs are of shape (batch_size, seq_len, ...)
        """
        observations = torch.transpose(observations, 0, 1)
        actions = torch.transpose(actions, 0, 1)
        rewards = torch.transpose(rewards, 0, 1)
        values = torch.transpose(values, 0, 1)
        log_probs_old = torch.transpose(log_probs_old, 0, 1)
        rewards_to_go = torch.transpose(rewards_to_go, 0, 1)
        advantages = torch.transpose(advantages, 0, 1)

        # self.model_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        self.action_optimizer.zero_grad()
        ppo_loss = self.calculate_ppo_loss(
            observations,
            actions,
            rewards,
            values,
            log_probs_old,
            rewards_to_go,
            advantages,
        )
        ppo_loss.backward()
        nn.utils.clip_grad_norm_(self.model_modules.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.agent.action_decoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.agent.value_model.parameters(), 1.0)
        # self.model_optimizer.step()
        self.value_optimizer.step()
        self.action_optimizer.step()

        self.ppo_training_steps += 1

    def calculate_ppo_loss(
        self,
        observations,
        actions,
        rewards,
        values,
        log_probs_old,
        rewards_to_go,
        advantages,
    ):
        """
        Inputs are of shape (seq_len, batch_size, ...)
        """

        epsilon = self.ppo_epsilon
        c1 = self.ppo_value_loss_coef
        c2 = self.ppo_entropy_coef

        seq_len, batch_size = observations.shape[:2]
        with FreezeParameters(self.model_modules):
            # compute embedding
            obs_embed = self.agent.observation_encoder(observations)
            # init prev state
            prev_state = self.agent.rssm.create_initial_state(
                batch_size, device=self.device
            )
            # Get prior and posterior and initialize stuff
            prior, posterior = self.agent.rssm.observe(
                seq_len, obs_embed, actions, prev_state
            )
            features = get_feat(posterior)

        # compute surrogate loss
        dist_now = self.agent.action_decoder(features)
        logprob_now = dist_now.log_prob(actions)
        ratio = torch.exp(logprob_now - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # value loss
        new_values = self.agent.value_model(features).squeeze(-1)
        value_loss = ((new_values - rewards_to_go) ** 2).mean()

        # entropy loss
        dist_entropy = dist_now.entropy().mean()

        # total loss
        ppo_loss = policy_loss + c1 * value_loss - c2 * dist_entropy

        wandb.log(
            {
                "ppo_training_steps": self.ppo_training_steps,
                "ppo/total_loss": ppo_loss.item(),
                "ppo/policy_loss": policy_loss.item(),
                "ppo/value_loss": value_loss.item(),
                "ppo/entropy": dist_entropy.item(),
            },
            commit=False,
        )
        return ppo_loss

    def update_dreamer(
        self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ):
        # swap batch and seq_len
        observations = torch.transpose(observations, 0, 1)
        actions = torch.transpose(actions, 0, 1)
        rewards = torch.transpose(rewards, 0, 1).unsqueeze(-1)

        self.model_optimizer.zero_grad()
        model_loss = self.calculate_loss(observations, actions, rewards)

        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_modules.parameters(), 100.0)
        self.model_optimizer.step()
        self.dreamer_training_steps += 1

    def calculate_loss(self, observations, actions, rewards):
        """
        Note: With observation[0], the agent took actions[0] and get rewards[0].
        Assume they are float32 and on the proper device.
        :param observations: (seq_len, batch_size, img_shape), normalized to [-0.5, 0.5]
        :param actions: (seq_len, batch_size, act_shape)
        :param rewards: (seq_len, batch_size)
        :return:
        """
        assert self.agent.explore is True
        seq_len, batch_size = observations.shape[:2]

        obs_embed = self.agent.observation_encoder(observations)

        # init prev state
        prev_state = self.agent.rssm.create_initial_state(
            batch_size, device=self.device
        )
        # get prior and posterior and initialize stuff
        prior, posterior = self.agent.rssm.observe(
            seq_len, obs_embed, actions, prev_state
        )  # (seq_len, batch_size, state_dim)
        prior_dist, posterior_dist = get_dist(prior), get_dist(posterior)
        features = get_feat(posterior)

        image_pred = self.agent.observation_decoder(features)
        image_loss = -torch.mean(image_pred.log_prob(observations))
        reward_pred = self.agent.reward_model(features)
        reward_loss = -torch.mean(reward_pred.log_prob(rewards))
        # TODO: also add pcont
        kl_div = torch.maximum(
            torch.mean(td.kl_divergence(posterior_dist, prior_dist)),
            torch.tensor(self.free_nats, dtype=torch.float32, device=self.device),
        )  # to prevent penalize small KL divergence
        model_loss = image_loss + reward_loss + self.kl_beta * kl_div

        # logging
        with torch.no_grad():
            wandb.log(
                {
                    "dreamer_training_steps": self.dreamer_training_steps,
                    "dreamer/model_loss": image_loss.item(),
                    "dreamer/reward_loss": reward_loss.item(),
                    "dreamer/kl_div": kl_div.item(),
                },
                commit=False,
            )
            # log images
            if self.dreamer_training_steps % 500 == 0:
                i = torch.randint(0, batch_size, (1,)).item()
                # reconstruction quality
                ground_truths = np.transpose(
                    denormalize_images(observations[:, i].detach().cpu().numpy()),
                    (0, 2, 3, 1),
                )
                pred_images = np.transpose(
                    denormalize_images(image_pred.mean[:, i].detach().cpu().numpy()),
                    (0, 2, 3, 1),
                )
                reconstruction_demo = merge_images_in_chunks(ground_truths, pred_images)
                # prediction + reconstruction
                # feed 5 obs to the model, and predict the next 45 obs
                given_obs = observations[:5, i]
                obs_embed = self.agent.observation_encoder(given_obs)
                prev_state = self.agent.rssm.create_initial_state(1, device=self.device)
                _, posterior = self.agent.rssm.observe(
                    5,
                    obs_embed.reshape(5, 1, -1),
                    actions[:5, i].reshape(5, 1, -1),
                    prev_state,
                )
                posterior = apply_states(posterior, lambda x: x[-1])
                states = self.agent.rssm.follow(
                    45, actions[5:, i].reshape(45, 1, -1), posterior
                )
                features = get_feat(states)
                pred_images = self.agent.observation_decoder(features)
                pred_images = np.transpose(
                    np.squeeze(
                        denormalize_images(pred_images.mean.detach().cpu().numpy()),
                        axis=1,
                    ),
                    (0, 2, 3, 1),
                )
                # first row is observed
                prediction_demo = merge_images_in_chunks(
                    ground_truths[5:], pred_images, chunk_size=5
                )

                wandb.log(
                    {
                        "dreamer/reconstruction": wandb.Image(reconstruction_demo),
                        "dreamer/prediction": wandb.Image(prediction_demo),
                    }
                )
        return model_loss

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

    # def imitation_update(self, observations, actions, rewards, values):
    #     observations = torch.transpose(observations, 0, 1)
    #     actions = torch.transpose(actions, 0, 1)
    #     rewards = torch.transpose(rewards, 0, 1).unsqueeze(-1)
    #     values = torch.transpose(values, 0, 1).unsqueeze(-1)
    #
    #     self.model_optimizer.zero_grad()
    #     self.action_optimizer.zero_grad()
    #     self.value_optimizer.zero_grad()
    #     self.imitator_optimizer.zero_grad()
    #
    #     # Unbiased state embedding
    #     seq_len, batch_size = observations.shape[:2]
    #
    #     obs_embed = self.agent.observation_encoder(observations)
    #
    #     # init prev state
    #     prev_state = self.agent.rssm.create_initial_state(
    #         batch_size, device=self.device
    #     )
    #     # get prior and posterior and initialize stuff
    #     prior, posterior = self.agent.rssm.observe(
    #         seq_len, obs_embed, actions, prev_state
    #     )  # (seq_len, batch_size, state_dim)
    #
    #     features = get_feat(posterior)
    #
    #     # Embed the actions, values
    #     imitator_features = self.agent.imitator_encoder(obs_embed, actions, values)
    #
    #     # using this to imitate action/values
    #     # Only compute loss for 50 steps +
    #     imitator_features = imitator_features[50:]
    #     imitation_action, imitation_value = self.agent.imitator_action_head(
    #         imitator_features
    #     )
    #
    #     action_redacted = actions[:, 50:, :]
    #     value_redacted = values[:, 50:]
    #
    #     # action_
    #
    #     # compute mse
    #     action_loss = nn.MSELoss()(imitation_action, action_redacted)
    #     value_loss = nn.MSELoss()(imitation_value, value_redacted)
    #
    #     # compute loss
    #     loss = action_loss + value_loss
    #     return loss
