import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim
import wandb

from models.agent import AgentModel
from models.rssm import get_feat, get_dist, apply_states
from utils import FreezeParameters, denormalize_images, merge_images_in_chunks


class Dreamer:

    def __init__(
        self,
        agent: AgentModel,
        model_lr=6e-4,
        action_lr=8e-5,
        value_lr=8e-5,
        discount=0.99,
        discount_lambda=0.95,
        horizon=15,
        free_nats=3,
        kl_beta=1.0,
        device: str = "cuda",
    ):
        self.agent = agent.to(device)
        self.discount = discount
        self.discount_lambda = discount_lambda
        self.horizon = horizon
        self.free_nats = free_nats
        self.kl_beta = kl_beta
        self.device = device

        self.model_modules = nn.ModuleList(
            [
                self.agent.observation_encoder,
                self.agent.observation_decoder,
                self.agent.rssm,
                self.agent.reward_model,
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

        self.training_steps = 0

    def update(
        self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ):
        # swap batch and seq_len
        observations = torch.transpose(observations, 0, 1)
        actions = torch.transpose(actions, 0, 1)
        rewards = torch.transpose(rewards, 0, 1).unsqueeze(-1)

        self.model_optimizer.zero_grad()
        self.action_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        model_loss, actor_loss, value_loss = self.calculate_loss(
            observations, actions, rewards
        )

        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_modules.parameters(), 100.0)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.action_decoder.parameters(), 100.0)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.value_model.parameters(), 100.0)

        self.model_optimizer.step()
        self.action_optimizer.step()
        self.value_optimizer.step()

        self.training_steps += 1

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

        # produce a gradient-free posterior for action network
        with torch.no_grad():
            # TODO: handle pcont
            flat_post = apply_states(
                posterior, lambda x: x.reshape(seq_len * batch_size, -1)
            )
        with FreezeParameters(self.model_modules):
            imagine_states, _ = self.agent.rssm.imagine(
                self.horizon, self.agent.policy, flat_post
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
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr])
        discount = torch.cumprod(discount_arr[:-2], 0)
        actor_loss = -torch.mean(discount * value_estimates)

        with torch.no_grad():
            value_features = imagine_features[:-1].detach()
            value_discount = discount.detach()
            value_target = value_estimates.detach()
        value_pred = self.agent.value_model(value_features)
        log_prob = value_pred.log_prob(value_target)
        value_loss = -torch.mean(value_discount * log_prob.unsqueeze(2))

        # logging
        with torch.no_grad():
            wandb.log(
                {
                    "training_steps": self.training_steps,
                    "train/model_loss": image_loss.item(),
                    "train/reward_loss": reward_loss.item(),
                    "train/kl_div": kl_div.item(),
                    "train/actor_loss": actor_loss.item(),
                    "train/value_loss": value_loss.item(),
                },
                commit=False,
            )
            # log images
            if self.training_steps % 500 == 0:
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
                        "train/reconstruction": wandb.Image(reconstruction_demo),
                        "train/prediction": wandb.Image(prediction_demo),
                    }
                )

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
