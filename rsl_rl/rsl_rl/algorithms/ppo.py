# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 use_cvar=False,
                 cvar_alpha=0.1,
                 cvar_tail_weight=3.0,
                 cvar_min_completed_episodes=16,
                 cvar_use_base_dr_only=True,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Optional CVaR reweighting over completed episodic returns in rollout.
        self.use_cvar = bool(use_cvar)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_tail_weight = float(cvar_tail_weight)
        self.cvar_min_completed_episodes = int(cvar_min_completed_episodes)
        self.cvar_use_base_dr_only = bool(cvar_use_base_dr_only)
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError(f"cvar_alpha must be in (0, 1], got {self.cvar_alpha}")
        if self.cvar_tail_weight <= 0.0:
            raise ValueError(f"cvar_tail_weight must be > 0, got {self.cvar_tail_weight}")
        if self.cvar_min_completed_episodes < 1:
            raise ValueError(
                f"cvar_min_completed_episodes must be >= 1, got {self.cvar_min_completed_episodes}"
            )
        self.last_cvar_stats = self._new_cvar_stats()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, transition_episode_ids=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.sample_weights = torch.ones_like(self.transition.rewards, device=self.device)
        if transition_episode_ids is None:
            self.transition.episode_ids = None
        else:
            self.transition.episode_ids = transition_episode_ids.to(self.device).view(-1)
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _new_cvar_stats(self):
        return {
            "use_cvar": float(self.use_cvar),
            "cvar_alpha": float(self.cvar_alpha),
            "cvar_tail_weight": float(self.cvar_tail_weight),
            "completed_episodes": 0.0,
            "tail_episodes": 0.0,
            "tail_threshold_return": None,
            "mean_tail_return": None,
            "tail_sample_fraction": 0.0,
            "applied": 0.0,
        }

    def prepare_cvar_sample_weights(self, completed_episode_ids=None, completed_episode_returns=None):
        stats = self._new_cvar_stats()
        self.storage.sample_weights.fill_(1.0)

        if not self.use_cvar:
            self.last_cvar_stats = stats
            return stats

        if completed_episode_ids is None or completed_episode_returns is None:
            self.last_cvar_stats = stats
            return stats

        episode_ids = torch.as_tensor(completed_episode_ids, device=self.device, dtype=torch.long).flatten()
        episode_returns = torch.as_tensor(completed_episode_returns, device=self.device, dtype=torch.float).flatten()
        if episode_ids.numel() != episode_returns.numel():
            raise ValueError(
                "completed_episode_ids and completed_episode_returns must have the same length"
            )

        valid_mask = torch.isfinite(episode_returns)
        episode_ids = episode_ids[valid_mask]
        episode_returns = episode_returns[valid_mask]
        completed_episodes = int(episode_returns.numel())
        stats["completed_episodes"] = float(completed_episodes)

        if completed_episodes < self.cvar_min_completed_episodes:
            self.last_cvar_stats = stats
            return stats

        threshold = torch.quantile(episode_returns, self.cvar_alpha)
        tail_episode_mask = episode_returns <= threshold
        tail_episode_ids = torch.unique(episode_ids[tail_episode_mask])
        tail_episode_returns = episode_returns[tail_episode_mask]

        stats["tail_threshold_return"] = float(threshold.item())
        stats["tail_episodes"] = float(tail_episode_ids.numel())
        if tail_episode_returns.numel() > 0:
            stats["mean_tail_return"] = float(tail_episode_returns.mean().item())
        if tail_episode_ids.numel() == 0:
            self.last_cvar_stats = stats
            return stats

        flat_episode_ids = self.storage.episode_ids.view(-1)
        valid_transition_mask = flat_episode_ids >= 0
        tail_episode_ids = torch.sort(tail_episode_ids).values
        search_pos = torch.searchsorted(tail_episode_ids, flat_episode_ids.clamp(min=0))
        in_bounds = search_pos < tail_episode_ids.numel()
        safe_pos = torch.clamp(search_pos, max=tail_episode_ids.numel() - 1)
        is_tail_transition = (
            valid_transition_mask
            & in_bounds
            & (tail_episode_ids[safe_pos] == flat_episode_ids)
        )

        flat_sample_weights = self.storage.sample_weights.view(-1)
        flat_sample_weights[is_tail_transition] = self.cvar_tail_weight

        stats["tail_sample_fraction"] = float(is_tail_transition.float().mean().item())
        stats["applied"] = 1.0
        self.last_cvar_stats = stats
        return stats

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, sample_weights_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_terms = torch.max(surrogate, surrogate_clipped)
                sample_weights = torch.squeeze(sample_weights_batch)
                if sample_weights.shape != surrogate_terms.shape:
                    sample_weights = sample_weights.reshape_as(surrogate_terms)
                sample_weights = torch.clamp(sample_weights, min=0.0)
                surrogate_loss = torch.sum(surrogate_terms * sample_weights) / torch.clamp(sample_weights.sum(), min=1e-8)

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
