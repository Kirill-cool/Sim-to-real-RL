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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.modules import DynamicsEncoder, ForwardDynamicsModel, ThetaDecoder
from rsl_rl.storage import UpesiReplayBuffer
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.upesi_cfg = train_cfg.get("upesi", {})
        self.device = device
        self.env = env

        self.base_actor_obs_dim = int(self.env.num_obs)
        if self.env.num_privileged_obs is not None:
            self.base_critic_obs_dim = int(self.env.num_privileged_obs)
        else:
            self.base_critic_obs_dim = int(self.env.num_obs)

        self.upesi_enabled = bool(self.upesi_cfg.get("enabled", False))
        self.upesi_last_stats = None
        self.upesi_out_of_bounds_warning_count = 0
        self.upesi_encoder_frozen = False
        self.upesi_freeze_encoder_after_iter = -1
        self.upesi_obs_dyn_indices = None
        self.upesi_obs_dyn_dim = 0

        if self.upesi_enabled:
            if not hasattr(self.env, "get_current_upesi_theta"):
                raise AttributeError(
                    "UPESI is enabled, but environment does not implement get_current_upesi_theta()."
                )
            self.upesi_embedding_dim = int(self.upesi_cfg.get("embedding_dim", 16))
            self.upesi_theta_dim = int(self.upesi_cfg.get("theta_dim", 2))
            if self.upesi_theta_dim != 2:
                raise ValueError(f"UPESI expects theta_dim=2, got {self.upesi_theta_dim}")
            self.upesi_theta_keys = list(self.upesi_cfg.get("theta_keys", ["added_mass", "friction_coeff"]))
            if self.upesi_theta_keys != ["added_mass", "friction_coeff"]:
                raise ValueError(
                    "UPESI requires theta_keys to be exactly ['added_mass', 'friction_coeff'] in fixed order"
                )
            self.upesi_theta_min = torch.tensor(
                self.upesi_cfg.get("theta_min", [-3.0, 0.1]), dtype=torch.float, device=self.device
            ).view(1, -1)
            self.upesi_theta_max = torch.tensor(
                self.upesi_cfg.get("theta_max", [3.0, 1.5]), dtype=torch.float, device=self.device
            ).view(1, -1)
            if self.upesi_theta_min.shape[-1] != self.upesi_theta_dim or self.upesi_theta_max.shape[-1] != self.upesi_theta_dim:
                raise ValueError("UPESI theta_min/theta_max shape mismatch with theta_dim")

            self.upesi_theta_eps = 1e-6
            self.upesi_detach_encoder_for_ppo = bool(self.upesi_cfg.get("detach_encoder_for_ppo", True))
            self.upesi_predict_delta_obs = bool(self.upesi_cfg.get("predict_delta_obs", True))
            self.upesi_dynamics_include_base_lin_vel = bool(
                self.upesi_cfg.get("dynamics_include_base_lin_vel", True)
            )
            self.upesi_lambda_rec = float(self.upesi_cfg.get("lambda_rec", 0.1))
            self.upesi_dynamics_batch_size = int(self.upesi_cfg.get("dynamics_batch_size", 2048))
            self.upesi_dynamics_updates_per_iter = int(self.upesi_cfg.get("dynamics_updates_per_iter", 4))
            self.upesi_identification_steps = int(self.upesi_cfg.get("identification_steps", 0))
            self.upesi_freeze_encoder_after_iter = int(
                self.upesi_cfg.get("freeze_encoder_after_iter", -1)
            )

            actor_obs_dim = self.base_actor_obs_dim + self.upesi_embedding_dim
            critic_obs_dim = self.base_critic_obs_dim + self.upesi_embedding_dim
        else:
            actor_obs_dim = self.base_actor_obs_dim
            critic_obs_dim = self.base_critic_obs_dim

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            actor_obs_dim,
            critic_obs_dim,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        storage_critic_obs_dim = critic_obs_dim if self.env.num_privileged_obs is not None else None
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [actor_obs_dim],
            [storage_critic_obs_dim],
            [self.env.num_actions],
        )

        if self.upesi_enabled:
            activation_name = self.policy_cfg.get("activation", "elu")
            self.upesi_encoder = DynamicsEncoder(
                theta_dim=self.upesi_theta_dim,
                embedding_dim=self.upesi_embedding_dim,
                hidden_dims=(64, 64),
                activation=activation_name,
            ).to(self.device)
            self.upesi_forward_model = ForwardDynamicsModel(
                obs_dim=self.base_actor_obs_dim,
                action_dim=self.env.num_actions,
                embedding_dim=self.upesi_embedding_dim,
                hidden_dims=(256, 256),
                output_dim=self.base_actor_obs_dim,
                activation=activation_name,
            ).to(self.device)
            self.upesi_decoder = ThetaDecoder(
                embedding_dim=self.upesi_embedding_dim,
                theta_dim=self.upesi_theta_dim,
                hidden_dims=(64, 64),
                activation=activation_name,
            ).to(self.device)

            upesi_params = list(self.upesi_encoder.parameters())
            upesi_params += list(self.upesi_forward_model.parameters())
            upesi_params += list(self.upesi_decoder.parameters())
            self.upesi_optimizer = torch.optim.Adam(
                upesi_params,
                lr=float(self.upesi_cfg.get("dynamics_lr", 1e-3)),
            )

            if self.upesi_identification_steps > 0:
                self.upesi_identification_optimizer = torch.optim.Adam(
                    list(self.upesi_encoder.parameters()) + list(self.upesi_decoder.parameters()),
                    lr=float(self.upesi_cfg.get("identification_lr", 1e-3)),
                )
            else:
                self.upesi_identification_optimizer = None

            self.upesi_buffer = UpesiReplayBuffer(
                capacity=int(self.upesi_cfg.get("buffer_size", 200000)),
                obs_dim=self.base_actor_obs_dim,
                action_dim=self.env.num_actions,
                theta_dim=self.upesi_theta_dim,
                device=self.device,
            )
            self.upesi_obs_dyn_indices = self._build_upesi_obs_dyn_indices()
            self.upesi_obs_dyn_dim = int(self.upesi_obs_dyn_indices.numel())
            if self.upesi_obs_dyn_dim <= 0:
                raise ValueError("[UPESI] obs_dyn_indices is empty.")
            if self.upesi_obs_dyn_dim >= int(self.base_actor_obs_dim):
                raise ValueError(
                    "[UPESI] obs_dyn_indices must be a strict subset of observation dimensions."
                )
            print(
                f"[UPESI] Dynamics obs subset dim: {self.upesi_obs_dyn_dim}/"
                f"{self.base_actor_obs_dim} "
                f"(include_base_lin_vel={int(self.upesi_dynamics_include_base_lin_vel)})"
            )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_cdr_window_success_rate = None
        self.use_cvar = bool(getattr(self.alg, "use_cvar", False))
        self.cvar_use_base_dr_only = bool(getattr(self.alg, "cvar_use_base_dr_only", True))
        if self.use_cvar and self.cvar_use_base_dr_only and bool(getattr(self.env, "cdr_enabled", False)):
            raise ValueError(
                "CVaR is configured with cvar_use_base_dr_only=True, but CDR is enabled in the environment. "
                "Use a base-DR task (without CDR) for CVaR experiments or disable this guard explicitly."
            )
        self.last_cvar_stats = getattr(self.alg, "last_cvar_stats", None)
        self.current_episode_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        self.next_episode_id = int(self.env.num_envs)

        _, _ = self.env.reset()

    def _set_upesi_encoder_trainable(self, trainable, reason=None):
        if not self.upesi_enabled:
            return
        for param in self.upesi_encoder.parameters():
            param.requires_grad_(bool(trainable))
        self.upesi_encoder_frozen = not bool(trainable)
        if self.upesi_encoder_frozen:
            self.upesi_encoder.eval()
            if reason is not None:
                print(f"[UPESI] Encoder frozen. {reason}")
        else:
            if reason is not None:
                print(f"[UPESI] Encoder unfrozen. {reason}")

    def _maybe_freeze_upesi_encoder(self, iteration):
        if not self.upesi_enabled:
            return
        if self.upesi_encoder_frozen:
            return
        if self.upesi_freeze_encoder_after_iter < 0:
            return
        if int(iteration) >= int(self.upesi_freeze_encoder_after_iter):
            self._set_upesi_encoder_trainable(
                False,
                reason=(
                    f"iteration={int(iteration)} >= "
                    f"freeze_encoder_after_iter={int(self.upesi_freeze_encoder_after_iter)}"
                ),
            )

    def _build_upesi_obs_dyn_indices(self):
        obs_dim = int(self.base_actor_obs_dim)
        num_actions = int(self.env.num_actions)
        min_expected_obs_dim = 12 + 3 * num_actions
        if obs_dim < min_expected_obs_dim:
            raise ValueError(
                "[UPESI] Observation layout is smaller than expected "
                f"(obs_dim={obs_dim}, expected_at_least={min_expected_obs_dim})."
            )

        dyn_indices = []
        # Optional physically meaningful term.
        if self.upesi_dynamics_include_base_lin_vel:
            dyn_indices.extend(range(0, 3))
        # base angular velocity
        dyn_indices.extend(range(3, 6))
        # projected gravity / orientation proxy
        dyn_indices.extend(range(6, 9))
        # Exclude commands [9:12]
        # joint positions
        dof_pos_start = 12
        dof_pos_end = dof_pos_start + num_actions
        dyn_indices.extend(range(dof_pos_start, dof_pos_end))
        # joint velocities
        dof_vel_start = dof_pos_end
        dof_vel_end = dof_vel_start + num_actions
        dyn_indices.extend(range(dof_vel_start, dof_vel_end))
        # Exclude previous actions and any auxiliary tails.

        return torch.as_tensor(dyn_indices, dtype=torch.long, device=self.device)

    def _compute_upesi_dyn_loss(self, next_obs_hat, next_obs):
        if self.upesi_obs_dyn_indices is None:
            return torch.mean((next_obs_hat - next_obs) ** 2)
        pred_dyn = next_obs_hat.index_select(dim=-1, index=self.upesi_obs_dyn_indices)
        target_dyn = next_obs.index_select(dim=-1, index=self.upesi_obs_dyn_indices)
        return torch.mean((pred_dyn - target_dyn) ** 2)

    def _to_float(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().mean().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _get_upesi_theta_norm(self):
        if hasattr(self.env, "get_current_upesi_theta_norm"):
            theta_norm = self.env.get_current_upesi_theta_norm(
                theta_min=self.upesi_theta_min.view(-1),
                theta_max=self.upesi_theta_max.view(-1),
                eps=self.upesi_theta_eps,
            )
            return theta_norm.to(self.device)

        theta = self.env.get_current_upesi_theta().to(self.device)
        theta_min = self.upesi_theta_min
        theta_max = self.upesi_theta_max
        theta_norm = 2.0 * (theta - theta_min) / (theta_max - theta_min + self.upesi_theta_eps) - 1.0
        out_of_bounds = (theta < (theta_min - self.upesi_theta_eps)) | (theta > (theta_max + self.upesi_theta_eps))
        if bool(torch.any(out_of_bounds).item()) and self.upesi_out_of_bounds_warning_count < 20:
            self.upesi_out_of_bounds_warning_count += 1
            print(
                f"[UPESI] Warning: theta out of global bounds "
                f"(warning {self.upesi_out_of_bounds_warning_count}/20)."
            )
        return theta_norm

    def _build_policy_observations(self, obs, critic_obs, theta_norm):
        alpha = self.upesi_encoder(theta_norm)
        if self.upesi_detach_encoder_for_ppo:
            alpha_policy = alpha.detach()
        else:
            alpha_policy = alpha

        obs_policy = torch.cat((obs, alpha_policy), dim=-1)
        critic_obs_policy = torch.cat((critic_obs, alpha_policy), dim=-1)
        return obs_policy, critic_obs_policy

    def _move_upesi_modules_to_device(self, device):
        if self.upesi_enabled:
            self.upesi_encoder.to(device)
            self.upesi_forward_model.to(device)
            self.upesi_decoder.to(device)

    def _compute_identification_loss(self, obs, actions, next_obs, alpha_vector):
        alpha_batch = alpha_vector.view(1, -1).expand(obs.shape[0], -1)
        model_out = self.upesi_forward_model(obs, actions, alpha_batch)
        if self.upesi_predict_delta_obs:
            next_obs_hat = obs + model_out
        else:
            next_obs_hat = model_out
        return self._compute_upesi_dyn_loss(next_obs_hat, next_obs)

    def identify_alpha(self, transitions, identification_steps=None, identification_lr=None, return_diagnostics=False):
        if not self.upesi_enabled:
            raise ValueError("identify_alpha is only available when UPESI is enabled.")

        if isinstance(transitions, dict):
            obs = transitions["obs"]
            actions = transitions["action"] if "action" in transitions else transitions["actions"]
            next_obs = transitions["next_obs"]
        elif isinstance(transitions, (tuple, list)) and len(transitions) == 3:
            obs, actions, next_obs = transitions
        else:
            raise ValueError(
                "transitions must be dict with keys {obs, action/actions, next_obs} "
                "or a tuple/list (obs, action, next_obs)"
            )

        obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float, device=self.device)

        if obs.ndim != 2 or actions.ndim != 2 or next_obs.ndim != 2:
            raise ValueError("obs, action, next_obs must be rank-2 tensors [N, dim]")
        if obs.shape != next_obs.shape:
            raise ValueError("obs and next_obs must have identical shapes")
        if obs.shape[0] != actions.shape[0]:
            raise ValueError("obs and action must have the same batch dimension")

        steps = int(self.upesi_identification_steps if identification_steps is None else identification_steps)
        lr = float(self.upesi_cfg.get("identification_lr", 1e-3) if identification_lr is None else identification_lr)
        steps = max(1, steps)

        modules_to_freeze = [
            self.alg.actor_critic,
            self.upesi_encoder,
            self.upesi_decoder,
            self.upesi_forward_model,
        ]

        module_training_states = [module.training for module in modules_to_freeze]
        param_requires_grad = []
        for module in modules_to_freeze:
            module.eval()
            for param in module.parameters():
                param_requires_grad.append(param.requires_grad)
                param.requires_grad_(False)

        alpha_param = torch.zeros(
            (self.upesi_embedding_dim,),
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([alpha_param], lr=lr)
        with torch.no_grad():
            loss_before = float(self._compute_identification_loss(obs, actions, next_obs, alpha_param.detach()).item())

        for _ in range(steps):
            loss = self._compute_identification_loss(obs, actions, next_obs, alpha_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        alpha_star = alpha_param.detach()
        with torch.no_grad():
            loss_after = float(self._compute_identification_loss(obs, actions, next_obs, alpha_star).item())

        for module, was_training in zip(modules_to_freeze, module_training_states):
            module.train(was_training)
        idx = 0
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad_(param_requires_grad[idx])
                idx += 1

        if return_diagnostics:
            eps = 1e-8
            diagnostics = {
                "identify_loss_before": loss_before,
                "identify_loss_after": loss_after,
                "identify_loss_ratio": loss_after / (loss_before + eps),
            }
            return alpha_star, diagnostics
        return alpha_star

    def get_oracle_alpha(self, device=None):
        if not self.upesi_enabled:
            raise ValueError("Oracle alpha is only available when UPESI is enabled.")
        target_device = device if device is not None else self.device
        self.upesi_encoder.eval()
        self.upesi_encoder.to(target_device)
        with torch.inference_mode():
            theta_norm = self._get_upesi_theta_norm().to(target_device)
            alpha_oracle = self.upesi_encoder(theta_norm)
        return alpha_oracle

    def get_oracle_inference_policy(self, device=None):
        if not self.upesi_enabled:
            raise ValueError("Oracle UPESI evaluation requires upesi.enabled = true.")
        self.alg.actor_critic.eval()
        self.upesi_encoder.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
            self.upesi_encoder.to(device)

        def _policy(obs):
            with torch.inference_mode():
                obs_local = obs
                if device is not None:
                    obs_local = obs_local.to(device)
                theta_norm = self._get_upesi_theta_norm().to(obs_local.device)
                alpha = self.upesi_encoder(theta_norm)
                obs_policy = torch.cat((obs_local, alpha), dim=-1)
                return self.alg.actor_critic.act_inference(obs_policy)

        return _policy

    def get_identified_inference_policy(
        self,
        transitions,
        device=None,
        identification_steps=None,
        identification_lr=None,
        return_diagnostics=False,
    ):
        if not self.upesi_enabled:
            raise ValueError("Identified UPESI evaluation requires upesi.enabled = true.")
        identify_result = self.identify_alpha(
            transitions=transitions,
            identification_steps=identification_steps,
            identification_lr=identification_lr,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            alpha_star, identify_diagnostics = identify_result
        else:
            alpha_star = identify_result
            identify_diagnostics = None
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        target_device = device if device is not None else self.device
        alpha_star = alpha_star.to(target_device).view(1, -1)

        def _policy(obs):
            with torch.inference_mode():
                obs_local = obs
                if device is not None:
                    obs_local = obs_local.to(device)
                alpha_batch = alpha_star.expand(obs_local.shape[0], -1)
                obs_policy = torch.cat((obs_local, alpha_batch), dim=-1)
                return self.alg.actor_critic.act_inference(obs_policy)

        if return_diagnostics:
            return _policy, alpha_star, identify_diagnostics
        return _policy, alpha_star

    def _append_upesi_rollout_batch(self, obs, actions, next_obs, dones, theta_norm):
        valid_mask = torch.logical_not(dones.bool())
        if valid_mask.ndim > 1:
            valid_mask = valid_mask.view(-1)
        if not bool(torch.any(valid_mask).item()):
            return
        self.upesi_buffer.add_batch(
            obs=obs[valid_mask],
            actions=actions[valid_mask],
            next_obs=next_obs[valid_mask],
            theta_norm=theta_norm[valid_mask],
        )

    def _train_upesi_modules(self):
        if not self.upesi_enabled:
            return None
        if len(self.upesi_buffer) < self.upesi_dynamics_batch_size:
            return {
                "loss_dyn": None,
                "loss_rec": None,
                "loss_total": None,
                "loss_ident": None,
                "buffer_size": float(len(self.upesi_buffer)),
            }

        if self.upesi_encoder_frozen:
            self.upesi_encoder.eval()
        else:
            self.upesi_encoder.train()
        self.upesi_forward_model.train()
        self.upesi_decoder.train()

        loss_dyn_accum = 0.0
        loss_rec_accum = 0.0
        loss_total_accum = 0.0

        for _ in range(self.upesi_dynamics_updates_per_iter):
            batch = self.upesi_buffer.sample_batch(self.upesi_dynamics_batch_size)
            obs_batch = batch["obs"]
            actions_batch = batch["actions"]
            next_obs_batch = batch["next_obs"]
            theta_norm_batch = batch["theta_norm"]

            alpha = self.upesi_encoder(theta_norm_batch)
            model_out = self.upesi_forward_model(obs_batch, actions_batch, alpha)
            if self.upesi_predict_delta_obs:
                next_obs_hat = obs_batch + model_out
            else:
                next_obs_hat = model_out

            loss_dyn = self._compute_upesi_dyn_loss(next_obs_hat, next_obs_batch)
            theta_hat_norm = self.upesi_decoder(alpha)
            loss_rec = torch.mean((theta_hat_norm - theta_norm_batch) ** 2)
            loss_total = loss_dyn + self.upesi_lambda_rec * loss_rec

            self.upesi_optimizer.zero_grad()
            loss_total.backward()
            self.upesi_optimizer.step()

            loss_dyn_accum += float(loss_dyn.item())
            loss_rec_accum += float(loss_rec.item())
            loss_total_accum += float(loss_total.item())

        loss_ident = None
        if self.upesi_identification_optimizer is not None and self.upesi_identification_steps > 0:
            loss_ident_accum = 0.0
            for _ in range(self.upesi_identification_steps):
                batch = self.upesi_buffer.sample_batch(self.upesi_dynamics_batch_size)
                theta_norm_batch = batch["theta_norm"]
                alpha = self.upesi_encoder(theta_norm_batch)
                theta_hat_norm = self.upesi_decoder(alpha)
                loss_ident_step = torch.mean((theta_hat_norm - theta_norm_batch) ** 2)
                self.upesi_identification_optimizer.zero_grad()
                loss_ident_step.backward()
                self.upesi_identification_optimizer.step()
                loss_ident_accum += float(loss_ident_step.item())
            loss_ident = loss_ident_accum / float(self.upesi_identification_steps)

        num_updates = max(1, self.upesi_dynamics_updates_per_iter)
        return {
            "loss_dyn": loss_dyn_accum / float(num_updates),
            "loss_rec": loss_rec_accum / float(num_updates),
            "loss_total": loss_total_accum / float(num_updates),
            "loss_ident": loss_ident,
            "buffer_size": float(len(self.upesi_buffer)),
        }

    def _collect_curriculum_metrics(self, ep_infos, iteration):
        metrics = {
            "iteration": int(iteration),
            "num_episodes": 0.0,
        }
        if len(ep_infos) == 0:
            return metrics

        weighted_sum = {}
        weight_sum = {}
        total_episodes = 0.0
        for ep_info in ep_infos:
            num_episodes = self._to_float(ep_info.get("num_episodes"))
            if num_episodes is None:
                num_episodes = 1.0
            if num_episodes <= 0.0:
                continue
            total_episodes += num_episodes

            for key, value in ep_info.items():
                val = self._to_float(value)
                if val is None:
                    continue
                weighted_sum[key] = weighted_sum.get(key, 0.0) + val * num_episodes
                weight_sum[key] = weight_sum.get(key, 0.0) + num_episodes

        for key, accum in weighted_sum.items():
            if weight_sum[key] > 0.0:
                metrics[key] = accum / weight_sum[key]

        metrics["num_episodes"] = total_episodes
        if "reward" in metrics:
            metrics["mean_reward"] = metrics["reward"]
        if "rew_tracking_lin_vel" in metrics and "rew_tracking_ang_vel" in metrics:
            metrics["tracking_reward"] = 0.5 * (metrics["rew_tracking_lin_vel"] + metrics["rew_tracking_ang_vel"])
        elif "rew_tracking_lin_vel" in metrics:
            metrics["tracking_reward"] = metrics["rew_tracking_lin_vel"]
        if "success_rate" not in metrics and "fall_rate" in metrics:
            metrics["success_rate"] = 1.0 - metrics["fall_rate"]
        return metrics

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            self._maybe_freeze_upesi_encoder(it)
            start = time.time()
            completed_episode_returns = []
            completed_episode_ids = []
            upesi_scale_stats = None
            if self.upesi_enabled:
                obs_stat_sum = 0.0
                obs_stat_sumsq = 0.0
                obs_stat_count = 0
                obs_stat_min = float("inf")
                obs_stat_max = float("-inf")

                alpha_stat_sum = 0.0
                alpha_stat_sumsq = 0.0
                alpha_stat_count = 0
                alpha_stat_min = float("inf")
                alpha_stat_max = float("-inf")
                alpha_norm_sum = 0.0
                alpha_norm_count = 0

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    if self.upesi_enabled:
                        theta_norm = self._get_upesi_theta_norm()
                        obs_policy, critic_obs_policy = self._build_policy_observations(obs, critic_obs, theta_norm)
                        alpha_policy = obs_policy[:, self.base_actor_obs_dim:]
                        obs_flat = obs.reshape(-1)
                        alpha_flat = alpha_policy.reshape(-1)
                        obs_stat_sum += float(obs_flat.sum().item())
                        obs_stat_sumsq += float((obs_flat * obs_flat).sum().item())
                        obs_stat_count += int(obs_flat.numel())
                        obs_stat_min = min(obs_stat_min, float(obs_flat.min().item()))
                        obs_stat_max = max(obs_stat_max, float(obs_flat.max().item()))
                        alpha_stat_sum += float(alpha_flat.sum().item())
                        alpha_stat_sumsq += float((alpha_flat * alpha_flat).sum().item())
                        alpha_stat_count += int(alpha_flat.numel())
                        alpha_stat_min = min(alpha_stat_min, float(alpha_flat.min().item()))
                        alpha_stat_max = max(alpha_stat_max, float(alpha_flat.max().item()))
                        alpha_norm_sum += float(alpha_policy.norm(dim=-1).sum().item())
                        alpha_norm_count += int(alpha_policy.shape[0])
                    else:
                        theta_norm = None
                        obs_policy, critic_obs_policy = obs, critic_obs

                    actions = self.alg.act(obs_policy, critic_obs_policy)
                    transition_episode_ids = self.current_episode_ids.clone()

                    next_obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    next_critic_obs = privileged_obs if privileged_obs is not None else next_obs
                    next_obs = next_obs.to(self.device)
                    next_critic_obs = next_critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                        transition_episode_ids=transition_episode_ids,
                    )

                    if self.upesi_enabled:
                        self._append_upesi_rollout_batch(obs, actions, next_obs, dones, theta_norm)

                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])

                    # Book keeping
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if new_ids.numel() > 0:
                        finished_returns = cur_reward_sum[new_ids].clone()
                        finished_episode_ids = transition_episode_ids[new_ids].clone()
                        completed_episode_returns.append(finished_returns)
                        completed_episode_ids.append(finished_episode_ids)
                        rewbuffer.extend(finished_returns.cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        num_finished = int(new_ids.numel())
                        self.current_episode_ids[new_ids] = torch.arange(
                            self.next_episode_id,
                            self.next_episode_id + num_finished,
                            device=self.device,
                            dtype=torch.long,
                        )
                        self.next_episode_id += num_finished

                    obs = next_obs
                    critic_obs = next_critic_obs

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if len(completed_episode_ids) > 0:
                    rollout_completed_episode_ids = torch.cat(completed_episode_ids, dim=0)
                    rollout_completed_episode_returns = torch.cat(completed_episode_returns, dim=0)
                else:
                    rollout_completed_episode_ids = torch.empty(0, dtype=torch.long, device=self.device)
                    rollout_completed_episode_returns = torch.empty(0, dtype=torch.float, device=self.device)

                cvar_stats = self.alg.prepare_cvar_sample_weights(
                    rollout_completed_episode_ids,
                    rollout_completed_episode_returns,
                )
                self.last_cvar_stats = cvar_stats

                if self.upesi_enabled:
                    theta_norm_for_returns = self._get_upesi_theta_norm()
                    _, critic_obs_policy = self._build_policy_observations(obs, critic_obs, theta_norm_for_returns)
                    self.alg.compute_returns(critic_obs_policy)
                else:
                    self.alg.compute_returns(critic_obs)

            if self.upesi_enabled and obs_stat_count > 0 and alpha_stat_count > 0:
                obs_mean = obs_stat_sum / float(obs_stat_count)
                obs_var = max(obs_stat_sumsq / float(obs_stat_count) - obs_mean * obs_mean, 0.0)
                obs_std = obs_var ** 0.5
                alpha_mean = alpha_stat_sum / float(alpha_stat_count)
                alpha_var = max(alpha_stat_sumsq / float(alpha_stat_count) - alpha_mean * alpha_mean, 0.0)
                alpha_std = alpha_var ** 0.5
                alpha_norm = alpha_norm_sum / float(max(alpha_norm_count, 1))
                upesi_scale_stats = {
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "obs_min": obs_stat_min,
                    "obs_max": obs_stat_max,
                    "alpha_mean": alpha_mean,
                    "alpha_std": alpha_std,
                    "alpha_min": alpha_stat_min,
                    "alpha_max": alpha_stat_max,
                    "alpha_norm": alpha_norm,
                }

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            upesi_stats = self._train_upesi_modules()
            self.upesi_last_stats = upesi_stats
            stop = time.time()
            learn_time = stop - start

            curriculum_metrics = self._collect_curriculum_metrics(ep_infos, it)
            curriculum_info = None
            if hasattr(self.env, "update_curriculum"):
                curriculum_info = self.env.update_curriculum(curriculum_metrics)

            if self.log_dir is not None:
                self.log(locals())
                if curriculum_info is not None:
                    self.writer.add_scalar('Curriculum/level', curriculum_info.get("level", 0.0), it)
                    self.writer.add_scalar('Curriculum/stage', curriculum_info.get("stage", 0), it)
                    self.writer.add_scalar('Curriculum/updated', 1.0 if curriculum_info.get("updated", False) else 0.0, it)
                    if curriculum_info.get("success_rate") is not None:
                        self.writer.add_scalar('Curriculum/success_rate_window', curriculum_info["success_rate"], it)
                    if curriculum_info.get("mean_reward") is not None:
                        self.writer.add_scalar('Curriculum/mean_reward_window', curriculum_info["mean_reward"], it)
                    if curriculum_info.get("tracking_reward") is not None:
                        self.writer.add_scalar('Curriculum/tracking_reward_window', curriculum_info["tracking_reward"], it)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        curriculum_info = locs.get('curriculum_info', None)
        cvar_stats = locs.get('cvar_stats', None)
        upesi_stats = locs.get('upesi_stats', None)
        upesi_scale_stats = locs.get('upesi_scale_stats', None)

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        if curriculum_info is not None:
            success_rate_window = curriculum_info.get("success_rate", None)
            if success_rate_window is not None:
                self.last_cdr_window_success_rate = success_rate_window
            elif self.last_cdr_window_success_rate is not None:
                success_rate_window = self.last_cdr_window_success_rate
            if success_rate_window is None:
                ep_string += f"""{'CDR window success_rate:':>{pad}} nan\n"""
            else:
                ep_string += f"""{'CDR window success_rate:':>{pad}} {success_rate_window:.4f}\n"""
            if curriculum_info.get("updated", None) is not None:
                ep_string += f"""{'CDR updated this iter:':>{pad}} {1 if curriculum_info.get('updated', False) else 0}\n"""

        if cvar_stats is not None:
            threshold = cvar_stats.get("tail_threshold_return", None)
            threshold_value = "nan" if threshold is None else f"{threshold:.4f}"
            tail_mean = cvar_stats.get("mean_tail_return", None)
            tail_mean_value = "nan" if tail_mean is None else f"{tail_mean:.4f}"
            ep_string += f"""{'CVaR enabled:':>{pad}} {int(cvar_stats.get('use_cvar', 0.0) > 0.5)}\n"""
            ep_string += f"""{'CVaR alpha:':>{pad}} {cvar_stats.get('cvar_alpha', 0.0):.4f}\n"""
            ep_string += f"""{'CVaR tail weight:':>{pad}} {cvar_stats.get('cvar_tail_weight', 1.0):.4f}\n"""
            ep_string += f"""{'CVaR completed episodes:':>{pad}} {int(cvar_stats.get('completed_episodes', 0.0))}\n"""
            ep_string += f"""{'CVaR tail episodes:':>{pad}} {int(cvar_stats.get('tail_episodes', 0.0))}\n"""
            ep_string += f"""{'CVaR return threshold:':>{pad}} {threshold_value}\n"""
            ep_string += f"""{'CVaR tail mean return:':>{pad}} {tail_mean_value}\n"""
            ep_string += f"""{'CVaR weighted sample frac:':>{pad}} {cvar_stats.get('tail_sample_fraction', 0.0):.4f}\n"""

        ep_string += f"""{'UPESI enabled:':>{pad}} {1 if self.upesi_enabled else 0}\n"""
        if self.upesi_enabled:
            ep_string += f"""{'UPESI encoder frozen:':>{pad}} {1 if self.upesi_encoder_frozen else 0}\n"""
            ep_string += f"""{'UPESI freeze iter:':>{pad}} {int(self.upesi_freeze_encoder_after_iter)}\n"""
        if upesi_stats is not None and self.upesi_enabled:
            loss_dyn = upesi_stats.get("loss_dyn", None)
            loss_rec = upesi_stats.get("loss_rec", None)
            loss_total = upesi_stats.get("loss_total", None)
            loss_ident = upesi_stats.get("loss_ident", None)
            buffer_size = upesi_stats.get("buffer_size", 0.0)
            ep_string += f"""{'UPESI buffer size:':>{pad}} {int(buffer_size)}\n"""
            ep_string += f"""{'UPESI loss_dyn:':>{pad}} {'nan' if loss_dyn is None else f'{loss_dyn:.6f}'}\n"""
            ep_string += f"""{'UPESI loss_rec:':>{pad}} {'nan' if loss_rec is None else f'{loss_rec:.6f}'}\n"""
            ep_string += f"""{'UPESI loss_total:':>{pad}} {'nan' if loss_total is None else f'{loss_total:.6f}'}\n"""
            if loss_ident is not None:
                ep_string += f"""{'UPESI loss_ident:':>{pad}} {loss_ident:.6f}\n"""
        if upesi_scale_stats is not None and self.upesi_enabled:
            ep_string += f"""{'UPESI obs_mean:':>{pad}} {upesi_scale_stats['obs_mean']:.6f}\n"""
            ep_string += f"""{'UPESI obs_std:':>{pad}} {upesi_scale_stats['obs_std']:.6f}\n"""
            ep_string += f"""{'UPESI obs_min:':>{pad}} {upesi_scale_stats['obs_min']:.6f}\n"""
            ep_string += f"""{'UPESI obs_max:':>{pad}} {upesi_scale_stats['obs_max']:.6f}\n"""
            ep_string += f"""{'UPESI alpha_mean:':>{pad}} {upesi_scale_stats['alpha_mean']:.6f}\n"""
            ep_string += f"""{'UPESI alpha_std:':>{pad}} {upesi_scale_stats['alpha_std']:.6f}\n"""
            ep_string += f"""{'UPESI alpha_min:':>{pad}} {upesi_scale_stats['alpha_min']:.6f}\n"""
            ep_string += f"""{'UPESI alpha_max:':>{pad}} {upesi_scale_stats['alpha_max']:.6f}\n"""
            ep_string += f"""{'UPESI alpha_norm:':>{pad}} {upesi_scale_stats['alpha_norm']:.6f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        self.writer.add_scalar('UPESI/enabled', 1.0 if self.upesi_enabled else 0.0, locs['it'])
        self.writer.add_scalar('UPESI/encoder_frozen', 1.0 if self.upesi_encoder_frozen else 0.0, locs['it'])

        if cvar_stats is not None:
            self.writer.add_scalar('CVaR/use_cvar', cvar_stats.get('use_cvar', 0.0), locs['it'])
            self.writer.add_scalar('CVaR/alpha', cvar_stats.get('cvar_alpha', 0.0), locs['it'])
            self.writer.add_scalar('CVaR/tail_weight', cvar_stats.get('cvar_tail_weight', 1.0), locs['it'])
            self.writer.add_scalar('CVaR/completed_episodes', cvar_stats.get('completed_episodes', 0.0), locs['it'])
            self.writer.add_scalar('CVaR/tail_episodes', cvar_stats.get('tail_episodes', 0.0), locs['it'])
            self.writer.add_scalar('CVaR/tail_sample_fraction', cvar_stats.get('tail_sample_fraction', 0.0), locs['it'])
            if cvar_stats.get('tail_threshold_return', None) is not None:
                self.writer.add_scalar('CVaR/return_threshold', cvar_stats['tail_threshold_return'], locs['it'])
            if cvar_stats.get('mean_tail_return', None) is not None:
                self.writer.add_scalar('CVaR/mean_tail_return', cvar_stats['mean_tail_return'], locs['it'])

        if upesi_stats is not None and self.upesi_enabled:
            if upesi_stats.get("loss_dyn", None) is not None:
                self.writer.add_scalar('upesi/loss_dyn', upesi_stats["loss_dyn"], locs['it'])
            if upesi_stats.get("loss_rec", None) is not None:
                self.writer.add_scalar('upesi/loss_rec', upesi_stats["loss_rec"], locs['it'])
            if upesi_stats.get("loss_total", None) is not None:
                self.writer.add_scalar('upesi/loss_total', upesi_stats["loss_total"], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        if hasattr(self.env, "domain_rand_current_ranges"):
            ranges = self.env.domain_rand_current_ranges
            friction_range = ranges["friction_range"]
            added_mass_range = ranges["added_mass_range"]
            push_vel_xy_range = ranges["push_vel_xy_range"]
            log_string += (
                f"""{'CDR ranges:':>{pad}} """
                f"""friction[{friction_range[0]:.2f}, {friction_range[1]:.2f}] """
                f"""added_mass[{added_mass_range[0]:.2f}, {added_mass_range[1]:.2f}] """
                f"""push_vel_xy[{push_vel_xy_range[0]:.2f}, {push_vel_xy_range[1]:.2f}]\n"""
            )
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        save_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.upesi_enabled:
            save_dict['upesi'] = {
                'encoder_state_dict': self.upesi_encoder.state_dict(),
                'forward_model_state_dict': self.upesi_forward_model.state_dict(),
                'decoder_state_dict': self.upesi_decoder.state_dict(),
                'optimizer_state_dict': self.upesi_optimizer.state_dict(),
                'identification_optimizer_state_dict': (
                    self.upesi_identification_optimizer.state_dict()
                    if self.upesi_identification_optimizer is not None
                    else None
                ),
            }
        torch.save(save_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        if self.upesi_enabled:
            upesi_dict = loaded_dict.get('upesi', None)
            if upesi_dict is not None:
                self.upesi_encoder.load_state_dict(upesi_dict['encoder_state_dict'])
                self.upesi_forward_model.load_state_dict(upesi_dict['forward_model_state_dict'])
                self.upesi_decoder.load_state_dict(upesi_dict['decoder_state_dict'])
                if load_optimizer and 'optimizer_state_dict' in upesi_dict:
                    self.upesi_optimizer.load_state_dict(upesi_dict['optimizer_state_dict'])
                if (
                    load_optimizer
                    and self.upesi_identification_optimizer is not None
                    and upesi_dict.get('identification_optimizer_state_dict', None) is not None
                ):
                    self.upesi_identification_optimizer.load_state_dict(
                        upesi_dict['identification_optimizer_state_dict']
                    )
            else:
                print("[UPESI] Checkpoint has no UPESI state. Training will continue with fresh UPESI modules.")

        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
            if self.upesi_enabled:
                self.upesi_encoder.to(device)

        if not self.upesi_enabled:
            return self.alg.actor_critic.act_inference
        return self.get_oracle_inference_policy(device=device)
