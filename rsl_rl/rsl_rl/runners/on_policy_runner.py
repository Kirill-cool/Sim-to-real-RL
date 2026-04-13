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
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

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
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            completed_episode_returns = []
            completed_episode_ids = []
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    transition_episode_ids = self.current_episode_ids.clone()
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                        transition_episode_ids=transition_episode_ids,
                    )

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
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
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
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
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
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

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
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
