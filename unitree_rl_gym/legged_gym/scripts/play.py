import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def _to_float(value):
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


def _aggregate_episode_metrics(ep_infos):
    if len(ep_infos) == 0:
        return {
            "fall_rate": None,
            "tracking_error": None,
            "timeout_rate": None,
            "contact_termination_rate": None,
            "orientation_termination_rate": None,
            "non_timeout_termination_rate": None,
        }

    weighted_sum = {}
    weight_sum = {}
    for ep_info in ep_infos:
        num_episodes = _to_float(ep_info.get("num_episodes"))
        if num_episodes is None or num_episodes <= 0.0:
            num_episodes = 1.0
        for key, value in ep_info.items():
            val = _to_float(value)
            if val is None:
                continue
            weighted_sum[key] = weighted_sum.get(key, 0.0) + val * num_episodes
            weight_sum[key] = weight_sum.get(key, 0.0) + num_episodes

    aggregated = {}
    for key, accum in weighted_sum.items():
        if weight_sum[key] > 0.0:
            aggregated[key] = accum / weight_sum[key]

    fall_rate = aggregated.get("fall_rate", None)
    if fall_rate is None and aggregated.get("success_rate", None) is not None:
        fall_rate = 1.0 - aggregated["success_rate"]
    tracking_error = aggregated.get("tracking_error", None)
    timeout_rate = aggregated.get("timeout_rate", None)
    contact_termination_rate = aggregated.get("contact_termination_rate", None)
    orientation_termination_rate = aggregated.get("orientation_termination_rate", None)
    non_timeout_termination_rate = aggregated.get("non_timeout_termination_rate", None)
    return {
        "fall_rate": fall_rate,
        "tracking_error": tracking_error,
        "timeout_rate": timeout_rate,
        "contact_termination_rate": contact_termination_rate,
        "orientation_termination_rate": orientation_termination_rate,
        "non_timeout_termination_rate": non_timeout_termination_rate,
    }


def _run_eval_rollout(env, policy, obs, num_steps):
    ep_infos = []
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    finished_returns = []
    finished_lengths = []
    done_total = 0
    timeout_total = 0
    contact_total = 0
    orientation_total = 0
    non_timeout_total = 0
    contact_only_non_timeout_total = 0
    orientation_only_non_timeout_total = 0
    contact_and_orientation_non_timeout_total = 0
    other_non_timeout_total = 0

    for _ in range(num_steps):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        rews = rews.to(env.device)
        dones = dones.to(env.device)
        cur_reward_sum += rews
        cur_episode_length += 1
        if isinstance(infos, dict) and "episode" in infos:
            ep_infos.append(infos["episode"])
        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            finished_returns.extend(cur_reward_sum[done_ids].detach().cpu().numpy().tolist())
            finished_lengths.extend(cur_episode_length[done_ids].detach().cpu().numpy().tolist())

            done_total += int(done_ids.numel())
            timeout_flags = None
            if hasattr(env, "termination_timeout_buf"):
                timeout_flags = env.termination_timeout_buf[done_ids].bool()
            elif isinstance(infos, dict) and "time_outs" in infos:
                timeout_flags = infos["time_outs"][done_ids].to(env.device).bool()
            else:
                timeout_flags = torch.zeros(done_ids.numel(), dtype=torch.bool, device=env.device)
            contact_flags = (
                env.termination_contact_buf[done_ids].bool()
                if hasattr(env, "termination_contact_buf")
                else torch.zeros(done_ids.numel(), dtype=torch.bool, device=env.device)
            )
            orientation_flags = (
                env.termination_orientation_buf[done_ids].bool()
                if hasattr(env, "termination_orientation_buf")
                else torch.zeros(done_ids.numel(), dtype=torch.bool, device=env.device)
            )

            non_timeout_flags = torch.logical_not(timeout_flags)
            timeout_total += int(timeout_flags.sum().item())
            contact_total += int(contact_flags.sum().item())
            orientation_total += int(orientation_flags.sum().item())
            non_timeout_total += int(non_timeout_flags.sum().item())
            contact_only_non_timeout_total += int(
                (contact_flags & torch.logical_not(orientation_flags) & non_timeout_flags).sum().item()
            )
            orientation_only_non_timeout_total += int(
                (orientation_flags & torch.logical_not(contact_flags) & non_timeout_flags).sum().item()
            )
            contact_and_orientation_non_timeout_total += int(
                (contact_flags & orientation_flags & non_timeout_flags).sum().item()
            )
            other_non_timeout_total += int(
                (non_timeout_flags & torch.logical_not(contact_flags) & torch.logical_not(orientation_flags)).sum().item()
            )

            cur_reward_sum[done_ids] = 0
            cur_episode_length[done_ids] = 0

    episode_metrics = _aggregate_episode_metrics(ep_infos)
    if done_total > 0:
        fall_rate = non_timeout_total / done_total
        timeout_rate = timeout_total / done_total
        contact_termination_rate = contact_total / done_total
        orientation_termination_rate = orientation_total / done_total
        non_timeout_termination_rate = non_timeout_total / done_total
        contact_only_non_timeout_rate = contact_only_non_timeout_total / done_total
        orientation_only_non_timeout_rate = orientation_only_non_timeout_total / done_total
        contact_and_orientation_non_timeout_rate = contact_and_orientation_non_timeout_total / done_total
        other_non_timeout_rate = other_non_timeout_total / done_total
    else:
        fall_rate = episode_metrics["fall_rate"]
        timeout_rate = episode_metrics["timeout_rate"]
        contact_termination_rate = episode_metrics["contact_termination_rate"]
        orientation_termination_rate = episode_metrics["orientation_termination_rate"]
        non_timeout_termination_rate = episode_metrics["non_timeout_termination_rate"]
        contact_only_non_timeout_rate = None
        orientation_only_non_timeout_rate = None
        contact_and_orientation_non_timeout_rate = None
        other_non_timeout_rate = None

    return {
        "obs": obs,
        "return": float(np.mean(finished_returns)) if len(finished_returns) > 0 else None,
        "episode_length": float(np.mean(finished_lengths)) if len(finished_lengths) > 0 else None,
        "episodes_finished": float(done_total),
        "fall_rate": fall_rate,
        "tracking_error": episode_metrics["tracking_error"],
        "timeout_rate": timeout_rate,
        "contact_termination_rate": contact_termination_rate,
        "orientation_termination_rate": orientation_termination_rate,
        "non_timeout_termination_rate": non_timeout_termination_rate,
        "contact_only_non_timeout_rate": contact_only_non_timeout_rate,
        "orientation_only_non_timeout_rate": orientation_only_non_timeout_rate,
        "contact_and_orientation_non_timeout_rate": contact_and_orientation_non_timeout_rate,
        "other_non_timeout_rate": other_non_timeout_rate,
    }


def _fmt_metric(value, precision=6):
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _print_upesi_eval_block(title, metrics_dict):
    print("[UPESI Eval]")
    print(f"  mode: {title}")
    for key, value in metrics_dict.items():
        if value is None:
            print(f"  {key}: n/a")
        elif isinstance(value, float):
            print(f"  {key}: {_fmt_metric(value)}")
        else:
            print(f"  {key}: {value}")


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Keep explicit play overrides, but set them equal to task/train config values.
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = env_cfg.terrain.curriculum
    env_cfg.noise.add_noise = env_cfg.noise.add_noise
    env_cfg.domain_rand.randomize_friction = env_cfg.domain_rand.randomize_friction
    env_cfg.domain_rand.push_robots = env_cfg.domain_rand.push_robots
    env_cfg.commands.curriculum = env_cfg.commands.curriculum
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.5]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env_cfg.commands.heading_command = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # OnPolicyRunner.__init__ performs env.reset(); re-fetch observations to avoid stale pre-reset tensors.
    obs = env.get_observations()
    upesi_eval_mode = str(getattr(args, "upesi_eval_mode", "standard")).strip().lower()
    if upesi_eval_mode not in {"standard", "oracle", "identified", "online_identified"}:
        raise ValueError(
            f"Unknown upesi_eval_mode='{upesi_eval_mode}'. "
            "Use one of: standard, oracle, identified, online_identified."
        )

    # Episode timeouts trigger at episode_length > max_episode_length in the env,
    # so we run one extra step to ensure timeout-completed episodes enter metrics.
    eval_steps = int(env.max_episode_length) + 1

    if upesi_eval_mode == "standard":
        policy = ppo_runner.get_inference_policy(device=env.device)
        for _ in range(eval_steps):
            actions = policy(obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())
    elif upesi_eval_mode == "oracle":
        policy = ppo_runner.get_oracle_inference_policy(device=env.device)
        alpha_oracle = ppo_runner.get_oracle_alpha(device=env.device)
        oracle_alpha_norm = float(alpha_oracle.norm(dim=-1).mean().item())
        oracle_eval = _run_eval_rollout(env, policy, obs, eval_steps)
        _print_upesi_eval_block(
            "oracle",
            {
                "oracle_alpha_norm": oracle_alpha_norm,
                "oracle_return": oracle_eval["return"],
                "oracle_episode_length": oracle_eval["episode_length"],
                "oracle_episodes_finished": oracle_eval["episodes_finished"],
                "oracle_fall_rate": oracle_eval["fall_rate"],
                "oracle_timeout_rate": oracle_eval["timeout_rate"],
                "oracle_contact_termination_rate": oracle_eval["contact_termination_rate"],
                "oracle_orientation_termination_rate": oracle_eval["orientation_termination_rate"],
                "oracle_non_timeout_termination_rate": oracle_eval["non_timeout_termination_rate"],
                "oracle_contact_only_non_timeout_rate": oracle_eval["contact_only_non_timeout_rate"],
                "oracle_orientation_only_non_timeout_rate": oracle_eval["orientation_only_non_timeout_rate"],
                "oracle_contact_and_orientation_non_timeout_rate": oracle_eval["contact_and_orientation_non_timeout_rate"],
                "oracle_other_non_timeout_rate": oracle_eval["other_non_timeout_rate"],
                "oracle_tracking_error": oracle_eval["tracking_error"],
            },
        )
    elif upesi_eval_mode == "identified":
        if not bool(getattr(ppo_runner, "upesi_enabled", False)):
            raise ValueError("identified UPESI eval requires upesi.enabled = true")
        warmup_steps = max(1, int(getattr(args, "upesi_identification_warmup_steps", 512)))
        ppo_runner.alg.actor_critic.eval()
        alpha_zeros = torch.zeros(
            (env.num_envs, ppo_runner.upesi_embedding_dim),
            dtype=torch.float,
            device=env.device,
        )
        transition_obs = []
        transition_actions = []
        transition_next_obs = []
        for _ in range(warmup_steps):
            obs_for_policy = torch.cat((obs.detach(), alpha_zeros), dim=-1)
            actions = ppo_runner.alg.actor_critic.act_inference(obs_for_policy)
            next_obs, _, _, _, _ = env.step(actions.detach())
            transition_obs.append(obs.detach().clone())
            transition_actions.append(actions.detach().clone())
            transition_next_obs.append(next_obs.detach().clone())
            obs = next_obs
        transitions = {
            "obs": torch.cat(transition_obs, dim=0),
            "action": torch.cat(transition_actions, dim=0),
            "next_obs": torch.cat(transition_next_obs, dim=0),
        }
        policy, alpha_star, identify_diag = ppo_runner.get_identified_inference_policy(
            transitions=transitions,
            device=env.device,
            identification_steps=getattr(args, "upesi_identification_steps", None),
            identification_lr=getattr(args, "upesi_identification_lr", None),
            return_diagnostics=True,
        )
        alpha_star_norm = float(alpha_star.norm().item())

        with torch.inference_mode():
            alpha_oracle = ppo_runner.get_oracle_alpha(device=env.device)
            alpha_oracle_norm = float(alpha_oracle.norm(dim=-1).mean().item())
            alpha_oracle_mean = alpha_oracle.mean(dim=0)
            alpha_star_oracle_dist = float(torch.norm(alpha_star.view(-1) - alpha_oracle_mean, p=2).item())

        obs, _ = env.reset()
        identified_eval = _run_eval_rollout(env, policy, obs, eval_steps)

        obs, _ = env.reset()
        oracle_policy = ppo_runner.get_oracle_inference_policy(device=env.device)
        oracle_eval = _run_eval_rollout(env, oracle_policy, obs, eval_steps)

        return_diff = None
        if identified_eval["return"] is not None and oracle_eval["return"] is not None:
            return_diff = identified_eval["return"] - oracle_eval["return"]

        _print_upesi_eval_block(
            "identified",
            {
                "alpha_star_norm": alpha_star_norm,
                "identify_loss_before": identify_diag["identify_loss_before"],
                "identify_loss_after": identify_diag["identify_loss_after"],
                "identify_loss_ratio": identify_diag["identify_loss_ratio"],
                "alpha_oracle_norm": alpha_oracle_norm,
                "alpha_star_oracle_dist": alpha_star_oracle_dist,
                "identified_return": identified_eval["return"],
                "identified_episode_length": identified_eval["episode_length"],
                "identified_episodes_finished": identified_eval["episodes_finished"],
                "identified_fall_rate": identified_eval["fall_rate"],
                "identified_timeout_rate": identified_eval["timeout_rate"],
                "identified_contact_termination_rate": identified_eval["contact_termination_rate"],
                "identified_orientation_termination_rate": identified_eval["orientation_termination_rate"],
                "identified_non_timeout_termination_rate": identified_eval["non_timeout_termination_rate"],
                "identified_contact_only_non_timeout_rate": identified_eval["contact_only_non_timeout_rate"],
                "identified_orientation_only_non_timeout_rate": identified_eval["orientation_only_non_timeout_rate"],
                "identified_contact_and_orientation_non_timeout_rate": identified_eval["contact_and_orientation_non_timeout_rate"],
                "identified_other_non_timeout_rate": identified_eval["other_non_timeout_rate"],
                "identified_tracking_error": identified_eval["tracking_error"],
                "oracle_return": oracle_eval["return"],
                "oracle_episode_length": oracle_eval["episode_length"],
                "oracle_episodes_finished": oracle_eval["episodes_finished"],
                "oracle_fall_rate": oracle_eval["fall_rate"],
                "oracle_timeout_rate": oracle_eval["timeout_rate"],
                "oracle_contact_termination_rate": oracle_eval["contact_termination_rate"],
                "oracle_orientation_termination_rate": oracle_eval["orientation_termination_rate"],
                "oracle_non_timeout_termination_rate": oracle_eval["non_timeout_termination_rate"],
                "oracle_contact_only_non_timeout_rate": oracle_eval["contact_only_non_timeout_rate"],
                "oracle_orientation_only_non_timeout_rate": oracle_eval["orientation_only_non_timeout_rate"],
                "oracle_contact_and_orientation_non_timeout_rate": oracle_eval["contact_and_orientation_non_timeout_rate"],
                "oracle_other_non_timeout_rate": oracle_eval["other_non_timeout_rate"],
                "oracle_tracking_error": oracle_eval["tracking_error"],
                "return_diff": return_diff,
            },
        )
    else:
        if not bool(getattr(ppo_runner, "upesi_enabled", False)):
            raise ValueError("online_identified UPESI eval requires upesi.enabled = true")

        upesi_cfg = getattr(ppo_runner, "upesi_cfg", {})
        online_window_size = max(1, int(upesi_cfg.get("online_window_size", 512)))
        online_min_buffer_size_cfg = upesi_cfg.get("online_min_buffer_size", None)
        if online_min_buffer_size_cfg is None:
            online_min_buffer_size = online_window_size
        else:
            online_min_buffer_size = int(online_min_buffer_size_cfg)
            if online_min_buffer_size <= 0:
                online_min_buffer_size = online_window_size
        online_update_interval = max(1, int(upesi_cfg.get("online_update_interval", 64)))
        online_alpha_init = str(upesi_cfg.get("online_alpha_init", "zero")).strip().lower()
        if online_alpha_init not in {"zero", "nominal"}:
            raise ValueError("upesi.online_alpha_init must be 'zero' or 'nominal'")
        online_alpha_smoothing_beta = float(upesi_cfg.get("online_alpha_smoothing_beta", 0.2))
        online_alpha_smoothing_beta = float(np.clip(online_alpha_smoothing_beta, 0.0, 1.0))
        online_max_alpha_norm = float(upesi_cfg.get("online_max_alpha_norm", 10.0))
        online_identify_accept_ratio = 0.999
        identification_steps = getattr(args, "upesi_identification_steps", None)
        identification_lr = getattr(args, "upesi_identification_lr", None)

        ppo_runner.alg.actor_critic.eval()
        if online_alpha_init == "nominal":
            with torch.inference_mode():
                theta_zero_norm = torch.zeros(
                    (1, ppo_runner.upesi_theta_dim),
                    dtype=torch.float,
                    device=env.device,
                )
                alpha_current = ppo_runner.upesi_encoder(theta_zero_norm).view(-1).detach().clone()
        else:
            alpha_current = torch.zeros(
                (ppo_runner.upesi_embedding_dim,),
                dtype=torch.float,
                device=env.device,
            )

        trans_obs_buf = None
        trans_action_buf = None
        trans_next_obs_buf = None
        online_ep_infos = []
        online_cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        online_cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        online_finished_returns = []
        online_finished_lengths = []

        for step_idx in range(1, eval_steps + 1):
            with torch.inference_mode():
                alpha_batch = alpha_current.view(1, -1).expand(obs.shape[0], -1)
                obs_for_policy = torch.cat((obs.detach(), alpha_batch), dim=-1)
                actions = ppo_runner.alg.actor_critic.act_inference(obs_for_policy)
                next_obs, _, rews, dones, infos = env.step(actions.detach())

            dones_mask = dones.to(env.device).view(-1).bool()
            valid_mask = torch.logical_not(dones_mask)
            if bool(torch.any(valid_mask).item()):
                obs_valid = obs.detach()[valid_mask].clone()
                actions_valid = actions.detach()[valid_mask].clone()
                next_obs_valid = next_obs.detach()[valid_mask].clone()
                if trans_obs_buf is None:
                    trans_obs_buf = obs_valid
                    trans_action_buf = actions_valid
                    trans_next_obs_buf = next_obs_valid
                    if trans_obs_buf.shape[0] > online_window_size:
                        trans_obs_buf = trans_obs_buf[-online_window_size:]
                        trans_action_buf = trans_action_buf[-online_window_size:]
                        trans_next_obs_buf = trans_next_obs_buf[-online_window_size:]
                else:
                    trans_obs_buf = torch.cat((trans_obs_buf, obs_valid), dim=0)
                    trans_action_buf = torch.cat((trans_action_buf, actions_valid), dim=0)
                    trans_next_obs_buf = torch.cat((trans_next_obs_buf, next_obs_valid), dim=0)
                    if trans_obs_buf.shape[0] > online_window_size:
                        trans_obs_buf = trans_obs_buf[-online_window_size:]
                        trans_action_buf = trans_action_buf[-online_window_size:]
                        trans_next_obs_buf = trans_next_obs_buf[-online_window_size:]

            rews = rews.to(env.device)
            online_cur_reward_sum += rews
            online_cur_episode_length += 1
            if isinstance(infos, dict) and "episode" in infos:
                online_ep_infos.append(infos["episode"])
            online_done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if online_done_ids.numel() > 0:
                online_finished_returns.extend(
                    online_cur_reward_sum[online_done_ids].detach().cpu().numpy().tolist()
                )
                online_finished_lengths.extend(
                    online_cur_episode_length[online_done_ids].detach().cpu().numpy().tolist()
                )
                online_cur_reward_sum[online_done_ids] = 0
                online_cur_episode_length[online_done_ids] = 0
            obs = next_obs

            if step_idx % online_update_interval != 0:
                continue

            online_buffer_size = 0 if trans_obs_buf is None else int(trans_obs_buf.shape[0])
            if online_buffer_size < online_min_buffer_size:
                _print_upesi_eval_block(
                    "online_identified",
                    {
                        "online_step": float(step_idx),
                        "online_update_status": "skipped",
                        "online_skip_reason": "insufficient_buffer",
                        "online_buffer_size/current_min_size": f"{online_buffer_size}/{online_min_buffer_size}",
                        "online_alpha_norm": float(alpha_current.norm().item()),
                    },
                )
                continue

            transitions = {
                "obs": trans_obs_buf,
                "action": trans_action_buf,
                "next_obs": trans_next_obs_buf,
            }
            alpha_new, identify_diag = ppo_runner.identify_alpha(
                transitions=transitions,
                identification_steps=identification_steps,
                identification_lr=identification_lr,
                return_diagnostics=True,
                init_alpha=alpha_current,
            )

            identify_loss_before = identify_diag["identify_loss_before"]
            identify_loss_after = identify_diag["identify_loss_after"]
            identify_loss_ratio = identify_diag["identify_loss_ratio"]
            alpha_new_norm = float(alpha_new.norm().item())

            accept_update = True
            if identify_loss_ratio >= online_identify_accept_ratio:
                accept_update = False
            if alpha_new_norm > online_max_alpha_norm:
                accept_update = False

            if accept_update:
                alpha_current = (
                    (1.0 - online_alpha_smoothing_beta) * alpha_current
                    + online_alpha_smoothing_beta * alpha_new
                )
                online_update_status = "accepted"
            else:
                online_update_status = "rejected"

            online_alpha_norm = float(alpha_current.norm().item())
            online_alpha_oracle_dist = None
            try:
                with torch.inference_mode():
                    alpha_oracle = ppo_runner.get_oracle_alpha(device=env.device)
                    alpha_oracle_mean = alpha_oracle.mean(dim=0)
                    online_alpha_oracle_dist = float(
                        torch.norm(alpha_current - alpha_oracle_mean, p=2).item()
                    )
            except Exception:
                online_alpha_oracle_dist = None

            _print_upesi_eval_block(
                "online_identified",
                {
                    "online_step": float(step_idx),
                    "online_update_status": online_update_status,
                    "online_alpha_norm": online_alpha_norm,
                    "online_identify_loss_before": identify_loss_before,
                    "online_identify_loss_after": identify_loss_after,
                    "online_identify_loss_ratio": identify_loss_ratio,
                    "online_buffer_size/current_min_size": f"{online_buffer_size}/{online_min_buffer_size}",
                    "online_alpha_oracle_dist": online_alpha_oracle_dist,
                },
            )

        online_episode_metrics = _aggregate_episode_metrics(online_ep_infos)
        final_online_alpha_oracle_dist = None
        try:
            with torch.inference_mode():
                alpha_oracle = ppo_runner.get_oracle_alpha(device=env.device)
                alpha_oracle_mean = alpha_oracle.mean(dim=0)
                final_online_alpha_oracle_dist = float(
                    torch.norm(alpha_current - alpha_oracle_mean, p=2).item()
                )
        except Exception:
            final_online_alpha_oracle_dist = None

        _print_upesi_eval_block(
            "online_identified_summary",
            {
                "online_return": float(np.mean(online_finished_returns)) if len(online_finished_returns) > 0 else None,
                "online_episode_length": float(np.mean(online_finished_lengths)) if len(online_finished_lengths) > 0 else None,
                "online_fall_rate": online_episode_metrics["fall_rate"],
                "online_timeout_rate": online_episode_metrics["timeout_rate"],
                "online_contact_termination_rate": online_episode_metrics["contact_termination_rate"],
                "online_orientation_termination_rate": online_episode_metrics["orientation_termination_rate"],
                "online_non_timeout_termination_rate": online_episode_metrics["non_timeout_termination_rate"],
                "online_tracking_error": online_episode_metrics["tracking_error"],
                "online_final_alpha_norm": float(alpha_current.norm().item()),
                "online_final_alpha_oracle_dist": final_online_alpha_oracle_dist,
            },
        )
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
