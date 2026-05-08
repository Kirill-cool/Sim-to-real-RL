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
import json


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


def _parse_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


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


def _init_eval_rollout_accumulators(env):
    return {
        "ep_infos": [],
        "cur_reward_sum": torch.zeros(env.num_envs, dtype=torch.float, device=env.device),
        "cur_episode_length": torch.zeros(env.num_envs, dtype=torch.float, device=env.device),
        "finished_returns": [],
        "finished_lengths": [],
        "done_total": 0,
        "timeout_total": 0,
        "contact_total": 0,
        "orientation_total": 0,
        "non_timeout_total": 0,
        "contact_only_non_timeout_total": 0,
        "orientation_only_non_timeout_total": 0,
        "contact_and_orientation_non_timeout_total": 0,
        "other_non_timeout_total": 0,
    }


def _accumulate_eval_step(env, accum, rews, dones, infos):
    rews = rews.to(env.device)
    dones = dones.to(env.device)
    accum["cur_reward_sum"] += rews
    accum["cur_episode_length"] += 1

    if isinstance(infos, dict) and "episode" in infos:
        accum["ep_infos"].append(infos["episode"])

    done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
    if done_ids.numel() == 0:
        return

    accum["finished_returns"].extend(accum["cur_reward_sum"][done_ids].detach().cpu().numpy().tolist())
    accum["finished_lengths"].extend(accum["cur_episode_length"][done_ids].detach().cpu().numpy().tolist())

    accum["done_total"] += int(done_ids.numel())
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

    accum["timeout_total"] += int(timeout_flags.sum().item())
    accum["contact_total"] += int(contact_flags.sum().item())
    accum["orientation_total"] += int(orientation_flags.sum().item())
    accum["non_timeout_total"] += int(non_timeout_flags.sum().item())
    accum["contact_only_non_timeout_total"] += int(
        (contact_flags & torch.logical_not(orientation_flags) & non_timeout_flags).sum().item()
    )
    accum["orientation_only_non_timeout_total"] += int(
        (orientation_flags & torch.logical_not(contact_flags) & non_timeout_flags).sum().item()
    )
    accum["contact_and_orientation_non_timeout_total"] += int(
        (contact_flags & orientation_flags & non_timeout_flags).sum().item()
    )
    accum["other_non_timeout_total"] += int(
        (non_timeout_flags & torch.logical_not(contact_flags) & torch.logical_not(orientation_flags)).sum().item()
    )

    accum["cur_reward_sum"][done_ids] = 0
    accum["cur_episode_length"][done_ids] = 0


def _finalize_eval_rollout_metrics(accum):
    episode_metrics = _aggregate_episode_metrics(accum["ep_infos"])
    done_total = int(accum["done_total"])
    if done_total > 0:
        fall_rate = accum["non_timeout_total"] / done_total
        timeout_rate = accum["timeout_total"] / done_total
        contact_termination_rate = accum["contact_total"] / done_total
        orientation_termination_rate = accum["orientation_total"] / done_total
        non_timeout_termination_rate = accum["non_timeout_total"] / done_total
        contact_only_non_timeout_rate = accum["contact_only_non_timeout_total"] / done_total
        orientation_only_non_timeout_rate = accum["orientation_only_non_timeout_total"] / done_total
        contact_and_orientation_non_timeout_rate = accum["contact_and_orientation_non_timeout_total"] / done_total
        other_non_timeout_rate = accum["other_non_timeout_total"] / done_total
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
        "return": float(np.mean(accum["finished_returns"])) if len(accum["finished_returns"]) > 0 else None,
        "episode_length": float(np.mean(accum["finished_lengths"])) if len(accum["finished_lengths"]) > 0 else None,
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


def _run_eval_rollout(env, policy, obs, num_steps):
    accum = _init_eval_rollout_accumulators(env)

    for _ in range(num_steps):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        _accumulate_eval_step(env, accum, rews, dones, infos)

    final_metrics = _finalize_eval_rollout_metrics(accum)

    return {
        "obs": obs,
        "return": final_metrics["return"],
        "episode_length": final_metrics["episode_length"],
        "episodes_finished": final_metrics["episodes_finished"],
        "fall_rate": final_metrics["fall_rate"],
        "tracking_error": final_metrics["tracking_error"],
        "timeout_rate": final_metrics["timeout_rate"],
        "contact_termination_rate": final_metrics["contact_termination_rate"],
        "orientation_termination_rate": final_metrics["orientation_termination_rate"],
        "non_timeout_termination_rate": final_metrics["non_timeout_termination_rate"],
        "contact_only_non_timeout_rate": final_metrics["contact_only_non_timeout_rate"],
        "orientation_only_non_timeout_rate": final_metrics["orientation_only_non_timeout_rate"],
        "contact_and_orientation_non_timeout_rate": final_metrics["contact_and_orientation_non_timeout_rate"],
        "other_non_timeout_rate": final_metrics["other_non_timeout_rate"],
    }


def _fmt_metric(value, precision=6):
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _load_online_alpha_from_file(path, expected_dim, device):
    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, dict):
        if "alpha" in loaded:
            alpha = loaded["alpha"]
        elif "alpha_current" in loaded:
            alpha = loaded["alpha_current"]
        else:
            raise ValueError(
                f"online_alpha_file='{path}' does not contain 'alpha' or 'alpha_current'."
            )
    else:
        alpha = loaded
    alpha_t = torch.as_tensor(alpha, dtype=torch.float, device=device)
    if alpha_t.ndim == 2 and alpha_t.shape[-1] == expected_dim:
        alpha_t = alpha_t.mean(dim=0)
    elif alpha_t.ndim == 1 and alpha_t.shape[0] == expected_dim:
        pass
    else:
        raise ValueError(
            f"Loaded alpha has shape {tuple(alpha_t.shape)}, expected [{expected_dim}] or [N, {expected_dim}]."
        )
    return alpha_t.detach().clone()


def _save_online_alpha_to_file(path, alpha, metadata=None):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    payload = {
        "alpha": alpha.detach().cpu(),
    }
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)


def _resolve_combined_run_name(train_cfg, args):
    candidates = [
        getattr(args, "load_run", None),
        getattr(getattr(train_cfg, "runner", None), "load_run", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_str = str(candidate).strip()
        if candidate_str in {"", "-1", "policies", "exported"}:
            continue
        return os.path.basename(candidate_str.rstrip("/"))

    combined_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "rough_go2_combined")
    try:
        run_dirs = [
            d for d in os.listdir(combined_root)
            if os.path.isdir(os.path.join(combined_root, d))
            and d != "exported"
            and not d.endswith("_adapted")
        ]
        if len(run_dirs) == 0:
            return "unknown_run"
        run_dirs.sort()
        return run_dirs[-1]
    except Exception:
        return "unknown_run"


def _make_unique_dir(parent_dir, base_name):
    os.makedirs(parent_dir, exist_ok=True)
    candidate = os.path.join(parent_dir, base_name)
    if not os.path.exists(candidate):
        os.makedirs(candidate, exist_ok=True)
        return candidate
    idx = 2
    while True:
        candidate = os.path.join(parent_dir, f"{base_name}_{idx:02d}")
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        idx += 1


def _create_adapted_export_package(train_cfg, ppo_runner, args, alpha, summary_metrics):
    run_name = _resolve_combined_run_name(train_cfg, args)
    run_name_clean = run_name.rstrip("_")
    combined_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "rough_go2_combined")
    adapted_group_dir = os.path.join(combined_root, f"{run_name_clean}_adapted")
    package_dir = _make_unique_dir(adapted_group_dir, f"{run_name_clean}_adapted_model")

    alpha_path = os.path.join(package_dir, "online_alpha.pt")
    _save_online_alpha_to_file(
        alpha_path,
        alpha=alpha,
        metadata=summary_metrics,
    )

    policy_jit_dir = os.path.join(package_dir, "policy_jit")
    export_policy_as_jit(ppo_runner.alg.actor_critic, policy_jit_dir)

    metadata_path = os.path.join(package_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=True)

    return {
        "run_name": run_name,
        "adapted_group_dir": adapted_group_dir,
        "package_dir": package_dir,
        "alpha_path": alpha_path,
        "policy_jit_dir": policy_jit_dir,
        "metadata_path": metadata_path,
    }


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
    env_cfg.commands.ranges.lin_vel_x = [0.7, 0.7]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env_cfg.commands.heading_command = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # Keep a reference for export helpers without changing external APIs.
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
    if upesi_eval_mode == "identified":
        upesi_cfg_for_identified = getattr(ppo_runner, "upesi_cfg", {})
        identified_episode_length_s = upesi_cfg_for_identified.get("identified_eval_episode_length_s", None)
        identified_episode_length_s = _to_float(identified_episode_length_s)
        if identified_episode_length_s is not None and identified_episode_length_s > 0.0:
            eval_steps = max(1, int(np.ceil(identified_episode_length_s / float(env.dt))) + 1)
            print(
                f"[UPESI] identified mode episode length override: "
                f"{identified_episode_length_s:.3f}s -> eval_steps={eval_steps}"
            )

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
        online_eval_rollout_multiplier = float(upesi_cfg.get("online_eval_rollout_multiplier", 1.0))
        if online_eval_rollout_multiplier <= 0.0:
            online_eval_rollout_multiplier = 1.0
        online_eval_steps = max(1, int(np.ceil(float(eval_steps) * online_eval_rollout_multiplier)))
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
        if online_alpha_init not in {"zero", "nominal", "file"}:
            raise ValueError("upesi.online_alpha_init must be 'zero', 'nominal' or 'file'")
        online_alpha_file = str(upesi_cfg.get("online_alpha_file", "")).strip()
        online_enable_updates = bool(upesi_cfg.get("online_enable_updates", True))
        cli_online_alpha_file = getattr(args, "upesi_online_alpha_file", None)
        if cli_online_alpha_file is not None and str(cli_online_alpha_file).strip() != "":
            online_alpha_file = str(cli_online_alpha_file).strip()
        cli_resume_alpha = _parse_optional_bool(getattr(args, "upesi_online_resume_alpha", None))
        if cli_resume_alpha is True:
            online_alpha_init = "file"
            online_enable_updates = True
        elif cli_resume_alpha is False:
            online_alpha_init = "nominal"
            online_enable_updates = True
        online_alpha_smoothing_beta = float(upesi_cfg.get("online_alpha_smoothing_beta", 0.2))
        online_alpha_smoothing_beta = float(np.clip(online_alpha_smoothing_beta, 0.0, 1.0))
        online_max_alpha_norm = float(upesi_cfg.get("online_max_alpha_norm", 10.0))
        online_identify_accept_ratio = float(upesi_cfg.get("online_identify_accept_ratio", 0.998))
        if not np.isfinite(online_identify_accept_ratio) or online_identify_accept_ratio <= 0.0:
            online_identify_accept_ratio = 0.998
        online_save_final_alpha = bool(upesi_cfg.get("online_save_final_alpha", False))
        identification_steps = getattr(args, "upesi_identification_steps", None)
        identification_lr = getattr(args, "upesi_identification_lr", None)

        ppo_runner.alg.actor_critic.eval()
        if online_alpha_init == "file":
            if online_alpha_file == "":
                raise ValueError("upesi.online_alpha_file must be set when online_alpha_init='file'")
            alpha_current = _load_online_alpha_from_file(
                online_alpha_file,
                expected_dim=ppo_runner.upesi_embedding_dim,
                device=env.device,
            )
            print(f"[UPESI] Loaded online alpha from file: {online_alpha_file}")
        elif online_alpha_init == "nominal":
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
        online_accum = _init_eval_rollout_accumulators(env)
        online_final_alpha_for_policy = None
        online_final_alpha_oracle_dist = None
        online_last_update_alpha_oracle_dist = None
        online_last_update_step = None

        for step_idx in range(1, online_eval_steps + 1):
            # Capture oracle distance for the exact alpha/state used by policy at this step,
            # before env.step() may trigger auto-resets and theta changes.
            if step_idx == online_eval_steps:
                online_final_alpha_for_policy = alpha_current.detach().clone()
                try:
                    with torch.inference_mode():
                        alpha_oracle = ppo_runner.get_oracle_alpha(device=env.device)
                        if alpha_oracle.ndim == 2 and alpha_oracle.shape[0] > 0:
                            dist_vec = torch.norm(
                                alpha_oracle - online_final_alpha_for_policy.view(1, -1),
                                dim=-1,
                                p=2,
                            )
                            online_final_alpha_oracle_dist = float(dist_vec.mean().item())
                        elif alpha_oracle.ndim == 1:
                            online_final_alpha_oracle_dist = float(
                                torch.norm(alpha_oracle - online_final_alpha_for_policy, p=2).item()
                            )
                        else:
                            online_final_alpha_oracle_dist = None
                except Exception:
                    online_final_alpha_oracle_dist = None

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

            _accumulate_eval_step(env, online_accum, rews, dones, infos)
            obs = next_obs

            if not online_enable_updates:
                continue
            if step_idx % online_update_interval != 0:
                continue

            online_base_height_z = None
            if hasattr(env, "root_states") and env.root_states is not None:
                online_base_height_z = float(env.root_states[:, 2].mean().item())

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
                        "online_base_height_z": online_base_height_z,
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
            online_last_update_alpha_oracle_dist = online_alpha_oracle_dist
            online_last_update_step = int(step_idx)

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
                    "online_base_height_z": online_base_height_z,
                },
            )

        online_final_metrics = _finalize_eval_rollout_metrics(online_accum)
        if online_final_alpha_for_policy is None:
            online_final_alpha_for_policy = alpha_current.detach().clone()

        _print_upesi_eval_block(
            "online_identified_summary",
            {
                "online_return": online_final_metrics["return"],
                "online_episode_length": online_final_metrics["episode_length"],
                "online_episodes_finished": online_final_metrics["episodes_finished"],
                "online_fall_rate": online_final_metrics["fall_rate"],
                "online_timeout_rate": online_final_metrics["timeout_rate"],
                "online_contact_termination_rate": online_final_metrics["contact_termination_rate"],
                "online_orientation_termination_rate": online_final_metrics["orientation_termination_rate"],
                "online_non_timeout_termination_rate": online_final_metrics["non_timeout_termination_rate"],
                "online_tracking_error": online_final_metrics["tracking_error"],
                "online_final_alpha_norm": float(online_final_alpha_for_policy.norm().item()),
                "online_final_alpha_oracle_dist": online_final_alpha_oracle_dist,
                "online_last_update_alpha_oracle_dist": online_last_update_alpha_oracle_dist,
                "online_last_update_step": float(online_last_update_step) if online_last_update_step is not None else None,
                "online_eval_steps": float(online_eval_steps),
                "online_enable_updates": float(1.0 if online_enable_updates else 0.0),
            },
        )
        if online_save_final_alpha:
            summary_metrics = {
                "mode": "online_identified",
                "task": str(args.task),
                "run_name": str(_resolve_combined_run_name(train_cfg, args)),
                "online_eval_steps": int(online_eval_steps),
                "online_enable_updates": bool(online_enable_updates),
                "online_return": online_final_metrics["return"],
                "online_episode_length": online_final_metrics["episode_length"],
                "online_episodes_finished": online_final_metrics["episodes_finished"],
                "online_fall_rate": online_final_metrics["fall_rate"],
                "online_timeout_rate": online_final_metrics["timeout_rate"],
                "online_contact_termination_rate": online_final_metrics["contact_termination_rate"],
                "online_orientation_termination_rate": online_final_metrics["orientation_termination_rate"],
                "online_non_timeout_termination_rate": online_final_metrics["non_timeout_termination_rate"],
                "online_tracking_error": online_final_metrics["tracking_error"],
                "online_final_alpha_norm": float(online_final_alpha_for_policy.norm().item()),
                "online_final_alpha_oracle_dist": online_final_alpha_oracle_dist,
                "online_last_update_alpha_oracle_dist": online_last_update_alpha_oracle_dist,
                "online_last_update_step": online_last_update_step,
                "source_experiment_name": str(getattr(train_cfg.runner, "experiment_name", "unknown")),
                "source_load_run": str(getattr(train_cfg.runner, "load_run", "unknown")),
                "source_checkpoint": str(getattr(train_cfg.runner, "checkpoint", "unknown")),
            }
            export_info = _create_adapted_export_package(
                train_cfg=train_cfg,
                ppo_runner=ppo_runner,
                args=args,
                alpha=online_final_alpha_for_policy,
                summary_metrics=summary_metrics,
            )
            print(f"[UPESI] Saved adapted package: {export_info['package_dir']}")
            if online_alpha_file != "":
                _save_online_alpha_to_file(
                    online_alpha_file,
                    alpha=online_final_alpha_for_policy,
                    metadata=summary_metrics,
                )
                print(f"[UPESI] Saved online alpha to explicit file: {online_alpha_file}")
    
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
