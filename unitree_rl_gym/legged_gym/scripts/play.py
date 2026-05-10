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


def _log_eval_early_termination(env, dones, step_idx, total_steps, mode_label):
    if step_idx >= total_steps:
        return
    done_ids = (dones.to(env.device) > 0).nonzero(as_tuple=False).flatten()
    if done_ids.numel() == 0:
        return

    done_count = int(done_ids.numel())
    timeout_count = 0
    contact_count = 0
    orientation_count = 0
    if hasattr(env, "termination_timeout_buf"):
        timeout_count = int(env.termination_timeout_buf[done_ids].bool().sum().item())
    if hasattr(env, "termination_contact_buf"):
        contact_count = int(env.termination_contact_buf[done_ids].bool().sum().item())
    if hasattr(env, "termination_orientation_buf"):
        orientation_count = int(env.termination_orientation_buf[done_ids].bool().sum().item())

    ids_preview = done_ids[:8].detach().cpu().numpy().tolist()
    print(
        f"[Play Early Termination] mode={mode_label} step={int(step_idx)}/{int(total_steps)} "
        f"done={done_count} timeout={timeout_count} contact={contact_count} orientation={orientation_count} "
        f"env_ids={ids_preview}"
    )


def _run_eval_rollout(env, policy, obs, num_steps, startup_cfg=None, global_step_offset=0, mode_label="eval"):
    accum = _init_eval_rollout_accumulators(env)

    for step_idx in range(1, num_steps + 1):
        actions = policy(obs.detach())
        actions = _apply_eval_startup_phase(env, actions, global_step_offset + step_idx, startup_cfg=startup_cfg)
        obs, _, rews, dones, infos = env.step(actions.detach())
        _log_eval_early_termination(env, dones, step_idx, num_steps, mode_label=mode_label)
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


def _get_range_center(range_like, fallback):
    if isinstance(range_like, (list, tuple, np.ndarray)) and len(range_like) >= 2:
        return 0.5 * (float(range_like[0]) + float(range_like[1]))
    return float(fallback)


def _prepare_eval_startup_cfg(args, upesi_cfg, env):
    stand_default = int(upesi_cfg.get("eval_startup_stand_steps", 0))
    ramp_default = int(upesi_cfg.get("eval_startup_ramp_steps", 0))
    hold_default = bool(upesi_cfg.get("eval_startup_hold_command", True))

    cli_stand = getattr(args, "play_startup_stand_steps", None)
    cli_ramp = getattr(args, "play_startup_ramp_steps", None)
    cli_hold = _parse_optional_bool(getattr(args, "play_startup_hold_command", None))

    stand_steps = int(stand_default if cli_stand is None else cli_stand)
    ramp_steps = int(ramp_default if cli_ramp is None else cli_ramp)
    hold_command = hold_default if cli_hold is None else bool(cli_hold)

    stand_steps = max(0, stand_steps)
    ramp_steps = max(0, ramp_steps)
    enabled = (stand_steps + ramp_steps) > 0

    target_command = None
    if hasattr(env, "commands") and env.commands is not None and env.commands.ndim == 2 and env.commands.shape[1] >= 3:
        target_command = (
            _get_range_center(getattr(env.cfg.commands.ranges, "lin_vel_x", [0.0, 0.0]), 0.0),
            _get_range_center(getattr(env.cfg.commands.ranges, "lin_vel_y", [0.0, 0.0]), 0.0),
            _get_range_center(getattr(env.cfg.commands.ranges, "ang_vel_yaw", [0.0, 0.0]), 0.0),
        )

    return {
        "enabled": bool(enabled),
        "stand_steps": int(stand_steps),
        "ramp_steps": int(ramp_steps),
        "hold_command": bool(hold_command),
        "target_command": target_command,
    }


def _apply_eval_startup_phase(env, actions, step_idx, startup_cfg):
    if startup_cfg is None or not bool(startup_cfg.get("enabled", False)):
        return actions

    stand_steps = int(startup_cfg.get("stand_steps", 0))
    ramp_steps = int(startup_cfg.get("ramp_steps", 0))
    total_steps = stand_steps + ramp_steps

    if hasattr(env, "commands") and env.commands is not None and env.commands.ndim == 2 and env.commands.shape[1] >= 3:
        if bool(startup_cfg.get("hold_command", False)) and step_idx <= stand_steps:
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
        elif step_idx <= total_steps:
            target_command = startup_cfg.get("target_command", None)
            if target_command is not None:
                env.commands[:, 0] = float(target_command[0])
                env.commands[:, 1] = float(target_command[1])
                env.commands[:, 2] = float(target_command[2])

    if step_idx <= stand_steps:
        action_scale = 0.0
    elif ramp_steps > 0 and step_idx <= total_steps:
        action_scale = float(step_idx - stand_steps) / float(ramp_steps)
    else:
        action_scale = 1.0

    return actions * float(np.clip(action_scale, 0.0, 1.0))


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


def _to_range_pair(value, fallback):
    if value is None:
        value = fallback
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) >= 2:
            low, high = float(value[0]), float(value[1])
        elif len(value) == 1:
            low = high = float(value[0])
        else:
            low, high = float(fallback[0]), float(fallback[1])
    else:
        low = high = float(value)
    if low > high:
        low, high = high, low
    return [low, high]


def _biased_subrange(base_range, bias="full", hardness=0.35):
    low, high = _to_range_pair(base_range, [0.0, 0.0])
    span = max(0.0, high - low)
    if span <= 0.0 or bias == "full":
        return [float(low), float(high)]
    hardness = float(np.clip(hardness, 0.0, 1.0))
    if bias == "low":
        return [float(low), float(low + hardness * span)]
    if bias == "high":
        return [float(high - hardness * span), float(high)]
    return [float(low), float(high)]


def _configure_upesi_eval_hard_domain(env):
    if not hasattr(env, "domain_rand_current_ranges"):
        return {}
    current = getattr(env, "domain_rand_current_ranges", None)
    if not isinstance(current, dict):
        return {}
    max_ranges = getattr(env, "domain_rand_max_ranges", {})
    if not isinstance(max_ranges, dict):
        max_ranges = {}

    def _get_range(key, fallback):
        if key in max_ranges:
            return _to_range_pair(max_ranges[key], fallback)
        if key in current:
            return _to_range_pair(current[key], fallback)
        return _to_range_pair(fallback, fallback)

    hard_ranges = {}
    if "added_mass_range" in current or "added_mass_range" in max_ranges:
        hard_ranges["added_mass_range"] = _biased_subrange(
            _get_range("added_mass_range", [0.0, 0.0]),
            bias="full",
            hardness=1.0,
        )
    if "friction_range" in current or "friction_range" in max_ranges:
        hard_ranges["friction_range"] = _biased_subrange(
            _get_range("friction_range", [1.0, 1.0]),
            bias="low",
            hardness=0.5,
        )
    if "motor_strength_range" in current or "motor_strength_range" in max_ranges:
        hard_ranges["motor_strength_range"] = _biased_subrange(
            _get_range("motor_strength_range", [1.0, 1.0]),
            bias="low",
            hardness=0.5,
        )
    if "joint_damping_range" in current or "joint_damping_range" in max_ranges:
        hard_ranges["joint_damping_range"] = _biased_subrange(
            _get_range("joint_damping_range", [1.0, 1.0]),
            bias="high",
            hardness=0.5,
        )
    if "static_joint_friction_range" in current or "static_joint_friction_range" in max_ranges:
        hard_ranges["static_joint_friction_range"] = _biased_subrange(
            _get_range("static_joint_friction_range", [0.0, 0.0]),
            bias="high",
            hardness=0.5,
        )

    # Keep observation noise unchanged by default, unless the task explicitly wants to expand it.
    if "observation_noise_range" in current:
        hard_ranges["observation_noise_range"] = _to_range_pair(
            current["observation_noise_range"],
            [1.0, 1.0],
        )

    for key, value in hard_ranges.items():
        current[key] = list(value)

    cfg_dr = getattr(getattr(env, "cfg", None), "domain_rand", None)
    if cfg_dr is not None:
        for key, value in hard_ranges.items():
            if hasattr(cfg_dr, key):
                setattr(cfg_dr, key, list(value))
    return hard_ranges


def _log_upesi_theta_stats(env, label):
    if not hasattr(env, "get_current_upesi_theta"):
        return
    theta = env.get_current_upesi_theta()
    if not isinstance(theta, torch.Tensor) or theta.ndim != 2 or theta.shape[0] == 0:
        return
    theta_np = theta.detach().cpu().numpy()
    if hasattr(env, "get_upesi_theta_keys"):
        theta_keys = list(env.get_upesi_theta_keys())
    else:
        theta_keys = [f"theta_{idx}" for idx in range(theta_np.shape[1])]
    parts = []
    for idx, key in enumerate(theta_keys):
        if idx >= theta_np.shape[1]:
            break
        col = theta_np[:, idx]
        parts.append(
            f"{key}:mean={float(col.mean()):.4f},std={float(col.std()):.4f},"
            f"min={float(col.min()):.4f},max={float(col.max()):.4f}"
        )
    print(f"[UPESI Eval] {label} theta stats | " + " | ".join(parts))


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Play-only manual heading overrides (edit directly here; no CLI flags required).
    manual_heading_command = False   # None | True | False
    manual_heading_target_rad = 0.0  # None | float (e.g. 0.0)
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

    if manual_heading_command is not None:
        env_cfg.commands.heading_command = bool(manual_heading_command)
    else:
        play_heading_command = _parse_optional_bool(getattr(args, "play_heading_command", None))
        if play_heading_command is not None:
            env_cfg.commands.heading_command = bool(play_heading_command)

    play_heading = manual_heading_target_rad
    if play_heading is None:
        play_heading = getattr(args, "play_heading", None)
    if play_heading is not None:
        env_cfg.commands.heading_command = True
        env_cfg.commands.ranges.heading = [float(play_heading), float(play_heading)]
        print(f"[Play] Fixed heading target: {float(play_heading):.4f} rad")
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
    hard_domain_flag = _parse_optional_bool(getattr(args, "upesi_eval_hard_domain", None))
    hard_domain_enabled = bool(hard_domain_flag) if hard_domain_flag is not None else False
    print(f"UPESI eval hard domain: {hard_domain_enabled}")
    if hard_domain_enabled and upesi_eval_mode in {"oracle", "identified", "online_identified"}:
        hard_ranges = _configure_upesi_eval_hard_domain(env)
        if len(hard_ranges) > 0:
            hard_ranges_pretty = " | ".join(
                f"{k}[{float(v[0]):.4f},{float(v[1]):.4f}]"
                for k, v in hard_ranges.items()
            )
            print(f"[UPESI Eval] hard-domain ranges: {hard_ranges_pretty}")
        else:
            print("[UPESI Eval] hard-domain ranges: n/a (env does not expose domain_rand_current_ranges)")
        obs, _ = env.reset()
        _log_upesi_theta_stats(env, label="post-hard-reset")

    upesi_cfg_for_eval = getattr(ppo_runner, "upesi_cfg", {})
    eval_startup_cfg = _prepare_eval_startup_cfg(args, upesi_cfg_for_eval, env)
    if eval_startup_cfg["enabled"]:
        print(
            "[Play Startup] enabled | "
            f"stand_steps={eval_startup_cfg['stand_steps']} "
            f"ramp_steps={eval_startup_cfg['ramp_steps']} "
            f"hold_command={eval_startup_cfg['hold_command']}"
        )
    else:
        print("[Play Startup] disabled")

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
        for step_idx in range(1, eval_steps + 1):
            actions = policy(obs.detach())
            actions = _apply_eval_startup_phase(env, actions, step_idx, startup_cfg=eval_startup_cfg)
            obs, _, _, dones, _ = env.step(actions.detach())
            _log_eval_early_termination(env, dones, step_idx, eval_steps, mode_label="standard")
    elif upesi_eval_mode == "oracle":
        upesi_cfg = getattr(ppo_runner, "upesi_cfg", {})
        oracle_eval_steps = int(eval_steps)
        online_episode_length_s = _to_float(upesi_cfg.get("online_identified_episode_length_s", None))
        if online_episode_length_s is not None and online_episode_length_s > 0.0:
            env.max_episode_length_s = float(online_episode_length_s)
            env.max_episode_length = float(np.ceil(online_episode_length_s / float(env.dt)))
            oracle_eval_steps = int(env.max_episode_length) + 1
            print(
                f"[UPESI] oracle mode episode length override (synced with online): "
                f"{online_episode_length_s:.3f}s -> max_episode_length={int(env.max_episode_length)} "
                f"eval_steps={oracle_eval_steps}"
            )
        online_eval_rollout_multiplier = float(upesi_cfg.get("online_eval_rollout_multiplier", 1.0))
        if online_eval_rollout_multiplier <= 0.0:
            online_eval_rollout_multiplier = 1.0
        oracle_eval_steps = max(1, int(np.ceil(float(oracle_eval_steps) * online_eval_rollout_multiplier)))
        print(
            f"[UPESI] oracle mode rollout multiplier (synced with online): "
            f"{online_eval_rollout_multiplier:.3f} -> oracle_eval_steps={oracle_eval_steps}"
        )

        policy = ppo_runner.get_oracle_inference_policy(device=env.device)
        alpha_oracle = ppo_runner.get_oracle_alpha(device=env.device)
        oracle_alpha_norm = float(alpha_oracle.norm(dim=-1).mean().item())
        oracle_eval = _run_eval_rollout(
            env, policy, obs, oracle_eval_steps, startup_cfg=eval_startup_cfg, mode_label="oracle"
        )
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
        identified_eval = _run_eval_rollout(
            env, policy, obs, eval_steps, startup_cfg=eval_startup_cfg, mode_label="identified"
        )

        obs, _ = env.reset()
        oracle_policy = ppo_runner.get_oracle_inference_policy(device=env.device)
        oracle_eval = _run_eval_rollout(
            env, oracle_policy, obs, eval_steps, startup_cfg=eval_startup_cfg, mode_label="identified_oracle_ref"
        )

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
        online_episode_length_s = _to_float(upesi_cfg.get("online_identified_episode_length_s", None))
        if online_episode_length_s is not None and online_episode_length_s > 0.0:
            env.max_episode_length_s = float(online_episode_length_s)
            env.max_episode_length = float(np.ceil(online_episode_length_s / float(env.dt)))
            eval_steps = int(env.max_episode_length) + 1
            print(
                f"[UPESI] online_identified episode length override: "
                f"{online_episode_length_s:.3f}s -> max_episode_length={int(env.max_episode_length)} "
                f"eval_steps={eval_steps}"
            )

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
        elif cli_resume_alpha is False:
            online_alpha_init = "nominal"
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
                if hasattr(ppo_runner, "_encode_upesi_alpha"):
                    alpha_nominal = ppo_runner._encode_upesi_alpha(theta_zero_norm)
                else:
                    raw_alpha = ppo_runner.upesi_encoder(theta_zero_norm)
                    alpha_scale = float(getattr(ppo_runner, "upesi_alpha_scale", 1.0))
                    alpha_nominal = alpha_scale * torch.tanh(raw_alpha)
                alpha_current = alpha_nominal.view(-1).detach().clone()
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
                actions = _apply_eval_startup_phase(env, actions, step_idx, startup_cfg=eval_startup_cfg)
                next_obs, _, rews, dones, infos = env.step(actions.detach())
            _log_eval_early_termination(
                env, dones, step_idx, online_eval_steps, mode_label="online_identified"
            )

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
