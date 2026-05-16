import numpy as np
import torch
from collections import deque

from legged_gym.envs.base.legged_robot import LeggedRobot


class Go2Robot(LeggedRobot):
    THETA_KEYS = [
        "added_mass",
        "surface_friction",
        "motor_strength_FL",
        "motor_strength_FR",
        "motor_strength_RL",
        "motor_strength_RR",
        "joint_damping_FL",
        "joint_damping_FR",
        "joint_damping_RL",
        "joint_damping_RR",
        "static_joint_friction_FL",
        "static_joint_friction_FR",
        "static_joint_friction_RL",
        "static_joint_friction_RR",
    ]
    LEG_ORDER = ["FL", "FR", "RL", "RR"]
    RANGE_KEY_TO_COLUMN_SPECS = [
        ("added_mass_range", ["added_mass"]),
        ("friction_range", ["surface_friction"]),
        (
            "motor_strength_range",
            ["motor_strength_FL", "motor_strength_FR", "motor_strength_RL", "motor_strength_RR"],
        ),
        (
            "joint_damping_range",
            ["joint_damping_FL", "joint_damping_FR", "joint_damping_RL", "joint_damping_RR"],
        ),
        (
            "static_joint_friction_range",
            [
                "static_joint_friction_FL",
                "static_joint_friction_FR",
                "static_joint_friction_RL",
                "static_joint_friction_RR",
            ],
        ),
    ]

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._theta_key_to_col = {key: idx for idx, key in enumerate(self.THETA_KEYS)}
        self._domain_rand_sample_counter = 0
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _ensure_env_theta_buffer(self):
        theta_dim = len(self.THETA_KEYS)
        if hasattr(self, "env_theta") and self.env_theta.shape == (self.num_envs, theta_dim):
            return
        self.env_theta = torch.zeros((self.num_envs, theta_dim), dtype=torch.float, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self._build_leg_dof_mapping()
        if not hasattr(self, "motor_strength_scales") or self.motor_strength_scales.shape != (self.num_envs, 4):
            self.motor_strength_scales = torch.ones((self.num_envs, 4), dtype=torch.float, device=self.device)
        if not hasattr(self, "joint_damping_scales") or self.joint_damping_scales.shape != (self.num_envs, 4):
            self.joint_damping_scales = torch.ones((self.num_envs, 4), dtype=torch.float, device=self.device)
        if not hasattr(self, "static_joint_friction") or self.static_joint_friction.shape != (self.num_envs, 4):
            self.static_joint_friction = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        if not hasattr(self, "obs_noise_scales") or self.obs_noise_scales.shape != (self.num_envs,):
            self.obs_noise_scales = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)
        self._sync_env_theta_from_current_params()

    def _build_leg_dof_mapping(self):
        if hasattr(self, "dof_leg_ids") and self.dof_leg_ids.shape[0] == self.num_dof:
            return
        leg_to_idx = {leg: idx for idx, leg in enumerate(self.LEG_ORDER)}
        dof_leg_ids = torch.full((self.num_dof,), -1, dtype=torch.long, device=self.device)
        self.leg_dof_indices = {leg: [] for leg in self.LEG_ORDER}
        for dof_idx, dof_name in enumerate(self.dof_names):
            prefix = dof_name.split("_")[0]
            if prefix in leg_to_idx:
                dof_leg_ids[dof_idx] = int(leg_to_idx[prefix])
                self.leg_dof_indices[prefix].append(dof_idx)
        for leg in self.LEG_ORDER:
            if len(self.leg_dof_indices[leg]) == 0:
                raise ValueError(f"[Go2 DR] No DOFs mapped to leg '{leg}'.")
        self.dof_leg_ids = dof_leg_ids
        self.leg_dof_name_mapping = {
            leg: [self.dof_names[idx] for idx in indices]
            for leg, indices in self.leg_dof_indices.items()
        }
        mapping_msg = " | ".join(
            f"{leg}:{self.leg_dof_indices[leg]}({','.join(self.leg_dof_name_mapping[leg])})"
            for leg in self.LEG_ORDER
        )
        print(f"[Go2 DR] DOF-to-leg mapping: {mapping_msg}")

    def get_dof_leg_mapping_summary(self):
        if not hasattr(self, "leg_dof_name_mapping"):
            return {}
        return dict(self.leg_dof_name_mapping)

    def get_upesi_theta_keys(self):
        return list(self.THETA_KEYS)

    def get_upesi_cdr_max_ranges(self):
        max_ranges = getattr(self, "domain_rand_max_ranges", {})
        cdr_max = {
            "added_mass": list(max_ranges.get("added_mass_range", [0.0, 0.0])),
            "surface_friction": list(max_ranges.get("friction_range", [1.0, 1.0])),
        }
        for leg in self.LEG_ORDER:
            cdr_max[f"motor_strength_{leg}"] = list(max_ranges.get("motor_strength_range", [1.0, 1.0]))
            cdr_max[f"joint_damping_{leg}"] = list(max_ranges.get("joint_damping_range", [1.0, 1.0]))
            cdr_max[f"static_joint_friction_{leg}"] = list(
                max_ranges.get("static_joint_friction_range", [0.0, 0.0])
            )
        return cdr_max

    def _set_theta_column(self, env_ids_torch, key, values):
        col = self._theta_key_to_col[key]
        self.env_theta[env_ids_torch, col] = values

    def _sync_env_theta_from_current_params(self):
        self._ensure_env_theta_buffer()
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if hasattr(self, "added_mass_offsets"):
            self._set_theta_column(
                env_ids,
                "added_mass",
                torch.from_numpy(self.added_mass_offsets).to(device=self.device, dtype=torch.float),
            )
        if hasattr(self, "friction_coeffs"):
            self._set_theta_column(
                env_ids,
                "surface_friction",
                self.friction_coeffs.view(-1).to(device=self.device, dtype=torch.float),
            )
        self._set_theta_column(env_ids, "motor_strength_FL", self.motor_strength_scales[:, 0])
        self._set_theta_column(env_ids, "motor_strength_FR", self.motor_strength_scales[:, 1])
        self._set_theta_column(env_ids, "motor_strength_RL", self.motor_strength_scales[:, 2])
        self._set_theta_column(env_ids, "motor_strength_RR", self.motor_strength_scales[:, 3])
        self._set_theta_column(env_ids, "joint_damping_FL", self.joint_damping_scales[:, 0])
        self._set_theta_column(env_ids, "joint_damping_FR", self.joint_damping_scales[:, 1])
        self._set_theta_column(env_ids, "joint_damping_RL", self.joint_damping_scales[:, 2])
        self._set_theta_column(env_ids, "joint_damping_RR", self.joint_damping_scales[:, 3])
        self._set_theta_column(env_ids, "static_joint_friction_FL", self.static_joint_friction[:, 0])
        self._set_theta_column(env_ids, "static_joint_friction_FR", self.static_joint_friction[:, 1])
        self._set_theta_column(env_ids, "static_joint_friction_RL", self.static_joint_friction[:, 2])
        self._set_theta_column(env_ids, "static_joint_friction_RR", self.static_joint_friction[:, 3])

    def _expand_leg_values_to_dofs(self, leg_values, default_value):
        expanded = torch.full(
            (leg_values.shape[0], self.num_dof),
            float(default_value),
            dtype=leg_values.dtype,
            device=leg_values.device,
        )
        leg_ids = self.dof_leg_ids.unsqueeze(0).expand(leg_values.shape[0], -1)
        valid_mask = leg_ids >= 0
        if torch.any(valid_mask):
            gather_idx = leg_ids.clamp_min(0)
            gathered = torch.gather(leg_values, 1, gather_idx)
            expanded[valid_mask] = gathered[valid_mask]
        return expanded

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        damping_scales = self._expand_leg_values_to_dofs(self.joint_damping_scales, default_value=1.0)
        if control_type == "P":
            pos_term = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
            damping_term = (self.d_gains * damping_scales) * self.dof_vel
            torques = pos_term - damping_term
        elif control_type == "V":
            vel_error = self.dof_vel - self.last_dof_vel
            damping_term = (self.d_gains * damping_scales) * vel_error / self.sim_params.dt
            torques = self.p_gains * (actions_scaled - self.dof_vel) - damping_term
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        static_eps = float(getattr(self.cfg.domain_rand, "static_friction_eps", 0.05))
        static_eps = max(static_eps, 1.0e-6)
        static_coeff = self._expand_leg_values_to_dofs(self.static_joint_friction, default_value=0.0)
        torques = torques - static_coeff * torch.tanh(self.dof_vel / static_eps)

        motor_scales = self._expand_leg_values_to_dofs(self.motor_strength_scales, default_value=1.0)
        torques = torques * motor_scales
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        if self.add_noise:
            noise = (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            noise = noise * self.obs_noise_scales.unsqueeze(-1)
            self.obs_buf += noise

    def _init_domain_rand_curriculum(self):
        cdr_cfg = getattr(self.cfg.domain_rand, "cdr", None)
        self.cdr_enabled = bool(getattr(cdr_cfg, "enabled", False)) if cdr_cfg is not None else False
        self.cdr_max_level = max(0.0, float(getattr(cdr_cfg, "max_level", 1.0))) if cdr_cfg is not None else 1.0
        initial_level = float(getattr(cdr_cfg, "initial_level", self.cdr_max_level)) if cdr_cfg is not None else 1.0
        self.cdr_level_step = max(0.0, float(getattr(cdr_cfg, "level_step", 0.0))) if cdr_cfg is not None else 0.0
        self.cdr_update_interval = max(1, int(getattr(cdr_cfg, "update_interval", 1))) if cdr_cfg is not None else 1
        self.cdr_success_threshold = float(getattr(cdr_cfg, "success_threshold", 1.0)) if cdr_cfg is not None else 1.0
        self.cdr_min_episodes_for_update = (
            max(1, int(getattr(cdr_cfg, "min_episodes_for_update", 1))) if cdr_cfg is not None else 1
        )
        self.cdr_use_stagewise_progression = (
            bool(getattr(cdr_cfg, "use_stagewise_progression", False)) if cdr_cfg is not None else False
        )
        self.cdr_stage_order = [
            "friction",
            "added_mass",
            "motor_strength",
            "joint_damping",
            "static_joint_friction",
            "observation_noise",
            "push_vel_xy",
        ]
        self.cdr_stage_idx = 0
        self.cdr_metrics_window = deque(maxlen=self.cdr_update_interval)
        self.cdr_iteration_counter = 0
        self._init_cdr_upesi_gate_state(cdr_cfg)

        if self.cdr_enabled:
            self.cdr_level = float(np.clip(initial_level, 0.0, self.cdr_max_level))
        else:
            self.cdr_level = float(np.clip(self.cdr_max_level, 0.0, 1.0))

        fallback_friction_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "friction_range", [1.0, 1.0]), [1.0, 1.0]
        )
        fallback_added_mass_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "added_mass_range", [0.0, 0.0]), [0.0, 0.0]
        )
        fallback_push_max = float(getattr(self.cfg.domain_rand, "max_push_vel_xy", 0.0))
        fallback_push_vel_max = [-fallback_push_max, fallback_push_max]
        fallback_motor_strength_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "motor_strength_range", [1.0, 1.0]), [1.0, 1.0]
        )
        fallback_joint_damping_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "joint_damping_range", [1.0, 1.0]), [1.0, 1.0]
        )
        fallback_static_joint_friction_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "static_joint_friction_range", [0.0, 0.0]), [0.0, 0.0]
        )
        fallback_observation_noise_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "observation_noise_range", [1.0, 1.0]), [1.0, 1.0]
        )

        friction_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "friction_max_range", fallback_friction_max), fallback_friction_max
        )
        added_mass_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "added_mass_max_range", fallback_added_mass_max), fallback_added_mass_max
        )
        push_vel_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "push_vel_xy_max_range", fallback_push_vel_max), fallback_push_vel_max
        )
        motor_strength_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "motor_strength_max_range", fallback_motor_strength_max),
            fallback_motor_strength_max,
        )
        joint_damping_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "joint_damping_max_range", fallback_joint_damping_max),
            fallback_joint_damping_max,
        )
        static_joint_friction_max = self._to_range_pair(
            getattr(
                self.cfg.domain_rand,
                "static_joint_friction_max_range",
                fallback_static_joint_friction_max,
            ),
            fallback_static_joint_friction_max,
        )
        observation_noise_max = self._to_range_pair(
            getattr(self.cfg.domain_rand, "observation_noise_max_range", fallback_observation_noise_max),
            fallback_observation_noise_max,
        )

        self.domain_rand_nominal = {
            "friction": float(
                getattr(self.cfg.domain_rand, "friction_nominal", 0.5 * (friction_max[0] + friction_max[1]))
            ),
            "added_mass": float(getattr(self.cfg.domain_rand, "added_mass_nominal", 0.0)),
            "push_vel_xy": float(getattr(self.cfg.domain_rand, "push_vel_xy_nominal", 0.0)),
            "motor_strength": float(getattr(self.cfg.domain_rand, "motor_strength_nominal", 1.0)),
            "joint_damping": float(getattr(self.cfg.domain_rand, "joint_damping_nominal", 1.0)),
            "static_joint_friction": float(getattr(self.cfg.domain_rand, "static_joint_friction_nominal", 0.0)),
            "observation_noise": float(getattr(self.cfg.domain_rand, "observation_noise_nominal", 1.0)),
        }
        self.domain_rand_max_ranges = {
            "friction_range": friction_max,
            "added_mass_range": added_mass_max,
            "push_vel_xy_range": push_vel_max,
            "motor_strength_range": motor_strength_max,
            "joint_damping_range": joint_damping_max,
            "static_joint_friction_range": static_joint_friction_max,
            "observation_noise_range": observation_noise_max,
        }
        self.compute_current_domain_rand_ranges()

    def compute_current_domain_rand_ranges(self):
        interpolation_level = float(np.clip(self.cdr_level, 0.0, 1.0))
        if self.cdr_use_stagewise_progression:
            param_levels = {}
            for idx, param_name in enumerate(self.cdr_stage_order):
                if idx < self.cdr_stage_idx:
                    param_levels[param_name] = 1.0
                elif idx == self.cdr_stage_idx:
                    param_levels[param_name] = interpolation_level
                else:
                    param_levels[param_name] = 0.0
        else:
            param_levels = {key: interpolation_level for key in self.cdr_stage_order}

        self.domain_rand_current_ranges = {
            "friction_range": self._interpolate_range(
                self.domain_rand_nominal["friction"],
                self.domain_rand_max_ranges["friction_range"],
                param_levels["friction"],
            ),
            "added_mass_range": self._interpolate_range(
                self.domain_rand_nominal["added_mass"],
                self.domain_rand_max_ranges["added_mass_range"],
                param_levels["added_mass"],
            ),
            "push_vel_xy_range": self._interpolate_range(
                self.domain_rand_nominal["push_vel_xy"],
                self.domain_rand_max_ranges["push_vel_xy_range"],
                param_levels["push_vel_xy"],
            ),
            "motor_strength_range": self._interpolate_range(
                self.domain_rand_nominal["motor_strength"],
                self.domain_rand_max_ranges["motor_strength_range"],
                param_levels["motor_strength"],
            ),
            "joint_damping_range": self._interpolate_range(
                self.domain_rand_nominal["joint_damping"],
                self.domain_rand_max_ranges["joint_damping_range"],
                param_levels["joint_damping"],
            ),
            "static_joint_friction_range": self._interpolate_range(
                self.domain_rand_nominal["static_joint_friction"],
                self.domain_rand_max_ranges["static_joint_friction_range"],
                param_levels["static_joint_friction"],
            ),
            "observation_noise_range": self._interpolate_range(
                self.domain_rand_nominal["observation_noise"],
                self.domain_rand_max_ranges["observation_noise_range"],
                param_levels["observation_noise"],
            ),
        }

        self.cfg.domain_rand.friction_range = self.domain_rand_current_ranges["friction_range"]
        self.cfg.domain_rand.added_mass_range = self.domain_rand_current_ranges["added_mass_range"]
        push_low, push_high = self.domain_rand_current_ranges["push_vel_xy_range"]
        self.cfg.domain_rand.max_push_vel_xy = max(abs(push_low), abs(push_high))
        self.cfg.domain_rand.motor_strength_range = self.domain_rand_current_ranges["motor_strength_range"]
        self.cfg.domain_rand.joint_damping_range = self.domain_rand_current_ranges["joint_damping_range"]
        self.cfg.domain_rand.static_joint_friction_range = self.domain_rand_current_ranges[
            "static_joint_friction_range"
        ]
        self.cfg.domain_rand.observation_noise_range = self.domain_rand_current_ranges["observation_noise_range"]
        return self.domain_rand_current_ranges

    def _sample_range(self, range_key, sample_shape):
        low, high = self.domain_rand_current_ranges[range_key]
        return np.random.uniform(low, high, size=sample_shape).astype(np.float32)

    def _sample_range_per_leg(self, range_key, per_leg_range_key, num_envs, num_legs=4):
        per_leg_ranges = self.domain_rand_current_ranges.get(per_leg_range_key, None)
        if not isinstance(per_leg_ranges, dict):
            return self._sample_range(range_key, (num_envs, num_legs))

        sampled = np.zeros((num_envs, num_legs), dtype=np.float32)
        for leg_idx, leg_name in enumerate(self.LEG_ORDER[:num_legs]):
            leg_range = per_leg_ranges.get(leg_name, None)
            if leg_range is None:
                low, high = self.domain_rand_current_ranges[range_key]
            else:
                low, high = float(leg_range[0]), float(leg_range[1])
                if low > high:
                    low, high = high, low
            sampled[:, leg_idx] = np.random.uniform(low, high, size=num_envs).astype(np.float32)
        return sampled

    def _assert_samples_in_range(self, values, range_key):
        low, high = self.domain_rand_current_ranges[range_key]
        eps = 1.0e-5
        if np.any(values < (low - eps)) or np.any(values > (high + eps)):
            raise ValueError(f"[Go2 DR] Sampled values for '{range_key}' are out of bounds [{low}, {high}].")

    def _log_theta_stats(self, sampled_theta_np):
        return

    def sample_domain_randomization(self, env_ids=None):
        if env_ids is None:
            env_ids_np = np.arange(self.num_envs, dtype=np.int32)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_np = env_ids.detach().cpu().numpy().astype(np.int32).reshape(-1)
        else:
            env_ids_np = np.asarray(env_ids, dtype=np.int32).reshape(-1)
        if env_ids_np.size == 0:
            return

        self._ensure_env_theta_buffer()
        env_ids_torch = torch.as_tensor(env_ids_np, dtype=torch.long, device=self.device)

        if not hasattr(self, "friction_coeffs") or self.friction_coeffs.shape[0] != self.num_envs:
            self.friction_coeffs = torch.zeros((self.num_envs, 1), dtype=torch.float, device="cpu")
        if not hasattr(self, "motor_strength_scales") or self.motor_strength_scales.shape != (self.num_envs, 4):
            self.motor_strength_scales = torch.ones((self.num_envs, 4), dtype=torch.float, device=self.device)
        if not hasattr(self, "joint_damping_scales") or self.joint_damping_scales.shape != (self.num_envs, 4):
            self.joint_damping_scales = torch.ones((self.num_envs, 4), dtype=torch.float, device=self.device)
        if not hasattr(self, "static_joint_friction") or self.static_joint_friction.shape != (self.num_envs, 4):
            self.static_joint_friction = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        if not hasattr(self, "obs_noise_scales") or self.obs_noise_scales.shape != (self.num_envs,):
            self.obs_noise_scales = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)

        if self.cfg.domain_rand.randomize_friction:
            sampled_friction = self._sample_range("friction_range", env_ids_np.size)
        else:
            sampled_friction = (
                self.friction_coeffs[torch.as_tensor(env_ids_np, dtype=torch.long, device="cpu"), 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        self._assert_samples_in_range(sampled_friction, "friction_range")

        for env_id, friction in zip(env_ids_np, sampled_friction):
            env_index = int(env_id)
            friction_value = float(friction)
            self.friction_coeffs[env_index, 0] = friction_value
            if self.cfg.domain_rand.randomize_friction:
                shape_props = self.gym.get_actor_rigid_shape_properties(
                    self.envs[env_index], self.actor_handles[env_index]
                )
                for prop in shape_props:
                    prop.friction = friction_value
                self.gym.set_actor_rigid_shape_properties(
                    self.envs[env_index], self.actor_handles[env_index], shape_props
                )

        if hasattr(self, "base_body_masses"):
            if self.cfg.domain_rand.randomize_base_mass:
                sampled_added_mass = self._sample_range("added_mass_range", env_ids_np.size)
            else:
                sampled_added_mass = np.zeros(env_ids_np.size, dtype=np.float32)
            self._assert_samples_in_range(sampled_added_mass, "added_mass_range")
            for env_id, added_mass in zip(env_ids_np, sampled_added_mass):
                env_index = int(env_id)
                added_mass_value = float(added_mass)
                self.added_mass_offsets[env_index] = added_mass_value
                if self.cfg.domain_rand.randomize_base_mass:
                    body_props = self.gym.get_actor_rigid_body_properties(
                        self.envs[env_index], self.actor_handles[env_index]
                    )
                    body_props[0].mass = float(self.base_body_masses[env_index] + added_mass_value)
                    self.gym.set_actor_rigid_body_properties(
                        self.envs[env_index],
                        self.actor_handles[env_index],
                        body_props,
                        recomputeInertia=True,
                    )
        else:
            sampled_added_mass = np.zeros(env_ids_np.size, dtype=np.float32)

        sampled_motor_strength = self._sample_range_per_leg(
            "motor_strength_range",
            "motor_strength_range_per_leg",
            env_ids_np.size,
            num_legs=4,
        )
        sampled_joint_damping = self._sample_range_per_leg(
            "joint_damping_range",
            "joint_damping_range_per_leg",
            env_ids_np.size,
            num_legs=4,
        )
        sampled_static_joint_friction = self._sample_range_per_leg(
            "static_joint_friction_range",
            "static_joint_friction_range_per_leg",
            env_ids_np.size,
            num_legs=4,
        )
        sampled_obs_noise = self._sample_range("observation_noise_range", env_ids_np.size)
        self._assert_samples_in_range(sampled_motor_strength, "motor_strength_range")
        self._assert_samples_in_range(sampled_joint_damping, "joint_damping_range")
        self._assert_samples_in_range(sampled_static_joint_friction, "static_joint_friction_range")
        self._assert_samples_in_range(sampled_obs_noise, "observation_noise_range")

        sampled_motor_strength_t = torch.from_numpy(sampled_motor_strength).to(device=self.device, dtype=torch.float)
        sampled_joint_damping_t = torch.from_numpy(sampled_joint_damping).to(device=self.device, dtype=torch.float)
        sampled_static_joint_friction_t = torch.from_numpy(sampled_static_joint_friction).to(
            device=self.device, dtype=torch.float
        )
        sampled_obs_noise_t = torch.from_numpy(sampled_obs_noise).to(device=self.device, dtype=torch.float)

        self.motor_strength_scales[env_ids_torch] = sampled_motor_strength_t
        self.joint_damping_scales[env_ids_torch] = sampled_joint_damping_t
        self.static_joint_friction[env_ids_torch] = sampled_static_joint_friction_t
        self.obs_noise_scales[env_ids_torch] = sampled_obs_noise_t

        self._set_theta_column(
            env_ids_torch,
            "added_mass",
            torch.from_numpy(sampled_added_mass).to(device=self.device, dtype=torch.float),
        )
        self._set_theta_column(
            env_ids_torch,
            "surface_friction",
            torch.from_numpy(sampled_friction).to(device=self.device, dtype=torch.float),
        )
        self._set_theta_column(env_ids_torch, "motor_strength_FL", sampled_motor_strength_t[:, 0])
        self._set_theta_column(env_ids_torch, "motor_strength_FR", sampled_motor_strength_t[:, 1])
        self._set_theta_column(env_ids_torch, "motor_strength_RL", sampled_motor_strength_t[:, 2])
        self._set_theta_column(env_ids_torch, "motor_strength_RR", sampled_motor_strength_t[:, 3])
        self._set_theta_column(env_ids_torch, "joint_damping_FL", sampled_joint_damping_t[:, 0])
        self._set_theta_column(env_ids_torch, "joint_damping_FR", sampled_joint_damping_t[:, 1])
        self._set_theta_column(env_ids_torch, "joint_damping_RL", sampled_joint_damping_t[:, 2])
        self._set_theta_column(env_ids_torch, "joint_damping_RR", sampled_joint_damping_t[:, 3])
        self._set_theta_column(env_ids_torch, "static_joint_friction_FL", sampled_static_joint_friction_t[:, 0])
        self._set_theta_column(env_ids_torch, "static_joint_friction_FR", sampled_static_joint_friction_t[:, 1])
        self._set_theta_column(env_ids_torch, "static_joint_friction_RL", sampled_static_joint_friction_t[:, 2])
        self._set_theta_column(env_ids_torch, "static_joint_friction_RR", sampled_static_joint_friction_t[:, 3])

        if self.env_theta.shape != (self.num_envs, len(self.THETA_KEYS)):
            raise ValueError(
                f"[Go2 DR] env_theta shape mismatch: got {tuple(self.env_theta.shape)}, "
                f"expected {(self.num_envs, len(self.THETA_KEYS))}"
            )
        sampled_theta_np = self.env_theta[env_ids_torch].detach().cpu().numpy()
        for range_key, theta_keys in self.RANGE_KEY_TO_COLUMN_SPECS:
            low, high = self.domain_rand_current_ranges[range_key]
            for key in theta_keys:
                col_idx = self._theta_key_to_col[key]
                col_values = sampled_theta_np[:, col_idx]
                if np.any(col_values < (low - 1.0e-5)) or np.any(col_values > (high + 1.0e-5)):
                    raise ValueError(
                        f"[Go2 DR] Theta key '{key}' has out-of-range values for {range_key}: "
                        f"expected [{low}, {high}]"
                    )
        self._log_theta_stats(sampled_theta_np)
