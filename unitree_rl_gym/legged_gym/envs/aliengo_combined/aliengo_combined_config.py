from legged_gym.envs.aliengo.aliengo_config import AliengoRoughCfg, AliengoRoughCfgPPO


class AliengoCombinedCfg(AliengoRoughCfg):
    class env(AliengoRoughCfg.env):
        termination_grace_steps = 0

    class domain_rand(AliengoRoughCfg.domain_rand):
        # Main DR ranges used by CDR interpolation (kept local to Aliengo combined).
        randomize_friction = True
        friction_range = [0.25, 1.75]
        friction_nominal = 1.0
        friction_max_range = [0.25, 1.75]

        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        added_mass_nominal = 0.0
        added_mass_max_range = [-1.0, 1.0]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        push_vel_xy_nominal = 0.0
        push_vel_xy_max_range = [-1.0, 1.0]

        motor_strength_range = [0.9, 1.1]
        motor_strength_nominal = 1.0
        motor_strength_max_range = [0.9, 1.1]

        joint_damping_range = [0.9, 1.1]
        joint_damping_nominal = 1.0
        joint_damping_max_range = [0.9, 1.1]

        static_joint_friction_range = [0.0, 0.6]
        static_joint_friction_nominal = 0.0
        static_joint_friction_max_range = [0.0, 0.6]
        static_friction_eps = 0.05

        observation_noise_range = [0.5, 1.5]
        observation_noise_nominal = 1.0
        observation_noise_max_range = [0.5, 1.5]

        class cdr(AliengoRoughCfg.domain_rand.cdr):
            # CDR enable flag expected by current environment logic.
            enabled = True
            initial_level = 0.1
            max_level = 1.0
            # Same curriculum cadence as go2_combined, kept conservative for Aliengo.
            level_step = 0.03
            update_interval = 100
            success_threshold = 0.8
            min_episodes_for_update = 500
            use_stagewise_progression = False

            # Gate CDR growth by UPESI stability.
            upesi_gate_enabled = True
            upesi_gate_metric = "upesi_loss_ident"
            upesi_gate_ewma_beta = 0.8
            upesi_gate_delta_max = 0.05
            upesi_gate_abs_min = 0.0012
            upesi_gate_ref_multiplier = 1.8
            upesi_gate_ref_windows = 5
            upesi_gate_cooldown_windows = 2


class AliengoCombinedCfgPPO(AliengoRoughCfgPPO):
    class algorithm(AliengoRoughCfgPPO.algorithm):
        # CVaR module enable/disable flag used by the current PPO implementation.
        use_cvar = True
        cvar_alpha = 0.05
        cvar_tail_weight = 1.2
        cvar_min_completed_episodes = 85
        cvar_use_base_dr_only = False

    class upesi(AliengoRoughCfgPPO.upesi):
        # UPESI module enable/disable flag used by the current runner/PPO pipeline.
        enabled = True

        # Keep 14D theta layout aligned with Go2Robot.THETA_KEYS.
        embedding_dim = 12
        theta_dim = 14
        alpha_scale = 1.0
        policy_alpha_input_scale = 1.0
        theta_keys = [
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

        # Conservative normalization bounds that cover Aliengo combined DR/CDR ranges.
        theta_min = [
            -1.0,
            0.25,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        theta_max = [
            1.0,
            1.75,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            0.6,
            0.6,
            0.6,
            0.6,
        ]

        # Combined defaults (same structure as go2_combined, Aliengo-safe values).
        dynamics_lr = 1.0e-4
        dynamics_batch_size = 4096
        lambda_rec = 0.25
        buffer_size = 500000
        predict_delta_obs = True
        dynamics_include_base_lin_vel = True
        detach_encoder_for_ppo = True
        dynamics_updates_per_iter = 1
        identification_lr = 5.0e-4
        identification_steps = 500
        freeze_encoder_after_iter = 4500
        identified_eval_episode_length_s = 5.0
        online_window_size = 2048
        online_min_buffer_size = 256
        online_update_interval = 128
        online_alpha_init = "nominal"
        online_alpha_file = ""
        online_enable_updates = True
        online_alpha_smoothing_beta = 0.2
        online_max_alpha_norm = 10.0
        online_identify_accept_ratio = 0.998
        online_save_final_alpha = False
        online_eval_rollout_multiplier = 1.0
        online_identified_episode_length_s = 20.0
        eval_startup_stand_steps = 0
        eval_startup_ramp_steps = 0
        eval_startup_hold_command = True

    class runner(AliengoRoughCfgPPO.runner):
        experiment_name = "rough_aliengo_combined"
