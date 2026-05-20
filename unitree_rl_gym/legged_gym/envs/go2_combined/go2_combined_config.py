from legged_gym.envs.go2_cdr.go2_cdr_config import GO2CDRRoughCfg, GO2CDRRoughCfgPPO


class GO2CombinedRoughCfg(GO2CDRRoughCfg):
    class env(GO2CDRRoughCfg.env):
        termination_grace_steps = 0
    class domain_rand(GO2CDRRoughCfg.domain_rand):
        class cdr(GO2CDRRoughCfg.domain_rand.cdr):
            # Slow down CDR expansion to give UPESI enough adaptation time on hard levels.
            level_step = 0.03
            update_interval = 50
            upesi_gate_enabled = True
            upesi_gate_metric = "upesi_loss_ident"
            upesi_gate_ewma_beta = 0.8
            upesi_gate_delta_max = 0.05
            upesi_gate_abs_min = 0.0012
            upesi_gate_ref_multiplier = 1.8
            upesi_gate_ref_windows = 5
            upesi_gate_cooldown_windows = 2


class GO2CombinedRoughCfgPPO(GO2CDRRoughCfgPPO):
    class algorithm(GO2CDRRoughCfgPPO.algorithm):
        use_cvar = False
        cvar_use_base_dr_only = False

    class upesi(GO2CDRRoughCfgPPO.upesi):
        enabled = True
        embedding_dim = 12
        lambda_rec = 0.20
        dynamics_updates_per_iter = 2
        identification_lr = 5.0e-4
        identification_steps = 500
        # Freeze encoder earlier once representation becomes stable.
        freeze_encoder_after_iter = 4500
        # Increase only online_identified evaluation horizon in play.py.
        online_eval_rollout_multiplier = 1.0
        # Override per-episode timeout length (seconds) only for online_identified mode.
        online_identified_episode_length_s = 30.0
        # Stabilization warm-start for play/eval before commanded forward motion.
        eval_startup_stand_steps = 0
        eval_startup_ramp_steps = 0
        eval_startup_hold_command = True
        

    class runner(GO2CDRRoughCfgPPO.runner):
        experiment_name = "rough_go2_combined"
