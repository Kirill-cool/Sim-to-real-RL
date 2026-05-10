from legged_gym.envs.go2_cdr.go2_cdr_config import GO2CDRRoughCfg, GO2CDRRoughCfgPPO


class GO2CombinedRoughCfg(GO2CDRRoughCfg):
    class env(GO2CDRRoughCfg.env):
        termination_grace_steps = 5


class GO2CombinedRoughCfgPPO(GO2CDRRoughCfgPPO):
    class algorithm(GO2CDRRoughCfgPPO.algorithm):
        use_cvar = True
        cvar_use_base_dr_only = False

    class upesi(GO2CDRRoughCfgPPO.upesi):
        enabled = True
        # Increase only online_identified evaluation horizon in play.py.
        online_eval_rollout_multiplier = 1.0
        # Override per-episode timeout length (seconds) only for online_identified mode.
        online_identified_episode_length_s = 40.0
        # Stabilization warm-start for play/eval before commanded forward motion.
        eval_startup_stand_steps = 0
        eval_startup_ramp_steps = 0
        eval_startup_hold_command = True
        

    class runner(GO2CDRRoughCfgPPO.runner):
        experiment_name = "rough_go2_combined"
