from legged_gym.envs.go2_cdr.go2_cdr_config import GO2CDRRoughCfg, GO2CDRRoughCfgPPO


class GO2CombinedRoughCfg(GO2CDRRoughCfg):
    pass


class GO2CombinedRoughCfgPPO(GO2CDRRoughCfgPPO):
    class algorithm(GO2CDRRoughCfgPPO.algorithm):
        use_cvar = True
        cvar_use_base_dr_only = False

    class upesi(GO2CDRRoughCfgPPO.upesi):
        enabled = True
        # Increase only online_identified evaluation horizon in play.py.
        online_eval_rollout_multiplier = 10.0
        

    class runner(GO2CDRRoughCfgPPO.runner):
        experiment_name = "rough_go2_combined"
