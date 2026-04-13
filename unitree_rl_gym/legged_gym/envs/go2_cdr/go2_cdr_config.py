from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class GO2CDRRoughCfg(GO2RoughCfg):
    class domain_rand(GO2RoughCfg.domain_rand):
        class cdr(GO2RoughCfg.domain_rand.cdr):
            enabled = True
            initial_level = 0.1
            max_level = 1.0
            level_step = 0.05
            update_interval = 20
            success_threshold = 0.8
            min_episodes_for_update = 100
            use_stagewise_progression = False


class GO2CDRRoughCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = 'rough_go2_cdr'
