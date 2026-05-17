from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class AliengoRoughCfg(GO2RoughCfg):
    class init_state(GO2RoughCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x, y, z [m]
        default_joint_angles = {
            "FL_hip_joint": -0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": -0.0,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }

    class control(GO2RoughCfg.control):
        control_type = "P"
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.2}
        action_scale = 0.25
        decimation = 4

    class asset(GO2RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_description/urdf/aliengo.urdf"
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "trunk",
            "base",
            "FL_thigh",
            "FR_thigh",
            "RL_thigh",
            "RR_thigh",
            "FL_calf",
            "FR_calf",
            "RL_calf",
            "RR_calf",
        ]
        self_collisions = 1

    class domain_rand(GO2RoughCfg.domain_rand):
        # Use Aliengo-oriented DR ranges from the reference config.
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

        # Keep Go2Robot DR mechanics, but narrow to Aliengo reference-style ranges.
        motor_strength_range = [0.9, 1.1]
        motor_strength_nominal = 1.0
        motor_strength_max_range = [0.9, 1.1]

        joint_damping_range = [0.9, 1.1]
        joint_damping_nominal = 1.0
        joint_damping_max_range = [0.9, 1.1]

    class rewards(GO2RoughCfg.rewards):
        base_height_target = 0.43

    class commands(GO2RoughCfg.commands):
        heading_command = False

        class ranges(GO2RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 2.4]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1.57, 1.57]
            heading = [-3.14, 3.14]


class AliengoRoughCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = "rough_aliengo"
