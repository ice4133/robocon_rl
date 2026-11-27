# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class IceRoboconRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "ice_robocon_rough"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class IceRoboconFlatPPORunnerCfg(IceRoboconRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 8000
        self.experiment_name = "ice_robocon_flat"
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.005


@configclass
class IceRoboconFlatSpeedPPORunnerCfg(IceRoboconFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "ice_robocon_flat_speed"
        self.algorithm.learning_rate = 2.0e-4
        self.algorithm.entropy_coef = 0.003


@configclass
class IceRoboconFlatAirTimePPORunnerCfg(IceRoboconFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "ice_robocon_flat_airtime"


@configclass
class IceRoboconFlatSmoothPPORunnerCfg(IceRoboconFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "ice_robocon_smooth_from4999"
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.002
        self.algorithm.clip_param = 0.1


@configclass
class IceRoboconFlatSmoothStraightPPORunnerCfg(IceRoboconFlatSmoothPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.experiment_name = "ice_robocon_smooth_straight_from7998"
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.entropy_coef = 0.001
