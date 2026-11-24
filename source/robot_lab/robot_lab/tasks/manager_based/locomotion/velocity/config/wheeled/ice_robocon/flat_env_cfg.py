# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Flat-ground Robocon task, the starting point for a new training run."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity import mdp

from .rough_env_cfg import IceRoboconRoughEnvCfg


@configclass
class IceRoboconFlatEnvCfg(IceRoboconRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.curriculum.terrain_levels = None
        self.terminations.fallen = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.18})

        self.disable_zero_weight_rewards()
