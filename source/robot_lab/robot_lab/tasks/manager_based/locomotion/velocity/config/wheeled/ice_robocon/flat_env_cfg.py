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

        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
        # Keep lateral speed and yaw ranges; change the stepping style, not the requested strafe speed.
        self.actions.joint_pos.scale = {".*_hip_joint": 0.18, "^(?!.*_hip_joint).*": 0.25}
        self.rewards.action_rate_l2.weight = -0.03
        self.rewards.joint_acc_l2.weight = -7.5e-7
        # Give lateral swings time to cover more ground, without inducing hopping when rolling straight.
        self.rewards.feet_air_time.func = mdp.lateral_feet_air_time
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params.update(
            threshold=0.30,
            max_air_time=0.50,
            command_threshold=0.10,
        )

        self.disable_zero_weight_rewards()
