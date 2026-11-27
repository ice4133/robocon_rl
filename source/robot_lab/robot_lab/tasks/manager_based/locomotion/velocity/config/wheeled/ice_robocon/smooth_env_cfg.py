# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Conservative smoothness fine-tuning of 2026-11-24_23-11-40/model_4999.pt."""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.ice import ROBOCON_WHEEL_RADIUS
from robot_lab.tasks.manager_based.locomotion.velocity import mdp

from .commands import BalancedVelocityCommandCfg
from .flat_env_cfg import IceRoboconFlatEnvCfg
from .rough_env_cfg import IceRoboconRewardsCfg
from .smooth_rewards import JointTargetSmoothness, airborne_wheel_overspeed, straight_rolling_l2


@configclass
class IceRoboconSmoothRewardsCfg(IceRoboconRewardsCfg):
    joint_target_smoothness = RewTerm(
        func=JointTargetSmoothness,
        weight=-0.01,
        params={
            "position_action_name": "joint_pos", "velocity_action_name": "joint_vel",
            "leg_second_difference_scale": 0.05, "wheel_second_difference_scale": 1.0,
        },
    )
    airborne_wheel_overspeed = RewTerm(
        func=airborne_wheel_overspeed,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[f"{leg}_foot_link" for leg in ("fr", "fl", "rr", "rl")],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"{leg}_foot_joint" for leg in ("fr", "fl", "rr", "rl")],
                preserve_order=True,
            ),
            "wheel_radius": ROBOCON_WHEEL_RADIUS, "turn_radius": 0.4,
            "speed_margin": 2.0, "air_time_threshold": 0.08, "speed_scale": 30.0,
        },
    )


@configclass
class IceRoboconFlatSmoothEnvCfg(IceRoboconFlatEnvCfg):
    rewards: IceRoboconSmoothRewardsCfg = IceRoboconSmoothRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Restore the best policy's objective, rather than inherit the failed v2
        # tuning. Keep the policy/observation, actuator and target-limit contract.
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.action_rate_l2.weight = -0.03
        self.rewards.joint_acc_l2.weight = -7.5e-7
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params.update(
            stand_still_scale=5.0, velocity_threshold=0.5, command_threshold=0.03,
        )
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["command_threshold"] = 0.03
        # Retain 4999's existing lateral swing term (this is not a new reward).
        self.rewards.feet_air_time = RewTerm(
            func=mdp.lateral_feet_air_time, weight=0.5,
            params={
                "command_name": "base_velocity", "threshold": 0.3, "max_air_time": 0.5,
                "command_threshold": 0.03,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
            },
        )
        # Keep touchdown counting disabled: stationary contact flicker is not a gait.
        self.rewards.feet_contact_without_cmd = None
        command = self.commands.base_velocity
        command.ranges.lin_vel_x = (-2.0, 2.0)
        command.resampling_time_range = (6.0, 9.0)
        command.rel_standing_envs = 0.10
        command.mode_probabilities = (0.10, 0.25, 0.25, 0.20, 0.20)
        command.low_speed_fraction = 0.20
        command.low_speed_ranges = BalancedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.3, 0.3), heading=None,
        )
        command.high_speed_fraction = 0.0
        # No range expansion during this smoothness fine-tuning stage.
        command.mixed_ranges = BalancedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.4, 0.4), heading=None,
        )


@configclass
class IceRoboconStraightRewardsCfg(IceRoboconSmoothRewardsCfg):
    straight_rolling = RewTerm(
        func=straight_rolling_l2, weight=-0.2,
        params={
            "command_name": "base_velocity",
            "wheel_asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"{leg}_foot_joint" for leg in ("fr", "fl", "rr", "rl")],
                preserve_order=True,
            ),
            "leg_asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"{leg}_{joint}_joint" for leg in ("fr", "fl", "rr", "rl")
                                      for joint in ("hip", "thigh", "calf")],
                preserve_order=True,
            ),
            "wheel_direction_signs": (-1.0, 1.0, -1.0, 1.0),
            "wheel_radius": ROBOCON_WHEEL_RADIUS,
            "wheel_speed_error_scale": 0.25, "lateral_speed_scale": 0.10, "yaw_rate_scale": 0.05,
            "leg_speed_scale": 0.5, "leg_speed_deadband": 0.05,
            "forward_command_threshold": 0.05, "other_command_threshold": 1.0e-6,
        },
    )


@configclass
class IceRoboconFlatSmoothStraightEnvCfg(IceRoboconFlatSmoothEnvCfg):
    """Refine Smooth 7998's straightness without changing other motion objectives."""

    rewards: IceRoboconStraightRewardsCfg = IceRoboconStraightRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_target_smoothness.weight = -0.02
        command = self.commands.base_velocity
        command.rel_standing_envs = 0.05
        command.mode_probabilities = (0.05, 0.35, 0.25, 0.15, 0.20)
        command.resampling_time_range = (8.0, 12.0)
        # Include more near-limit forward AND reverse samples, where FL unloads.
        command.high_speed_fraction = 0.10
        command.high_speed_min = 1.75
