# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Flat-ground Robocon task, the starting point for a new training run."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_lab.assets.ice import ROBOCON_WHEEL_RADIUS, ROBOCON_WHEEL_VELOCITY_LIMIT
from robot_lab.tasks.manager_based.locomotion.velocity import mdp

from .commands import BalancedVelocityCommandCfg
from .rough_env_cfg import IceRoboconRoughEnvCfg


@configclass
class IceRoboconFlatEnvCfg(IceRoboconRoughEnvCfg):
    def _speed_stage(self) -> bool:
        return False

    def _air_time_weight(self) -> float:
        return 0.0

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

        # Learn coordinated modes before introducing the final rolling-speed range.
        speed_stage = self._speed_stage()
        forward_limit = 2.5 if speed_stage else 1.5
        self.commands.base_velocity = BalancedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 7.0),
            debug_vis=True,
            vel_xy_success_threshold=0.05,
            vel_yaw_success_threshold=0.05,
            rel_standing_envs=0.05,
            mode_probabilities=(0.05, 0.20, 0.30, 0.20, 0.25),
            low_speed_fraction=0.30,
            high_speed_fraction=0.10 if speed_stage else 0.0,
            high_speed_min=1.75,
            rolling_speed_limit=ROBOCON_WHEEL_RADIUS * ROBOCON_WHEEL_VELOCITY_LIMIT,
            ramp_commands=True,
            ranges=BalancedVelocityCommandCfg.Ranges(
                lin_vel_x=(-forward_limit, forward_limit), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8), heading=None
            ),
            # Spend more samples around the small strafe/yaw commands that the
            # first native run ignored, without dropping full-range coverage.
            low_speed_ranges=BalancedVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.12, 0.12), ang_vel_z=(-0.25, 0.25), heading=None
            ),
            # Leave wheel-speed headroom when translation and yaw are combined.
            mixed_ranges=BalancedVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.5, 1.5) if speed_stage else (-0.5, 0.5),
                lin_vel_y=(-0.3, 0.3) if speed_stage else (-0.2, 0.2),
                ang_vel_z=(-0.6, 0.6) if speed_stage else (-0.4, 0.4),
                heading=None,
            ),
        )
        # Keep the action-to-joint contract across training and deployment.
        # A larger scale permits sharper joint targets; it does not ensure longer strides.
        self.actions.joint_pos.scale = {".*_hip_joint": 0.18, "^(?!.*_hip_joint).*": 0.25}
        # Tighten tracking while giving moving legs more freedom to leave the
        # default pose. Keep all terms native; do not reward contact flicker.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.4
        self.rewards.action_rate_l2.weight = -0.025
        self.rewards.joint_acc_l2.weight = -5.0e-7
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_pos_penalty.weight = -0.4
        # The stationary effective weight remains -1.6 (-0.4 * 4.0).
        self.rewards.joint_pos_penalty.params.update(stand_still_scale=4.0, command_threshold=0.03)
        self.rewards.stand_still.weight = -1.0
        self.rewards.stand_still.params["command_threshold"] = 0.03
        # This native term counts new contacts, rather than sustained support.
        # Do not incentivize repeated touchdowns while nominally stationary.
        self.rewards.feet_contact_without_cmd.weight = 0.0
        # Disabled in the native baseline. The separate AirTime task changes only
        # this term's weight, permitting a controlled, weaker swing-reward ablation.
        self.rewards.feet_air_time.func = mdp.lateral_feet_air_time
        self.rewards.feet_air_time.weight = self._air_time_weight()
        self.rewards.feet_air_time.params.update(
            threshold=0.10,
            max_air_time=0.50,
            command_threshold=0.03,
        )

        self.disable_zero_weight_rewards()


@configclass
class IceRoboconFlatSpeedEnvCfg(IceRoboconFlatEnvCfg):
    """Expand speed after the coordination checkpoint passes per-mode evaluation."""

    def _speed_stage(self) -> bool:
        return True


@configclass
class IceRoboconFlatAirTimeEnvCfg(IceRoboconFlatEnvCfg):
    """Same coordination task, with exactly one reward-weight difference."""

    def _air_time_weight(self) -> float:
        return 0.2


@configclass
class IceRoboconFlatSpeedAirTimeEnvCfg(IceRoboconFlatSpeedEnvCfg):
    """Retain the same weak air-time term if that coordination ablation wins."""

    def _air_time_weight(self) -> float:
        return 0.2
