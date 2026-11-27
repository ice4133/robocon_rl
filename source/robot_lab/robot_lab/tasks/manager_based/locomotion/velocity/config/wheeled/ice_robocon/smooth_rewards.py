# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Small, robot-specific additions for fine-tuning the selected 4999 baseline."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg


class JointTargetSmoothness(ManagerTermBase):
    """Penalize curvature of the clipped joint targets, not raw network outputs.

    Leg targets are positions and wheel targets are velocities. Separate scales
    keep their units comparable. The nominal differences refer to a 20 ms policy
    step; timestep correction retains the same acceleration/jerk interpretation.
    Two warm-up samples after each reset exclude episode-boundary discontinuities.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._position = env.action_manager.get_term(cfg.params["position_action_name"])
        self._velocity = env.action_manager.get_term(cfg.params["velocity_action_name"])
        self._previous = torch.zeros_like(self._targets())
        self._previous_previous = torch.zeros_like(self._previous)
        self._samples = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def _targets(self):
        return torch.cat((self._position.processed_actions, self._velocity.processed_actions), dim=1)

    def reset(self, env_ids=None):
        self._samples[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self, env, position_action_name: str, velocity_action_name: str,
        leg_second_difference_scale: float, wheel_second_difference_scale: float,
    ):
        if leg_second_difference_scale <= 0 or wheel_second_difference_scale <= 0:
            raise ValueError("Target smoothness scales must be positive")
        targets = self._targets()
        curvature = (targets - 2 * self._previous + self._previous_previous) * (0.02 / env.step_dt) ** 2
        legs = self._position.processed_actions.shape[1]
        cost = (curvature[:, :legs] / leg_second_difference_scale).square().mean(dim=1)
        cost += (curvature[:, legs:] / wheel_second_difference_scale).square().mean(dim=1)
        cost = torch.where(self._samples >= 2, cost, 0.0)
        self._previous_previous.copy_(self._previous)
        self._previous.copy_(targets)
        self._samples.add_(1).clamp_(max=2)
        return cost


def airborne_wheel_overspeed(
    env, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    wheel_radius: float, turn_radius: float, speed_margin: float,
    air_time_threshold: float, speed_scale: float,
):
    """Penalize sustained airborne overspeed without braking useful rolling.

    Body/joint selections must list the same wheels in the same order. Allow
    forward/yaw rolling speed plus a margin; normal short swings are exempt.
    The native wheel_vel_penalty only samples first liftoff, so it cannot replace
    this sustained-airborne check.
    """
    if wheel_radius <= 0 or speed_scale <= 0 or min(turn_radius, speed_margin, air_time_threshold) < 0:
        raise ValueError("Invalid airborne wheel overspeed parameters")
    sensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    speed = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids].abs()
    if speed.shape != air_time.shape:
        raise ValueError("Wheel joint and contact-body selections must have matching dimensions")
    command = env.command_manager.get_command(command_name)
    allowance = (command[:, 0].abs() + turn_radius * command[:, 2].abs()) / wheel_radius + speed_margin
    excess = torch.relu(speed - allowance[:, None]) / speed_scale
    return (excess.square() * (air_time > air_time_threshold)).mean(dim=1)


def straight_rolling_l2(
    env, command_name: str, wheel_asset_cfg: SceneEntityCfg, leg_asset_cfg: SceneEntityCfg,
    wheel_direction_signs: tuple[float, ...], wheel_radius: float,
    wheel_speed_error_scale: float, lateral_speed_scale: float, yaw_rate_scale: float,
    leg_speed_scale: float, leg_speed_deadband: float, forward_command_threshold: float,
    other_command_threshold: float,
):
    """Constrain straight rolling, including a single restless or slow wheel.

    Apply only to pure forward/reverse commands, never requested lateral/yaw
    motion (including small nonzero commands). URDF wheel-axis signs convert
    angular velocities to the same physical forward direction. Match wheel
    surface speed to measured body speed so acceleration is not penalized for
    lagging a command. This is a soft flat-ground constraint, not a heading lock.
    """
    scales = (wheel_radius, wheel_speed_error_scale, lateral_speed_scale, yaw_rate_scale, leg_speed_scale)
    if any(value <= 0 for value in scales) or min(
        leg_speed_deadband, forward_command_threshold, other_command_threshold
    ) < 0:
        raise ValueError("Straight rolling scales must be positive and thresholds nonnegative")
    asset = env.scene[wheel_asset_cfg.name]
    if leg_asset_cfg.name != wheel_asset_cfg.name:
        raise ValueError("Straight rolling legs and wheels must select the same robot")
    command = env.command_manager.get_command(command_name)
    straight = (
        (command[:, 0].abs() > forward_command_threshold)
        & (command[:, 1].abs() <= other_command_threshold)
        & (command[:, 2].abs() <= other_command_threshold)
    )
    wheel_velocity = asset.data.joint_vel.torch[:, wheel_asset_cfg.joint_ids]
    signs = wheel_velocity.new_tensor(wheel_direction_signs)
    if len(wheel_direction_signs) != wheel_velocity.shape[1] or any(abs(s) != 1 for s in wheel_direction_signs):
        raise ValueError("Provide one +1/-1 URDF direction sign per selected wheel")
    linear = asset.data.root_lin_vel_b.torch
    angular = asset.data.root_ang_vel_b.torch
    surface_speed = wheel_velocity * signs * wheel_radius
    wheel_cost = ((surface_speed - linear[:, :1]) / wheel_speed_error_scale).square().mean(dim=1)
    lateral_cost = (linear[:, 1] / lateral_speed_scale).square()
    yaw_cost = (angular[:, 2] / yaw_rate_scale).square()
    leg_velocity = asset.data.joint_vel.torch[:, leg_asset_cfg.joint_ids]
    if leg_velocity.shape[1] % 3:
        raise ValueError("Leg selection must group hip/thigh/calf for each leg")
    moving = torch.relu(leg_velocity.abs() - leg_speed_deadband) / leg_speed_scale
    per_leg_cost = moving.square().reshape(env.num_envs, -1, 3).mean(dim=2)
    # A single moving leg must not disappear in the mean of twelve joints.
    leg_cost = 0.5 * (per_leg_cost.mean(dim=1) + per_leg_cost.amax(dim=1))
    return (wheel_cost + lateral_cost + yaw_cost + leg_cost) * straight
