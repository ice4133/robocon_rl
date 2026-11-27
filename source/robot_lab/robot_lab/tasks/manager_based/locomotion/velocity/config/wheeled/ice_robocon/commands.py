# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Balanced body-velocity commands for Robocon flat-ground fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class BalancedVelocityCommand(UniformVelocityCommand):
    """Sample stand, forward/reverse, strafe, turn and mixed commands independently of heading.

    Each resample chooses a new mode, including during an episode. Low-speed
    commands are retained verbatim rather than cleared by a translation deadzone.
    """

    cfg: BalancedVelocityCommandCfg

    def __init__(self, cfg: BalancedVelocityCommandCfg, env):
        probabilities = cfg.mode_probabilities
        if (
            len(probabilities) != 5
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1.0e-6)
        ):
            raise ValueError("mode_probabilities must contain five nonnegative probabilities summing to one")
        if cfg.heading_command:
            raise ValueError("BalancedVelocityCommand requires heading_command=False")
        if not math.isclose(cfg.rel_standing_envs, probabilities[0], abs_tol=1.0e-6):
            raise ValueError("rel_standing_envs must match the standing mode probability")
        if not 0 <= cfg.low_speed_fraction <= 1:
            raise ValueError("low_speed_fraction must be between zero and one")
        if not 0 <= cfg.high_speed_fraction <= 1 - cfg.low_speed_fraction:
            raise ValueError("Low/high speed fractions must be nonnegative and sum to at most one")
        if cfg.high_speed_fraction and not 0 < cfg.high_speed_min < min(
            -cfg.ranges.lin_vel_x[0], cfg.ranges.lin_vel_x[1]
        ):
            raise ValueError("high_speed_min must lie inside both forward/reverse ranges")
        if cfg.rolling_speed_limit is not None:
            if not math.isfinite(cfg.rolling_speed_limit) or cfg.rolling_speed_limit <= 0:
                raise ValueError("rolling_speed_limit must be finite and positive")
            if max(abs(v) for v in cfg.ranges.lin_vel_x) > cfg.rolling_speed_limit:
                raise ValueError("Forward command exceeds the configured wheel radius * output-speed bound")
        for limits in (cfg.acceleration_limits, cfg.deceleration_limits):
            if len(limits) != 3 or any(not math.isfinite(v) or v <= 0 for v in limits):
                raise ValueError("Acceleration/deceleration limits require three finite positive values")
        for ranges in (cfg.ranges, cfg.low_speed_ranges, cfg.mixed_ranges):
            if ranges is None:
                continue
            for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                lower, upper = getattr(ranges, axis)
                if not (math.isfinite(lower) and math.isfinite(upper) and lower <= 0 <= upper):
                    raise ValueError(f"{axis} ranges must be finite and include zero")
                main_lower, main_upper = getattr(cfg.ranges, axis)
                if ranges is cfg.mixed_ranges and (lower < main_lower or upper > main_upper):
                    raise ValueError("Mixed-command ranges must fit inside the full command ranges")
        super().__init__(cfg, env)
        self._mode_probabilities = torch.tensor(probabilities, device=self.device)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_command_b = torch.zeros_like(self.vel_command_b)
        self._acceleration = torch.tensor(cfg.acceleration_limits, device=self.device)
        self._deceleration = torch.tensor(cfg.deceleration_limits, device=self.device)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        count = len(env_ids)
        if count == 0:
            return
        modes = torch.multinomial(self._mode_probabilities, count, replacement=True)
        speed_group = torch.rand(count, device=self.device)
        low_speed = speed_group < self.cfg.low_speed_fraction
        high_speed = (speed_group >= 1 - self.cfg.high_speed_fraction) & (modes == 1)
        commands = torch.empty(count, 3, device=self.device)
        for axis, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            lower, upper = getattr(self.cfg.ranges, name)
            low_lower, low_upper = getattr(self.cfg.low_speed_ranges, name)
            # Respect runtime range reductions as well as the configured low-speed range.
            low_lower, low_upper = max(lower, low_lower), min(upper, low_upper)
            values = torch.empty(count, device=self.device).uniform_(lower, upper)
            if self.cfg.mixed_ranges is not None:
                mixed_lower, mixed_upper = getattr(self.cfg.mixed_ranges, name)
                mixed_values = torch.empty_like(values).uniform_(mixed_lower, mixed_upper)
                values = torch.where(modes == 4, mixed_values, values)
            low_values = torch.empty_like(values).uniform_(low_lower, low_upper)
            commands[:, axis] = torch.where(low_speed, low_values, values)
        if self.cfg.high_speed_fraction:
            lower, upper = self.cfg.ranges.lin_vel_x
            positive = torch.rand(count, device=self.device) < 0.5
            forward = torch.empty(count, device=self.device).uniform_(self.cfg.high_speed_min, upper)
            reverse = -torch.empty(count, device=self.device).uniform_(self.cfg.high_speed_min, -lower)
            commands[:, 0] = torch.where(high_speed, torch.where(positive, forward, reverse), commands[:, 0])
        # 0: stand; 1: forward/reverse; 2: strafe; 3: turn; 4: mixed.
        commands[:, 0] *= (modes == 1) | (modes == 4)
        commands[:, 1] *= (modes == 2) | (modes == 4)
        commands[:, 2] *= (modes == 3) | (modes == 4)
        self.target_command_b[env_ids] = commands
        if not self.cfg.ramp_commands:
            self.vel_command_b[env_ids] = commands
        self.mode[env_ids] = modes
        self.is_standing_env[env_ids] = modes == 0
        self.is_heading_env[env_ids] = False

    def _update_command(self):
        if not self.cfg.ramp_commands:
            super()._update_command()
            return
        # Match the deployed FSM: brake to zero before reversing an axis.
        current, target = self.vel_command_b, self.target_command_b
        reversing = current * target < 0
        speeding_up = (~reversing) & (target.abs() > current.abs())
        maximum_step = torch.where(speeding_up, self._acceleration, self._deceleration) * self._env.step_dt
        target = torch.where(reversing, 0.0, target)
        current += torch.clamp(target - current, min=-maximum_step, max=maximum_step)

    def reset(self, env_ids: Sequence[int] | None = None):
        extras = super().reset(env_ids)
        if self.cfg.ramp_commands:
            self.vel_command_b[slice(None) if env_ids is None else env_ids] = 0
        return extras


@configclass
class BalancedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Mode probabilities are ordered stand, forward/reverse, strafe, turn, mixed."""

    class_type: type = BalancedVelocityCommand
    heading_command: bool = False
    rel_heading_envs: float = 0.0
    rel_standing_envs: float = 0.2
    mode_probabilities: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    low_speed_fraction: float = 0.5
    high_speed_fraction: float = 0.0
    high_speed_min: float = 1.75
    mixed_ranges: UniformVelocityCommandCfg.Ranges | None = None
    rolling_speed_limit: float | None = None
    ramp_commands: bool = False
    acceleration_limits: tuple[float, float, float] = (1.0, 0.3, 0.8)
    deceleration_limits: tuple[float, float, float] = (1.5, 0.5, 1.2)
    low_speed_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.3, 0.3)
    )
