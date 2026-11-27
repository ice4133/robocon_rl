# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Check training coverage and the command contract without starting a simulator."""

from types import SimpleNamespace

import pytest
import torch

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.ice_robocon.commands import (
    BalancedVelocityCommand,
    BalancedVelocityCommandCfg,
)


def make_command(count=10000, **overrides):
    settings = dict(
        asset_name="robot",
        debug_vis=False,
        resampling_time_range=(3.0, 6.0),
        ranges=BalancedVelocityCommandCfg.Ranges(lin_vel_x=(-2.0, 2.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8)),
    )
    settings.update(overrides)
    cfg = BalancedVelocityCommandCfg(**settings)
    registry = SimpleNamespace(clear_debug_vis_callback=lambda term: None)
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        extras={},
        scene={"robot": SimpleNamespace()},
        sim=SimpleNamespace(vis_marker_registry=registry),
    )
    return BalancedVelocityCommand(cfg, env)


def test_training_covers_each_pure_mode_and_retains_small_signed_commands():
    torch.manual_seed(42)
    term = make_command()
    term._resample_command(slice(None))
    term._update_command()
    command = term.command
    fractions = torch.bincount(term.mode, minlength=5) / term.num_envs
    torch.testing.assert_close(fractions, torch.full((5,), 0.2), atol=0.02, rtol=0)
    assert torch.all(command[term.mode == 0] == 0)
    for mode, axis, limit in [(1, 0, 2.0), (2, 1, 0.3), (3, 2, 0.8)]:
        samples = command[term.mode == mode]
        other_axes = [i for i in range(3) if i != axis]
        assert torch.all(samples[:, other_axes] == 0)
        assert torch.all(samples[:, axis].abs() <= limit)
        # Both signs and nonzero speeds below the previous 0.2 deadzone survive.
        assert torch.count_nonzero((samples[:, axis] > 0.05) & (samples[:, axis] < 0.1)) > 20
        assert torch.count_nonzero((samples[:, axis] < -0.05) & (samples[:, axis] > -0.1)) > 20
    assert torch.all(command[term.mode == 4] != 0)
    before = command.clone()
    term._update_command()
    torch.testing.assert_close(term.command, before)  # Yaw is not replaced by a heading controller.


def test_low_speed_mixture_is_configurable_without_losing_full_speed_coverage():
    torch.manual_seed(7)
    term = make_command(low_speed_fraction=1.0)
    term._resample_command(slice(None))
    assert torch.all(term.command.abs() <= torch.tensor([0.3, 0.2, 0.3]))
    term.cfg.low_speed_fraction = 0.0
    term._resample_command(slice(None))
    assert term.command[:, 0].abs().max() > 1.9
    assert term.command[:, 1].abs().max() > 0.29
    assert term.command[:, 2].abs().max() > 0.75


def test_partial_resampling_preserves_other_envs_and_can_brake_a_moving_env():
    term = make_command(count=8, mode_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0), rel_standing_envs=1.0)
    term.vel_command_b[:] = torch.tensor([0.5, 0.2, 0.6])
    term.is_standing_env[:] = False
    before = term.command.clone()
    term._resample_command(torch.tensor([1, 5]))
    term._update_command()
    assert torch.all(term.command[[1, 5]] == 0)
    torch.testing.assert_close(term.command[[0, 2, 3, 4, 6, 7]], before[[0, 2, 3, 4, 6, 7]])
    term._resample_command([])
    term._resample_command(slice(2, 4))
    assert torch.all(term.command[[2, 3]] == 0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode_probabilities": (0.2, 0.2)},
        {"mode_probabilities": (-0.2, 0.3, 0.3, 0.3, 0.3)},
        {"heading_command": True},
        {"rel_standing_envs": 0.5},
        {"low_speed_fraction": 1.1},
        {"high_speed_fraction": 0.8},
        {"high_speed_fraction": 0.2, "high_speed_min": 2.5},
        {"rolling_speed_limit": 1.5},
        {"acceleration_limits": (1.0, 0.0, 0.8)},
    ],
)
def test_invalid_sampling_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_command(**overrides)


def test_tracking_parameter_ablation_changes_only_the_requested_existing_terms():
    from isaaclab_tasks.utils import resolve_task_config

    task = "RobotLab-Isaac-Velocity-Flat-ICE-Robocon-v0"
    baseline, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    tuned, _ = resolve_task_config(
        task,
        "rsl_rl_cfg_entry_point",
        overrides=[
            "env.rewards.track_lin_vel_xy_exp.params.std=0.3",
            "env.rewards.track_ang_vel_z_exp.params.std=0.3",
        ],
    )
    assert baseline.rewards.track_lin_vel_xy_exp.params["std"] == 0.35
    assert baseline.rewards.track_ang_vel_z_exp.params["std"] == 0.4
    assert tuned.rewards.track_lin_vel_xy_exp.params["std"] == 0.3
    assert tuned.rewards.track_ang_vel_z_exp.params["std"] == 0.3
    assert tuned.rewards.track_lin_vel_xy_exp.weight == baseline.rewards.track_lin_vel_xy_exp.weight == 3.0
    assert tuned.rewards.track_ang_vel_z_exp.weight == baseline.rewards.track_ang_vel_z_exp.weight == 1.5
    assert baseline.rewards.action_rate_l2.weight == -0.025
    assert baseline.rewards.joint_acc_l2.weight == -5.0e-7
    assert baseline.rewards.feet_air_time is None
    assert baseline.rewards.feet_contact_without_cmd is None
    for name in ("stand_base_motion", "stand_wheel_velocity", "stand_wheel_action", "wheel_vel_penalty"):
        assert getattr(tuned.rewards, name) is None


def test_command_ramps_are_bounded_and_reverse_through_zero():
    term = make_command(count=8, ramp_commands=True)
    term.target_command_b[:] = torch.tensor([1.0, 0.2, 0.6])
    for _ in range(100):
        before = term.command.clone()
        term._update_command()
        assert torch.all((term.command - before).abs() <= torch.tensor([0.02, 0.006, 0.016]) + 1e-6)
    torch.testing.assert_close(term.command, term.target_command_b)
    term.target_command_b *= -1
    for _ in range(150):
        before = term.command.clone()
        term._update_command()
        assert torch.all((term.command - before).abs() <= torch.tensor([0.03, 0.01, 0.024]) + 1e-6)
        assert not torch.any((before > 1e-6) & (term.command < -1e-6))
    torch.testing.assert_close(term.command, term.target_command_b)


def test_standing_resample_brakes_instead_of_immediately_clearing_the_command():
    term = make_command(count=8, ramp_commands=True, mode_probabilities=(1.0, 0, 0, 0, 0), rel_standing_envs=1.0)
    term.vel_command_b[:] = torch.tensor([0.5, 0.2, 0.6])
    term._resample_command(slice(None))
    torch.testing.assert_close(term.command[0], torch.tensor([0.5, 0.2, 0.6]))
    term._update_command()
    torch.testing.assert_close(term.command[0], torch.tensor([0.47, 0.19, 0.576]))
    for _ in range(100):
        term._update_command()
    assert torch.all(term.command == 0)


def test_episode_reset_clears_old_ramped_commands_only_for_reset_envs():
    term = make_command(count=8, ramp_commands=True)
    term.vel_command_b[:] = torch.tensor([0.5, 0.2, 0.6])
    term.reset(torch.tensor([1, 5]))
    assert torch.all(term.command[[1, 5]] == 0)
    torch.testing.assert_close(term.command[[0, 2]], torch.tensor([[0.5, 0.2, 0.6]]).expand(2, 3))


def test_fast_rolling_profile_covers_both_directions_and_feasible_mixed_commands():
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config(
        "RobotLab-Isaac-Velocity-Flat-ICE-Robocon-Speed-v0", "rsl_rl_cfg_entry_point", overrides=[]
    )
    sampling = cfg.commands.base_velocity
    assert sampling.ranges.lin_vel_x == (-2.5, 2.5)
    assert sampling.ranges.ang_vel_z == (-0.8, 0.8)
    assert sampling.rolling_speed_limit == pytest.approx(2.64)
    assert sampling.ramp_commands
    torch.manual_seed(123)
    term = make_command(
        mode_probabilities=sampling.mode_probabilities,
        rel_standing_envs=sampling.rel_standing_envs,
        low_speed_fraction=sampling.low_speed_fraction,
        high_speed_fraction=sampling.high_speed_fraction,
        high_speed_min=sampling.high_speed_min,
        ranges=sampling.ranges,
        mixed_ranges=sampling.mixed_ranges,
        rolling_speed_limit=sampling.rolling_speed_limit,
        ramp_commands=True,
    )
    term._resample_command(slice(None))
    proportions = torch.bincount(term.mode, minlength=5) / term.num_envs
    torch.testing.assert_close(proportions, torch.tensor(sampling.mode_probabilities), atol=0.02, rtol=0)
    forward = term.target_command_b[term.mode == 1, 0]
    assert torch.count_nonzero(forward > 2.3) > 50
    assert torch.count_nonzero(forward < -2.3) > 50
    assert term.target_command_b[term.mode == 4, 0].abs().max() <= 1.5
    assert torch.all(term.command == 0)  # Actual commands start from a ramp, including at reset.


def test_coordination_stage_defers_high_speed_and_airtime_ablation_is_isolated():
    from isaaclab_tasks.utils import resolve_task_config

    task = "RobotLab-Isaac-Velocity-Flat-ICE-Robocon"
    base, _ = resolve_task_config(task + "-v0", "rsl_rl_cfg_entry_point", overrides=[])
    air, _ = resolve_task_config(task + "-AirTime-v0", "rsl_rl_cfg_entry_point", overrides=[])
    assert base.commands.base_velocity.ranges.lin_vel_x == (-1.5, 1.5)
    assert base.commands.base_velocity.mixed_ranges.lin_vel_x == (-0.5, 0.5)
    assert base.commands.base_velocity.high_speed_fraction == 0.0
    baseline_rewards = base.rewards.to_dict()
    air_rewards = air.rewards.to_dict()
    assert baseline_rewards.pop("feet_air_time") is None
    term = air_rewards.pop("feet_air_time")
    assert term["weight"] == 0.2
    assert baseline_rewards == air_rewards
    assert base.commands.to_dict() == air.commands.to_dict()
    speed, _ = resolve_task_config(task + "-Speed-v0", "rsl_rl_cfg_entry_point", overrides=[])
    speed_air, _ = resolve_task_config(task + "-Speed-AirTime-v0", "rsl_rl_cfg_entry_point", overrides=[])
    speed_rewards, speed_air_rewards = speed.rewards.to_dict(), speed_air.rewards.to_dict()
    assert speed_rewards.pop("feet_air_time") is None
    assert speed_air_rewards.pop("feet_air_time")["weight"] == 0.2
    assert speed_rewards == speed_air_rewards
    assert speed.commands.to_dict() == speed_air.commands.to_dict()
