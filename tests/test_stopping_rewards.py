# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Verify stopping costs distinguish stationary commands from slow driving/turning."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import (
    stand_base_motion_l2,
    stand_wheel_action_l2,
    stand_wheel_velocity_l2,
)


def make_env(commands, linear=None, angular=None, joint_velocity=None, wheel_actions=None):
    commands = torch.tensor(commands, dtype=torch.float32)
    count = commands.shape[0]
    data = SimpleNamespace(
        root_link_lin_vel_b=SimpleNamespace(torch=torch.zeros(count, 3) if linear is None else torch.tensor(linear)),
        root_link_ang_vel_b=SimpleNamespace(torch=torch.zeros(count, 3) if angular is None else torch.tensor(angular)),
        joint_vel=SimpleNamespace(
            torch=torch.zeros(count, 7) if joint_velocity is None else torch.tensor(joint_velocity)
        ),
    )
    velocity_action = SimpleNamespace(
        raw_actions=torch.zeros(count, 4) if wheel_actions is None else torch.tensor(wheel_actions)
    )
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: commands),
        action_manager=SimpleNamespace(get_term=lambda name: {"joint_vel": velocity_action}[name]),
    )


def test_base_cost_brakes_a_moving_robot_but_does_not_suppress_small_commands():
    # A stop must be penalized even above the old wheel penalty's speed threshold.
    env = make_env(
        [[0, 0, 0], [0.0001, 0, 0], [0, -0.0001, 0], [0, 0, 0.0001]],
        linear=[[0.8, -0.6, 0.0]] * 4,
        angular=[[0.0, 0.0, 2.0]] * 4,
    )
    torch.testing.assert_close(stand_base_motion_l2(env, "base_velocity"), torch.tensor([2.0, 0.0, 0.0, 0.0]))


def test_balance_adjustments_and_exact_rest_have_no_planar_stopping_cost():
    env = make_env(
        [[0, 0, 0], [0, 0, 0]],
        linear=[[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
        angular=[[1.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
    )
    torch.testing.assert_close(stand_base_motion_l2(env, "base_velocity"), torch.zeros(2))


def test_wheel_cost_uses_selected_actual_joints_without_a_contact_sensor():
    # Leg speeds must not enter the cost; wheel DOFs are deliberately interleaved.
    env = make_env(
        [[0, 0, 0], [0, 0, 0.1]],
        joint_velocity=[[1.0, 100.0, -2.0, 100.0, 3.0, 100.0, -4.0]] * 2,
    )
    cost = stand_wheel_velocity_l2(env, "base_velocity", SceneEntityCfg("robot", joint_ids=[0, 2, 4, 6]))
    torch.testing.assert_close(cost, torch.tensor([7.5, 0.0]))


def test_opposing_wheel_targets_are_penalized_even_if_actual_wheels_are_stopped():
    env = make_env([[0, 0, 0], [0.1, 0, 0]], wheel_actions=[[1.0, -1.0, 1.0, -1.0]] * 2)
    torch.testing.assert_close(stand_wheel_action_l2(env, "base_velocity"), torch.tensor([1.0, 0.0]))
    actual = stand_wheel_velocity_l2(env, "base_velocity", SceneEntityCfg("robot", joint_ids=[0, 2, 4, 6]))
    torch.testing.assert_close(actual, torch.zeros(2))


@pytest.mark.parametrize("terrain", ["Flat", "Rough"])
def test_stopping_rewards_are_disabled_for_the_baseline(terrain):
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config(
        f"RobotLab-Isaac-Velocity-{terrain}-ICE-Robocon-v0", "rsl_rl_cfg_entry_point", overrides=[]
    )
    for name in ("stand_base_motion", "stand_wheel_velocity", "stand_wheel_action"):
        assert getattr(cfg.rewards, name) is None
    assert cfg.commands.base_velocity.rel_standing_envs == (0.05 if terrain == "Flat" else 0.25)
    assert cfg.commands.base_velocity.resampling_time_range == ((4.0, 7.0) if terrain == "Flat" else (3.0, 6.0))
    assert cfg.observations.policy.base_lin_vel is None
    assert cfg.actions.joint_pos.joint_names + cfg.actions.joint_vel.joint_names == cfg.joint_names

    go2w, _ = resolve_task_config(
        f"RobotLab-Isaac-Velocity-{terrain}-Unitree-Go2W-v0", "rsl_rl_cfg_entry_point", overrides=[]
    )
    assert not hasattr(go2w.rewards, "stand_base_motion")
