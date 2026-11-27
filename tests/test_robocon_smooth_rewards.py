# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Check target history/reset handling and command-aware airborne spin costs."""

from types import SimpleNamespace

import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.ice_robocon.smooth_rewards import (
    JointTargetSmoothness,
    airborne_wheel_overspeed,
    straight_rolling_l2,
)


def target_term(count=2, dt=0.02):
    position = SimpleNamespace(processed_actions=torch.zeros(count, 3), raw_actions=torch.zeros(count, 3))
    velocity = SimpleNamespace(processed_actions=torch.zeros(count, 4), raw_actions=torch.zeros(count, 4))
    env = SimpleNamespace(
        num_envs=count, device="cpu", step_dt=dt,
        action_manager=SimpleNamespace(get_term=lambda name: {"joint_pos": position, "joint_vel": velocity}[name]),
    )
    params = dict(position_action_name="joint_pos", velocity_action_name="joint_vel",
                  leg_second_difference_scale=0.05, wheel_second_difference_scale=1.0)
    term = JointTargetSmoothness(RewardTermCfg(func=JointTargetSmoothness, weight=-0.01, params=params), env)
    return env, position, velocity, term, params


def test_constant_target_velocity_is_free_but_an_abrupt_target_turn_is_penalized():
    env, position, velocity, term, params = target_term()
    # Large initial offsets are excluded, and a linear ramp has no curvature.
    for step in range(5):
        position.processed_actions[:] = 3.0 + 0.05 * step
        velocity.processed_actions[:] = -5.0 + step
        torch.testing.assert_close(term(env, **params), torch.zeros(2), atol=1e-8, rtol=0)
    position.processed_actions += 0.15  # acceleration in target space
    velocity.processed_actions += 3.0
    torch.testing.assert_close(term(env, **params), torch.full((2,), 8.0), atol=1e-4, rtol=0)


def test_partial_reset_excludes_two_samples_without_erasing_other_env_history():
    env, position, _, term, params = target_term()
    term(env, **params)
    term(env, **params)
    term.reset(torch.tensor([1]))
    position.processed_actions[:] = 1.0
    cost = term(env, **params)
    assert cost[0] > 0 and cost[1] == 0
    assert term(env, **params)[1] == 0
    position.processed_actions[1] = -1.0
    assert term(env, **params)[1] > 0
    term.reset()
    torch.testing.assert_close(term(env, **params), torch.zeros(2))


def test_raw_outputs_beyond_clipping_do_not_change_the_target_curvature_cost():
    env, position, velocity, term, params = target_term()
    position.processed_actions[:] = 0.675
    velocity.processed_actions[:] = 30
    for raw in (10, 100, 200, 500):
        position.raw_actions[:] = raw
        velocity.raw_actions[:] = raw
        torch.testing.assert_close(term(env, **params), torch.zeros(2))


def test_same_physical_target_acceleration_has_same_cost_at_different_timesteps():
    costs = []
    for dt in (0.02, 0.04):
        env, position, _, term, params = target_term(dt=dt)
        for step in range(4):
            position.processed_actions[:] = 10 * (step * dt) ** 2
            cost = term(env, **params)
        costs.append(cost)
    torch.testing.assert_close(costs[0], costs[1])


def spin_cost(commands, speed, air):
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={"contact": SimpleNamespace(data=SimpleNamespace(
                current_air_time=SimpleNamespace(torch=torch.tensor(air, dtype=torch.float32))))},
        ),
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor(commands, dtype=torch.float32)),
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=torch.tensor(speed, dtype=torch.float32))))
    class Scene:
        sensors = env.scene.sensors
        def __getitem__(self, name):
            return robot
    env.scene = Scene()
    return airborne_wheel_overspeed(
        env, "velocity", SceneEntityCfg("contact", body_ids=[1, 0]),
        SceneEntityCfg("robot", joint_ids=[2, 0]),
        wheel_radius=0.088, turn_radius=0.4, speed_margin=2.0, air_time_threshold=0.08, speed_scale=30.0,
    )


def test_airborne_overspeed_ignores_grounded_wheels_short_swings_and_leg_joints():
    result = spin_cost(
        [[0, 0, 0]] * 3,
        [[30, 10000, -30]] * 3,
        [[0, 0], [0.05, 0.05], [0.1, 0.1]],
    )
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, (28 / 30) ** 2]))


def test_airborne_speed_allowance_preserves_forward_and_turning_wheel_motion():
    result = spin_cost(
        [[2, 0, 0], [0, 0, 0.8], [0, 0.2, 0]],
        [[22, 10000, -22], [5, 10000, -5], [30, 10000, -30]],
        [[0.2, 0.2]] * 3,
    )
    assert result[0] == 0 and result[1] == 0 and result[2] > 0


def straight_env(commands, speed):
    command = torch.tensor(commands, dtype=torch.float32)
    count = len(commands)
    linear = torch.zeros(count, 3)
    linear[:, 0] = torch.tensor(speed)
    angular = torch.zeros(count, 3)
    velocity = torch.zeros(count, 16)
    wheel_ids = [0, 4, 8, 12]  # Deliberately interleaved with the legs.
    leg_ids = [i for i in range(16) if i not in wheel_ids]
    velocity[:, wheel_ids] = linear[:, :1] / .088 * torch.tensor([-1, 1, -1, 1])
    data = SimpleNamespace(joint_vel=SimpleNamespace(torch=velocity), root_lin_vel_b=SimpleNamespace(torch=linear),
                           root_ang_vel_b=SimpleNamespace(torch=angular))
    env = SimpleNamespace(num_envs=count, scene={"robot": SimpleNamespace(data=data)},
                          command_manager=SimpleNamespace(get_command=lambda name: command))
    params = dict(command_name="velocity", wheel_asset_cfg=SceneEntityCfg("robot", joint_ids=wheel_ids),
                  leg_asset_cfg=SceneEntityCfg("robot", joint_ids=leg_ids), wheel_direction_signs=(-1, 1, -1, 1),
                  wheel_radius=.088, wheel_speed_error_scale=.25, lateral_speed_scale=.1, yaw_rate_scale=.05,
                  leg_speed_scale=.5, leg_speed_deadband=.05, forward_command_threshold=.05,
                  other_command_threshold=1e-6)
    return env, data, params


def test_straight_rolling_respects_opposite_urdf_axes_and_reverse_motion():
    env, data, params = straight_env([[2, 0, 0], [-2, 0, 0]], [2, -2])
    # Tiny balance adjustments remain inside the leg-velocity deadband.
    data.joint_vel.torch[:, params['leg_asset_cfg'].joint_ids] = .03
    torch.testing.assert_close(straight_rolling_l2(env, **params), torch.zeros(2))


def test_single_slow_front_left_wheel_and_restless_leg_cannot_hide_in_the_average():
    env, data, params = straight_env([[-2, 0, 0], [-2, 0, 0]], [-2, -2])
    data.joint_vel.torch[0, 4] = -1 / .088  # FL rolls at -1 instead of -2 m/s.
    data.joint_vel.torch[1, 5] = .25  # One FL leg joint moves while the others rest.
    cost = straight_rolling_l2(env, **params)
    torch.testing.assert_close(cost[0], torch.tensor(4.0))
    assert cost[1] > 0


def test_straight_cost_detects_yaw_and_lateral_drift():
    env, data, params = straight_env([[1, 0, 0]], [1])
    data.root_ang_vel_b.torch[0, 2] = .01
    data.root_lin_vel_b.torch[0, 1] = .03
    torch.testing.assert_close(straight_rolling_l2(env, **params), torch.tensor([.13]))


def test_lateral_yaw_mixed_and_tiny_nonzero_turn_commands_are_exempt():
    commands = [[0, 0, 0], [0, .2, 0], [0, 0, .6], [.5, .2, .4], [1, .001, 0], [1, 0, .001]]
    env, data, params = straight_env(commands, [1] * len(commands))
    data.joint_vel.torch[:] = 10
    data.root_ang_vel_b.torch[:] = 1
    torch.testing.assert_close(straight_rolling_l2(env, **params), torch.zeros(len(commands)))


def test_acceleration_is_not_penalized_for_lagging_the_command():
    env, _, params = straight_env([[2, 0, 0]], [0])
    torch.testing.assert_close(straight_rolling_l2(env, **params), torch.zeros(1))
