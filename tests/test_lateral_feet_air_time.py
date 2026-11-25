# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Check that the strafe swing incentive stays bounded and only scores landings."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import lateral_feet_air_time


class Scene(dict):
    @property
    def sensors(self):
        return self


@pytest.mark.parametrize(
    "lateral_speed,air_time,landing,upright,expected",
    [
        (0.2, 0.10, True, True, -0.8),
        (0.2, 0.40, True, True, 0.4),
        (-0.2, 0.40, True, True, 0.4),
        (0.2, 0.50, True, True, 0.8),
        (0.2, 5.00, True, True, 0.8),
        (0.0, 0.40, True, True, 0.0),
        (0.1, 0.40, True, True, 0.0),
        (0.2, 0.40, False, True, 0.0),
        (0.2, 0.40, True, False, 0.0),
    ],
)
def test_lateral_swing_reward(lateral_speed, air_time, landing, upright, expected):
    # The fifth body is not a foot and must not contribute to the landing score.
    sensor = SimpleNamespace(
        compute_first_contact=lambda dt: SimpleNamespace(torch=torch.full((1, 5), landing)),
        data=SimpleNamespace(last_air_time=SimpleNamespace(torch=torch.tensor([[air_time] * 4 + [10.0]]))),
    )
    env = SimpleNamespace(
        step_dt=0.02,
        scene=Scene(
            {
                "contact_forces": sensor,
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -float(upright)]]))
                    )
                ),
            }
        ),
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[2.0, lateral_speed, 0.0]])),
    )
    reward = lateral_feet_air_time(
        env,
        command_name="base_velocity",
        sensor_cfg=SceneEntityCfg("contact_forces", body_ids=[0, 1, 2, 3]),
        threshold=0.30,
        max_air_time=0.50,
        command_threshold=0.10,
    )
    assert reward.item() == pytest.approx(expected)
