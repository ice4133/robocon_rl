# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Validate the prepared CAD asset and the task's coordinate conventions."""

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from isaaclab.utils.string import resolve_matching_names_values

from robot_lab.assets.ice import ICE_ROBOCON_CFG, ROBOCON_ASSET_DIR


@pytest.fixture
def robot_description():
    return ET.parse(ICE_ROBOCON_CFG.spawn.asset_path).getroot()


def test_robocon_export_has_usable_meshes_inertias_and_joint_limits(robot_description):
    """Catch missing package paths and the all-zero limits in the original CAD export."""
    for mesh in robot_description.findall(".//mesh"):
        relative_path = mesh.attrib["filename"].removeprefix("package://robocon/")
        assert (ROBOCON_ASSET_DIR / relative_path).is_file()

    for link in robot_description.findall("link"):
        assert float(link.find("inertial/mass").attrib["value"]) > 0
        inertia = link.find("inertial/inertia").attrib
        tensor = np.array(
            [
                [float(inertia["ixx"]), float(inertia["ixy"]), float(inertia["ixz"])],
                [float(inertia["ixy"]), float(inertia["iyy"]), float(inertia["iyz"])],
                [float(inertia["ixz"]), float(inertia["iyz"]), float(inertia["izz"])],
            ]
        )
        principal = np.linalg.eigvalsh(tensor)
        assert principal[0] > 0
        assert principal[2] <= principal[0] + principal[1]

    joints = robot_description.findall("joint")
    names = [joint.attrib["name"] for joint in joints]
    _, matched_names, values = resolve_matching_names_values(ICE_ROBOCON_CFG.init_state.joint_pos, names)
    default_positions = dict(zip(matched_names, values, strict=True))
    assert len(joints) == 16
    for joint in joints:
        limit = joint.find("limit").attrib
        assert float(limit["effort"]) > 0
        assert float(limit["velocity"]) > 0
        actuator = ICE_ROBOCON_CFG.actuators["wheels" if joint.attrib["type"] == "continuous" else "legs"]
        assert float(limit["effort"]) == actuator.joint_effort_limit
        assert float(limit["velocity"]) == actuator.joint_velocity_limit
        if joint.attrib["type"] == "revolute":
            assert float(limit["lower"]) < default_positions[joint.attrib["name"]] < float(limit["upper"])


@pytest.mark.parametrize("terrain", ["Flat", "Rough"])
def test_robocon_task_matches_the_exported_actuators(robot_description, terrain):
    from isaaclab_tasks.utils import resolve_task_config

    env, agent = resolve_task_config(
        f"RobotLab-Isaac-Velocity-{terrain}-ICE-Robocon-v0", "rsl_rl_cfg_entry_point", overrides=[]
    )
    joint_names = {joint.attrib["name"] for joint in robot_description.findall("joint")}
    action_names = env.actions.joint_pos.joint_names + env.actions.joint_vel.joint_names
    assert len(action_names) == len(set(action_names)) == 16
    assert set(action_names) == joint_names
    assert env.observations.policy.joint_vel.params["asset_cfg"].joint_names == action_names
    assert env.observations.critic.joint_vel.params["asset_cfg"].joint_names == action_names
    assert env.scene.robot.spawn.asset_path == ICE_ROBOCON_CFG.spawn.asset_path
    assert agent.experiment_name == f"ice_robocon_{terrain.lower()}"
    assert env.rewards.joint_mirror is None

    for name in joint_names:
        matches = [
            actuator
            for actuator in env.scene.robot.actuators.values()
            if any(re.fullmatch(pattern, name) for pattern in actuator.joint_names_expr)
        ]
        assert len(matches) == 1


def test_robocon_stance_clears_the_ground_and_wheels_share_forward_direction(robot_description):
    """Check the actual CAD kinematics; copied Go2W angles/signs fail this check."""
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.ice_robocon.flat_env_cfg import (
        IceRoboconFlatEnvCfg,
    )

    cfg = IceRoboconFlatEnvCfg()
    joints = robot_description.findall("joint")
    names = [joint.attrib["name"] for joint in joints]
    _, matched_names, values = resolve_matching_names_values(cfg.scene.robot.init_state.joint_pos, names)
    positions = dict(zip(matched_names, values, strict=True))
    _, matched_names, values = resolve_matching_names_values(cfg.actions.joint_vel.scale, cfg.wheel_joint_names)
    wheel_scale = dict(zip(matched_names, values, strict=True))

    transforms = {"base_link": np.eye(4)}
    transforms["base_link"][:3, 3] = cfg.scene.robot.init_state.pos
    wheel_heights = []
    for joint in joints:
        origin = joint.find("origin").attrib
        local = np.eye(4)
        local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin["rpy"], sep=" ")).as_matrix()
        local[:3, 3] = np.fromstring(origin["xyz"], sep=" ")
        joint_transform = transforms[joint.find("parent").attrib["link"]] @ local
        axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
        name = joint.attrib["name"]
        if name in wheel_scale:
            angular_direction = joint_transform[:3, :3] @ axis * wheel_scale[name]
            assert angular_direction[1] > 0
            link = robot_description.find(f"link[@name='{joint.find('child').attrib['link']}']")
            radius = float(link.find("collision/geometry/cylinder").attrib["radius"])
            wheel_heights.append(joint_transform[2, 3] - radius)
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_rotvec(axis * positions[name]).as_matrix()
        transforms[joint.find("child").attrib["link"]] = joint_transform @ rotation

    assert len(wheel_heights) == 4
    np.testing.assert_allclose(wheel_heights, wheel_heights[0], atol=1e-6)
    assert 0 < min(wheel_heights) < 0.02
