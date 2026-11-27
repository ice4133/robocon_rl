# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Robocon wheel-leg velocity tracking, using the Go2W task's MDP terms."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass, replace

from robot_lab.assets.ice import ICE_ROBOCON_CFG
from robot_lab.tasks.manager_based.locomotion.velocity import mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRewardsCfg,
    UnitreeGo2WRoughEnvCfg,
)


@configclass
class IceRoboconRewardsCfg(UnitreeGo2WRewardsCfg):
    """Optional stopping costs, disabled for the balanced-command baseline."""

    stand_base_motion = RewTerm(
        func=mdp.stand_base_motion_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "yaw_weight": 0.25, "command_threshold": 1.0e-6},
    )
    stand_wheel_velocity = RewTerm(
        func=mdp.stand_wheel_velocity_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            "command_threshold": 1.0e-6,
        },
    )
    stand_wheel_action = RewTerm(
        func=mdp.stand_wheel_action_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "action_name": "joint_vel", "command_threshold": 1.0e-6},
    )


@configclass
class IceRoboconRoughEnvCfg(UnitreeGo2WRoughEnvCfg):
    rewards: IceRoboconRewardsCfg = IceRoboconRewardsCfg()
    base_link_name = "base_link"
    foot_link_name = ".*_foot_link"

    # Keep action and observation order explicit, independently of the CAD tree order.
    leg_joint_names = [
        "fr_hip_joint",
        "fr_thigh_joint",
        "fr_calf_joint",
        "fl_hip_joint",
        "fl_thigh_joint",
        "fl_calf_joint",
        "rr_hip_joint",
        "rr_thigh_joint",
        "rr_calf_joint",
        "rl_hip_joint",
        "rl_thigh_joint",
        "rl_calf_joint",
    ]
    wheel_joint_names = ["fr_foot_joint", "fl_foot_joint", "rr_foot_joint", "rl_foot_joint"]
    joint_names = leg_joint_names + wheel_joint_names

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = replace(ICE_ROBOCON_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 1024
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.6
        self.scene.terrain.physics_material.restitution = 0.0
        self.sim.physics_material = self.scene.terrain.physics_material

        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # Positive wheel actions drive every wheel forward despite opposite URDF axes.
        self.actions.joint_vel.scale = {"[fr]l_foot_joint": 5.0, "[fr]r_foot_joint": -5.0}
        # JointAction clips the final position/velocity targets, not the raw policy output.
        self.actions.joint_pos.clip = {
            ".*_hip_joint": (-0.675, 0.675),
            "[fr]l_thigh_joint": (-0.875, 1.375),
            "[fr]r_thigh_joint": (-1.375, 0.875),
            "[fr]l_calf_joint": (-1.395, 0.495),
            "[fr]r_calf_joint": (-0.495, 1.395),
        }
        self.actions.joint_vel.clip = {".*": (-30.0, 30.0)}

        # Start upright; the source Go2W task samples completely inverted resets.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.03),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-3.14, 3.14),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (0.0, 0.0),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.5, 1.0)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.9, 1.1)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-2.0, 2.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-1.0, 1.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}

        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        # Resample stand/move for each environment during an episode, so the
        # policy learns to brake after driving instead of only standing at reset.
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.resampling_time_range = (3.0, 6.0)

        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.base_height_l2.params["target_height"] = 0.38
        # Opposite pitch axes make the Go2W raw-angle equality penalty inappropriate.
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("fl_foot_link", "rr_foot_link"),
            ("fr_foot_link", "rl_foot_link"),
        )
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})

        if self.__class__.__name__ == "IceRoboconRoughEnvCfg":
            self.disable_zero_weight_rewards()
