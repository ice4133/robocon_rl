# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ICE laboratory robots.

Robocon geometry, mass, inertia and joint axes come from the supplied SolidWorks
export. Legs use Unitree IM6014 and wheels use GO-M8010-6. Effort/speed limits use
the manufacturer's output-side specifications at 60 V and 24 V, respectively,
assuming no additional external transmission. Peak torque and no-load speed are
reference bounds, not a measured torque-speed or thermal model. Mechanical joint
limits, PD gains and the stance remain simulation starting values.
"""

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from robot_lab.assets import ROBOT_LAB_ASSETS_DIR

ROBOCON_ASSET_DIR = ROBOT_LAB_ASSETS_DIR / "Robots" / "ice" / "robocon"

# Output-side bounds; the motors' internal gear ratios are already included.
# IM6014 (N6014B-12.6), reference 60 V: use the motoring peak, not the 34.4 N m braking peak.
# https://www.unitree.com/mobile/IM6014/
ROBOCON_LEG_EFFORT_LIMIT = 31.7  # N m, peak
ROBOCON_LEG_VELOCITY_LIMIT = 54.2  # rad/s, no load at 60 V
# GO-M8010-6, reference 24 V.
# https://www.unitree.com/mobile/go1/motor/
ROBOCON_WHEEL_EFFORT_LIMIT = 23.7  # N m, peak
ROBOCON_WHEEL_VELOCITY_LIMIT = 30.0  # rad/s at 24 V
ROBOCON_WHEEL_RADIUS = 0.088  # m, measured from the exported wheel meshes

# The CAD export already includes the bent-leg geometry in the link frames.
# Left/right pitch axes have opposite signs, so Go2W's raw angles cannot be reused.
# This symmetric pose puts the wheel centers about 0.322 m below base_link.
ROBOCON_DEFAULT_JOINT_POS = {
    ".*_hip_joint": 0.0,
    "[fr]l_thigh_joint": 0.2,
    "[fr]r_thigh_joint": -0.2,
    "[fr]l_calf_joint": -0.4,
    "[fr]r_calf_joint": 0.4,
    ".*_foot_joint": 0.0,
}

ICE_ROBOCON_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ROBOCON_ASSET_DIR / "urdf" / "robocon.urdf"),
        ros_package_paths=[{"name": "robocon", "path": str(ROBOCON_ASSET_DIR)}],
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos=ROBOCON_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            joint_effort_limit=ROBOCON_LEG_EFFORT_LIMIT,
            joint_velocity_limit=ROBOCON_LEG_VELOCITY_LIMIT,
            stiffness=60.0,
            damping=2.0,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            joint_effort_limit=ROBOCON_WHEEL_EFFORT_LIMIT,
            joint_velocity_limit=ROBOCON_WHEEL_VELOCITY_LIMIT,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
