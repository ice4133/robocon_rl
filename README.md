# robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.2%20(optional)-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**robot_lab** is a RL extension library for robots, based on IsaacLab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

The table below lists representative environments. Use `isaaclab list_envs` for all task IDs and agent configurations:

| Category   | Robot Model         | Environment Name (<ENV_NAME>)                                      | Screenshot |
|------------|---------------------|------------------------------------------------------------|------------|
| **Quadruped** | [Anymal D](https://www.anybotics.com/robotics/anymal) | Isaac-Velocity-Flat-Anymal-D-v0 | <img src="./docs/imgs/anymal_d.png" alt="anymal_d" width="75"> |
|            | [Unitree Go2](https://www.unitree.com/go2) | RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 | <img src="./docs/imgs/unitree_go2.png" alt="unitree_go2" width="75"> |
|            | [Unitree B2](https://www.unitree.com/b2) | RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0 | <img src="./docs/imgs/unitree_b2.png" alt="unitree_b2" width="75"> |
|            | [Unitree A1](https://www.unitree.com/a1) | RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 | <img src="./docs/imgs/unitree_a1.png" alt="unitree_a1" width="75"> |
|            | [Deeprobotics Lite3](https://www.deeprobotics.cn/robot/index/product1.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0 | <img src="./docs/imgs/deeprobotics_lite3.png" alt="Lite3" width="75"> |
|            | [Zsibot ZSL1](https://www.zsibot.com/zsl1) | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1-v0 | <img src="./docs/imgs/zsibot_zsl1.png" alt="zsibot_zsl1" width="75"> |
|            | [Magiclab MagicDog](https://www.magiclab.top/dog) | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-v0 | <img src="./docs/imgs/magiclab_magicdog.png" alt="magiclab_magicdog" width="75"> |
|            | [Agibot D1](https://www.agibot.com/) | RobotLab-Isaac-Velocity-Rough-Agibot-D1-v0 | <img src="./docs/imgs/agibot_d1.png" alt="magiclab_magicdog" width="75">  |
| **Wheeled** | [Unitree Go2W](https://www.unitree.com/go2-w) | RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 | <img src="./docs/imgs/unitree_go2w.png" alt="unitree_go2w" width="75"> |
|            | [Unitree B2W](https://www.unitree.com/b2-w) | RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0 | <img src="./docs/imgs/unitree_b2w.png" alt="unitree_b2w" width="75"> |
|            | [Deeprobotics M20](https://www.deeprobotics.cn/robot/index/lynx.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0 | <img src="./docs/imgs/deeprobotics_m20.png" alt="deeprobotics_m20" width="75"> |
|            | [DDTRobot Tita](https://directdrive.com/product_TITA) | RobotLab-Isaac-Velocity-Rough-DDTRobot-Tita-v0 | <img src="./docs/imgs/ddtrobot_tita.png" alt="ddtrobot_tita" width="75"> |
|            | [Zsibot ZSL1W](https://www.zsibot.com/zsl1) | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1W-v0 | <img src="./docs/imgs/zsibot_zsl1w.png" alt="zsibot_zsl1w" width="75"> |
|            | [Magiclab MagicDog-W](https://www.magiclab.top/dog-w) | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-W-v0 | <img src="./docs/imgs/magiclab_magicdog_w.png" alt="magiclab_magicdog_w" width="75"> |
| **Humanoid** | [Unitree G1](https://www.unitree.com/g1) | RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 | <img src="./docs/imgs/unitree_g1.png" alt="unitree_g1" width="75"> |
|             | [Unitree H1](https://www.unitree.com/h1) | RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0 | <img src="./docs/imgs/unitree_h1.png" alt="unitree_h1" width="75"> |
|             | [FFTAI GR1T1](https://www.fftai.com/products-gr1) | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0 | <img src="./docs/imgs/fftai_gr1t1.png" alt="fftai_gr1t1" width="75"> |
|             | [FFTAI GR1T2](https://www.fftai.com/products-gr1) | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T2-v0 | <img src="./docs/imgs/fftai_gr1t2.png" alt="fftai_gr1t2" width="75"> |
|             | [Booster T1](https://www.boosterobotics.com/) | RobotLab-Isaac-Velocity-Rough-Booster-T1-v0 | <img src="./docs/imgs/booster_t1.png" alt="booster_t1" width="75"> |
|             | [RobotEra Xbot](https://www.robotera.com/) | RobotLab-Isaac-Velocity-Rough-RobotEra-Xbot-v0 | <img src="./docs/imgs/robotera_xbot.png" alt="robotera_xbot" width="75"> |
|             | [Openloong Loong](https://www.openloong.net/) | RobotLab-Isaac-Velocity-Rough-Openloong-Loong-v0 | <img src="./docs/imgs/openloong_loong.png" alt="openloong_loong" width="75"> |
|             | [RoboParty ATOM01](https://roboparty.cn/) | RobotLab-Isaac-Velocity-Rough-RoboParty-ATOM01-v0 | <img src="./docs/imgs/roboparty_atom01.png" alt="roboparty_atom01" width="75"> |
|             | [Magiclab MagicBot-Gen1](https://www.magiclab.top/human) | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Gen1-v0 | <img src="./docs/imgs/magiclab_magicbot_gen1.png" alt="magiclab_magicbot_gen1" width="75"> |
|             | [Magiclab MagicBot-Z1](https://www.magiclab.top/z1) | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Z1-v0 | <img src="./docs/imgs/magiclab_magicbot_z1.png" alt="magiclab_magicbot_z1" width="75"> |

> [!NOTE]
> If you want to run policy in gazebo or real robot, please use [rl_sar](https://github.com/fan-ziqi/rl_sar) project.
>
> Discuss in [Github Discussion](https://github.com/fan-ziqi/robot_lab/discussions) or [Discord](http://www.robotsfan.com/dc_robot_lab).

## Version Dependency

| robot_lab Version | Isaac Lab Version             | Isaac Sim Version         |
|------------------ | ----------------------------- | ------------------------- |
| `main` branch     | `v3.0.0`                      | Isaac Sim 6.2 (optional for Newton) |
| `v2.3.2`          | `v2.3.2`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.2`          | `v2.2.1`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.1`          | `v2.1.1`                      | Isaac Sim 4.5             |
| `v1.1`            | `v1.4.1`                      | Isaac Sim 4.2             |

## Installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). This source project follows Isaac Lab's
external-project template and expects an Isaac Lab **v3.0.0** checkout beside it:

```text
workspaces/
├── IsaacLab/     # v3.0.0
└── robot_lab/
```

For a new workspace:

```bash
git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git
git clone https://github.com/fan-ziqi/robot_lab.git
cd robot_lab
uv sync --extra importers
uv run --extra importers isaaclab list_envs --show_presets
```

If the checkouts already exist, verify the Isaac Lab version and run `uv sync --extra importers`
from this repository. Relative source paths are declared in `pyproject.toml`; update them if the
Isaac Lab directory moves. Commit `uv.lock` with dependency changes.

For an existing, fully provisioned Isaac Lab environment, install only this extension:

```bash
uv pip install --no-deps -e .
uv run --active --no-project isaaclab list_envs
```

### Newton without Isaac Sim

Velocity tasks use Isaac Lab's `RoughPhysicsCfg` and default to Newton MJWarp. The `importers`
extra provides URDF/MJCF conversion without installing the full Isaac Sim runtime or Kit.

```bash
uv run --extra importers isaaclab zero_agent --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4 --max_steps 100 --viz none physics=newton_mjwarp
uv run --extra importers isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --viz none physics=newton_mjwarp
uv run --extra importers isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint latest --num_envs 4 --viz newton_gl physics=newton_mjwarp
```

The standalone importer retains the namespace `isaacsim.asset.importer`. It converts assets and
registers USD schemas without `SimulationApp`. Project initialization imports it before the first
USD stage is created when standalone importers are available.

### Isaac Sim PhysX

Use a separate environment for Isaac Sim. Its regular `isaacsim` package shadows the standalone
importer namespace, so URDF import in that environment requires Kit.

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv sync --extra isaacsim
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim isaaclab zero_agent --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4 --max_steps 100 --viz none physics=isaacsim_physx
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --viz none physics=isaacsim_physx
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint latest --num_envs 4 --viz kit physics=isaacsim_physx
```

When Newton runs in the full Isaac Sim environment, add `--viz kit` for URDF import. USD assets can
run without Kit unless another selected component requires it. The AMP and BeyondMimic tasks
currently configure Isaac Sim PhysX directly.

## Train, play and record

Use the upstream CLI and select the RL library with `--rl_library`. RSL-RL is included; install
`--extra skrl` for tasks with a skrl agent configuration. Select a task and its available libraries
with `isaaclab list_envs`.

```bash
# Zero or random actions
uv run --extra importers isaaclab random_agent --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4 --max_steps 100 --viz none physics=newton_mjwarp

# Choose a checkpoint explicitly, or use latest/best
uv run --extra importers isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint /path/to/model.pt --viz newton_gl physics=newton_mjwarp

# Resume training
uv run --extra importers isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint /path/to/model.pt --viz none physics=newton_mjwarp

# Record a bounded playback through the selected visualizer
uv run --extra importers --extra video isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint latest --video --video_length 200 --viz newton_gl physics=newton_mjwarp

# Multiple GPUs
uv run --extra importers isaaclab train_multigpu --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_gpus 2 --viz none physics=newton_mjwarp

# Training logs
uv run tensorboard --logdir logs
```

The scripts under `scripts/reinforcement_learning/train.py`, `play.py`, and the existing `rsl_rl`
and `skrl` directories delegate to upstream entry points. The compatibility scripts still accept
`--headless`; new commands use `--viz none`.

### CusRL

CusRL is a project integration and uses its own scripts. Select its extra together with the runtime:

```bash
uv run --extra importers --extra cusrl python scripts/reinforcement_learning/cusrl/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --viz none physics=newton_mjwarp
uv run --extra importers --extra cusrl python scripts/reinforcement_learning/cusrl/play.py --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint /path/to/model.pt --viz newton_gl physics=newton_mjwarp
```

### Motion tracking and AMP

Acquire the motion data under its original license. Convert retargeted G1 CSV files for BeyondMimic
with the repository tool, then set the resulting archive in the environment configuration:

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim python scripts/tools/beyondmimic/csv_to_npz.py -f /path/to/motion.csv --input_fps 60 --viz none --exit_after_save
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim python scripts/tools/beyondmimic/replay_npz.py -f /path/to/motion.npz --viz kit
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --viz none env.commands.motion.motion_file=/path/to/motion.npz
```

For G1 AMP, use a motion archive containing joint names, body names, positions, rotations and
velocities. See the [motion format and conversion tools](source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/README.md).

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim --extra skrl isaaclab train --rl_library skrl --task RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --viz none
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim --extra skrl isaaclab play --rl_library skrl --task RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --checkpoint latest --num_envs 4 --viz kit
```

### Symmetry, distillation and handstand

ANYmal-D registers separate RSL-RL agent configurations for symmetry and distillation:

```bash
uv run --extra importers isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --agent rsl_rl_with_symmetry_cfg_entry_point --viz none physics=newton_mjwarp agent.algorithm.symmetry_cfg.use_data_augmentation=true
uv run --extra importers isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --agent rsl_rl_distillation_cfg_entry_point --checkpoint /path/to/teacher/model.pt --viz none physics=newton_mjwarp
uv run --extra importers isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0 --viz none physics=newton_mjwarp
```

For playback in a custom scene, use `scripts/reinforcement_learning/rsl_rl/play_cs.py` with
`--map /path/to/scene.usd`, the task, checkpoint and backend arguments.

## Add a robot or task

### ICE Robocon

The current straightness refinement starts from the best tracking checkpoint
`ice_robocon_smooth_from4999/2026-11-27_20-15-53_target_smooth_airspin_s42/model_7998.pt`:

```bash
cd /home/jasper/robot_lab
.venv/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-ICE-Robocon-Smooth-Straight-v0 \
  --num_envs 1024 --viz none --max_iterations 2000 \
  --checkpoint /home/jasper/robot_lab/logs/rsl_rl/ice_robocon_smooth_from4999/2026-11-27_20-15-53_target_smooth_airspin_s42/model_7998.pt \
  --no-resume_optimizer_state \
  --experiment_name ice_robocon_smooth_straight_from7998 \
  --run_name straight_s42 --seed 42 \
  physics=newton_mjwarp
```

The 2000 iterations are additional fine-tuning, with a fresh optimizer, fixed
learning rate 3e-5, entropy coefficient 0.001 and PPO clip 0.1. The previous Smooth
task is retained unchanged as a control. The refinement adds exactly one active
term (21 total), `straight_rolling`, at weight -0.2:

- Only pure forward/reverse commands with abs(vx)>0.05 and abs(vy),abs(wz)<=1e-6
  activate it. Lateral, yaw and mixed commands, including small nonzero yaw, are
  exempt.
- Wheel directions (-1,+1,-1,+1) match the URDF. Signed wheel surface speeds must
  agree with measured body vx (scale 0.25 m/s), independently of command-ramp lag.
- Body lateral speed (scale 0.10 m/s) and yaw rate (scale 0.05 rad/s) are penalized
  during straight rolling. These are normalization scales, not acceptance limits.
- Leg velocity beyond 0.05 rad/s is penalized with scale 0.5 rad/s; half the
  per-leg mean and half the worst-leg cost prevent a single restless leg from
  disappearing in a twelve-joint average. This treats every leg equally.

The existing target-curvature cost changes from -0.01 to -0.02. All other reward
weights, action scaling, joint/motor limits, 57-input/16-output policy structure,
and command ranges stay as in Smooth. Modes stand/forward/lateral/yaw/mixed are
5/35/25/15/20%; targets last 8–12 s, with 10% of forward/reverse samples in the
1.75–2 m/s magnitude band, covering both signs. Low-speed sampling remains 20%.
The sustained-airborne spin cost and existing lateral air-time reward remain on.

This addresses observed FL leg motion and rolling mismatch: in the previous
-2 m/s replay, FL leg velocity RMS was about 0.20 rad/s versus 0.02–0.03 in the
other legs; the FL rolling speed was about -1.24 m/s versus -1.94 in the body.
Mean body-frame velocity alone missed accumulated heading error. The companion
`rl_sar/scripts/check_ice_straight.py` measures native qpos trajectories relative
to the initial heading, for forward and reverse. Its initial acceptance limits
are 2 degrees maximum heading change and 0.10 m maximum cross-track deviation
over a 14 s command rollout. Existing 7998 fails these stricter path checks.
The reward is a soft constraint, not a deployment heading-lock controller;
the new checkpoint must demonstrate straightness and retain lateral/yaw/mixed
tracking before acceptance. Existing policies are not altered by this source edit.

The current smoothness experiment fine-tunes the selected original checkpoint
`ice_robocon_flat/2026-11-24_23-11-40/model_4999.pt`. Use the dedicated **Smooth**
task below; the older native-only coordination tasks remain available as controls.

```bash
cd /home/jasper/robot_lab
.venv/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-ICE-Robocon-Smooth-v0 \
  --num_envs 1024 --viz none --max_iterations 3000 \
  --checkpoint /home/jasper/robot_lab/logs/rsl_rl/ice_robocon_flat/2026-11-24_23-11-40/model_4999.pt \
  --no-resume_optimizer_state \
  --experiment_name ice_robocon_smooth_from4999 \
  --run_name target_smooth_airspin_s42 --seed 42 \
  physics=newton_mjwarp
```

This loads the original actor/critic and starts a fresh optimizer for the changed
objective. The 3000 iterations are **additional** training; the iteration counter
starts at the checkpoint's 4999. A fixed learning rate of 5e-5 prevents adaptive
learning-rate growth, PPO clip is 0.1, entropy coefficient is 0.002, and checkpoints
are saved every 100 iterations into a new experiment directory. Select by motion
evaluation, not by the largest checkpoint number. For an interruption of this
same experiment, resume from its checkpoint **with** `--resume_optimizer_state`.

The Smooth task restores 4999's native reward settings: XY/yaw tracking std 0.5,
weights 3.0/1.5, action-rate -0.03, leg acceleration -7.5e-7, joint posture -1.0
with stationary multiplier 5, and stand-still posture -2.0. Its pre-existing
custom lateral air-time term remains at weight 0.5, threshold 0.3 s and cap 0.5 s.
Command gates for posture/air-time change from 0.1 to 0.03 so small nonzero
commands are not treated as standing. The native touchdown-count reward changes
from 0.1 to disabled; it should not reward stationary contact flicker.

Exactly **two new costs** are enabled, giving 20 active terms in total:

- `joint_target_smoothness`, weight -0.01: second differences of the final clipped
  leg-position and wheel-velocity targets, normalized separately by 0.05 rad and
  1 rad/s per nominal 20 ms step. Uniform target ramps have zero curvature;
  abrupt target acceleration/reversal is penalized. The first two samples after
  each reset are excluded, independently for every environment.
- `airborne_wheel_overspeed`, weight -0.25: wheels airborne longer than 0.08 s are
  penalized for speed above `(abs(vx) + 0.4*abs(wz))/0.088 + 2` rad/s, normalized
  by 30 rad/s and averaged across wheels. Grounded wheels and short swings are
  exempt. This uses sustained air time, rather than the native wheel penalty's
  first-liftoff event. It is a conservative command-based allowance, not a
  complete wheel-slip model.

Forward/reverse stays ±2 m/s, lateral ±0.3 m/s, yaw ±0.8 rad/s, and mixed targets
stay within ±(0.5, 0.2, 0.4). Modes stand/forward/lateral/yaw/mixed have probabilities
10/25/25/20/20%, resampled every 6–9 s; 20% use low-speed ranges ±(0.3, 0.15, 0.3).
Training uses deployment-matched command ramps. Actuators, motor/joint limits,
action scales (0.18/0.25 for legs, signed 5 for wheels), and the 57-input/16-output
policy contract are unchanged. This is a fine-tuning candidate; smoothness and
tracking still need evaluation together, including low-speed, reverse, lateral,
yaw, all mixed signs and mode changes. A stationary or saturated failed action
must not be accepted just because its target differences are small.

The laboratory wheel-leg robot is packaged under
`source/robot_lab/robot_lab/assets/data/Robots/ice/robocon/`. Its Flat and Rough tasks
are registered as `RobotLab-Isaac-Velocity-{Flat,Rough}-ICE-Robocon-v0`, with separate
RSL-RL experiments `ice_robocon_flat` and `ice_robocon_rough`. Start on flat ground:

```bash
uv sync --locked --extra importers
uv run --extra importers isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-ICE-Robocon-v0 --num_envs 1024 --viz none physics=newton_mjwarp
```

The default Flat task is now a **coordination baseline**, trained from scratch.
It uses only existing robot_lab / Isaac Lab reward functions. The custom
`lateral_feet_air_time` and all three stopping costs are disabled. The previous
aggressive swing-duration and smoothness combination did not pass evaluation;
changing checkpoints alone is not treated as a solution.

Training is split into two explicitly registered tasks:

| Task suffix | Forward/reverse | Mixed vx / vy / wz | High-speed sampling |
|-------------|-----------------|--------------------|---------------------|
| `Robocon-v0` | ±1.5 m/s | ±0.5 / ±0.2 / ±0.4 | Disabled |
| `Robocon-Speed-v0` | ±2.5 m/s | ±1.5 / ±0.3 / ±0.6 | 10% of forward/reverse targets, magnitude 1.75–2.5 |

Both retain pure lateral commands ±0.3 m/s and pure yaw commands ±0.8 rad/s.
Mode probabilities are stand 5%, forward/reverse 20%, lateral 30%, turn 20%,
and mixed 25%. Commands resample every 4–7 seconds; 30% of targets use low-speed
ranges ±(0.3, 0.12, 0.25), in vx / vy / wz order. The remainder retains the full
pure-mode or mixed-mode ranges above. Acceleration limits are (1.0, 0.3, 0.8),
deceleration limits (1.5, 0.5, 1.2).
Reversals brake through zero and resets clear old commands. Hip/leg action scales
remain 0.18 / 0.25; motor, effort and joint limits are unchanged. Pure rolling at
3 m/s is outside the current 0.088 m * 30 rad/s = 2.64 m/s geometric bound, so the
confirmed final target remains ±2.5 m/s, subject to loaded-performance evaluation.

The revised native coordination baseline uses XY tracking weight 3.0 / std 0.35,
yaw tracking 1.5 / std 0.4, action-rate penalty -0.025, leg-acceleration penalty
-5e-7 and wheel-acceleration penalty -2.5e-9. Joint-posture weight is -0.4 with
standing multiplier 4.0, preserving the previous stationary effective weight
of -1.6 while allowing more leg motion under nonzero commands. Its command
threshold remains 0.03; stand-still posture is
-1.0 with threshold 0.03. `feet_contact_without_cmd` remains disabled because its
implementation counts new touchdowns rather than sustained support.

Run **stage 1 from scratch** without a checkpoint. Old `4999` is a possible
initialization control experiment, not a required starting point:

```bash
cd /home/jasper/robot_lab
.venv/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-ICE-Robocon-v0 \
  --num_envs 1024 --viz none --max_iterations 8000 \
  --experiment_name ice_robocon_native_coordination_v2 \
  --run_name scratch_s42 --seed 42 \
  physics=newton_mjwarp
```

Defaults for this stage are learning rate 3e-4 and entropy coefficient 0.005.
The previous scratch run `ice_robocon_native_coordination/2026-11-27_15-29-09_scratch_s42`
is retained as the v1 control. Its final checkpoint rolled smoothly but failed
lateral tracking in both Newton and `rl_sar`, and mixed-command evaluation in `rl_sar`.
This v2 changes native parameters and sampling together; it is a candidate
configuration, not an isolated causal ablation or a demonstrated gait improvement.
It adds no reward functions or active reward terms (17 remain enabled). The
earlier v1 settings can be reproduced with these overrides appended to the same
task, using a separate experiment name:

```bash
env.rewards.track_lin_vel_xy_exp.params.std=0.5 \
env.rewards.track_ang_vel_z_exp.params.std=0.5 \
env.rewards.action_rate_l2.weight=-0.02 \
env.rewards.joint_pos_penalty.weight=-0.8 \
env.rewards.joint_pos_penalty.params.stand_still_scale=2.0 \
'env.commands.base_velocity.mode_probabilities=[0.05,0.25,0.25,0.25,0.20]' \
env.commands.base_velocity.low_speed_fraction=0.20 \
'env.commands.base_velocity.low_speed_ranges.lin_vel_y=[-0.2,0.2]' \
'env.commands.base_velocity.low_speed_ranges.ang_vel_z=[-0.3,0.3]' \
agent.algorithm.entropy_coef=0.01
```

Before stage 2, evaluate both lateral directions, both yaw directions, mixed
commands, contact/swing timing, joint/wheel target variations and normal state
transitions in the original runtime and in `rl_sar`. Do not advance solely because
training finished, reward increased, or the final checkpoint has a larger number.
Choose a checkpoint that passes these checks, then expand speed:

```bash
coordination_checkpoint="/absolute/path/to/accepted/stage1/model_XXXX.pt"
.venv/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-ICE-Robocon-Speed-v0 \
  --num_envs 1024 --viz none --max_iterations 3000 \
  --checkpoint "$coordination_checkpoint" --no-resume_optimizer_state \
  --run_name speed25_s42 --seed 42 \
  physics=newton_mjwarp
```

Stage 2 defaults to learning rate 2e-4 and entropy coefficient 0.003. It changes
command coverage, not reward weights. It is a new training objective range, so a
fresh optimizer is selected; recovering an interrupted stage uses
`--resume_optimizer_state` with the same task and settings instead.

For an isolated swing-reward comparison, repeat stage 1 from scratch with the
same seed and arguments, changing only the task to
`RobotLab-Isaac-Velocity-Flat-ICE-Robocon-AirTime-v0`, the experiment name to
`ice_robocon_native_airtime02`, and the run name to `scratch_airtime02_s42`.
Its only reward difference is the existing lateral air-time term at weight 0.2,
threshold 0.10 s, cap 0.50 s and lateral command gate 0.03 m/s. All other reward
and command settings are identical. If this ablation is selected after evaluation,
use `RobotLab-Isaac-Velocity-Flat-ICE-Robocon-Speed-AirTime-v0` for stage 2 so the
same reward is retained. Do not claim longer strides just from a longer reward
threshold; measure the actual gait and tracking.

For MuJoCo validation, `rl_sar` now supports `mujoco_joint_drive: implicit`, matching
the training actuator's feedback formulation. It evaluates affine PD feedback at
each physics step and caps the combined force after feedback. Explicit mode remains
available for comparison and is the fallback for other robot configurations.
This removes the specific sampled-explicit/implicit drive discrepancy, not every
possible difference between Newton and native MuJoCo. Training completion alone
still does not establish acceptable motion quality.

For a server, place the same Isaac Lab v3.0.0 checkout next to `robot_lab`, as described
in Installation. Use `--num_envs 256` on a smaller GPU and increase it with available
memory. Preview a checkpoint with:

```bash
uv run --extra importers isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-ICE-Robocon-v0 --checkpoint latest --num_envs 4 --viz newton_gl physics=newton_mjwarp
```

For interactive single-robot evaluation in the Newton window:

```bash
.venv/bin/python scripts/reinforcement_learning/rsl_rl/play.py --keyboard --task RobotLab-Isaac-Velocity-Flat-ICE-Robocon-v0 --checkpoint latest --num_envs 1 --viz newton_gl physics=newton_mjwarp
```

Click the viewer and hold **I/K** to move forward/backward, **J/L** to turn
left/right, or **U/O** to move sideways. Releasing the keys commands zero velocity;
**X** also commands zero velocity while held. **B** resets the robot. The policy
still controls all joints and wheels. The camera follows the robot, and the terminal
prints commanded versus measured body velocities once per second. Defaults are
0.5 m/s forward/backward, 0.2 m/s sideways, and 0.6 rad/s turning; adjust with
`--speed`, `--strafe-speed`, and `--turn-speed`. Use `--no-follow-camera` for a free
camera, and Ctrl+C to exit.

The Newton window also shows a **Velocity tracking** panel with Target, Actual,
and signed Delta (`Actual - Target`) for forward/reverse speed `vx`, lateral speed
`vy` (both m/s), and turning rate `wz` (rad/s). Measurements use the robot's body
frame, so turning does not change the meaning of forward or sideways. Enable
**Show last 5 seconds** to overlay target and actual traces for each axis. History
clears on robot reset. Positive values mean forward, left, and left turn;
negative values mean reverse, right, and right turn.

The prepared URDF preserves the supplied CAD geometry, 23.47056 kg total mass,
inertias, link names and joint axes. Its original leg limits were all zero and its
wheel effort/speed limits were missing. The prepared copy supplies provisional
simulation limits, resolves the mesh package as `robocon`, names empty materials,
and replaces wheel collision meshes with smooth cylinders of radius 0.088 m and
width 0.030 m. The original uploaded export is left untouched.

Legs use Unitree IM6014 and wheels use GO-M8010-6. Their actuator and prepared URDF
limits use the following manufacturer output-side reference values, assuming no
additional external transmission:

| Actuator group | Motor | Reference voltage | Peak torque (N m) | Speed bound (rad/s) |
|----------------|-------|-------------------|-------------------|---------------------|
| Legs | IM6014 (N6014B-12.6) | 60 V | 31.7 | 54.2 (no load) |
| Wheels | GO-M8010-6 | 24 V | 23.7 | 30.0 |

References: [IM6014 specifications](https://www.unitree.com/mobile/IM6014/),
[GO-M8010-6 specifications](https://www.unitree.com/mobile/go1/motor/).
These peak/no-load bounds are a simulation approximation; the implicit PD actuators
do not model a measured torque-speed envelope, continuous thermal ratings or voltage
changes. Actual supply voltages and any external transmission remain to be confirmed.
Internal motor reduction is already included in these output-side specifications.
Motor limits, PD gains and the default stance are centralized in
`source/robot_lab/robot_lab/assets/ice.py`. Legs use position control (Kp 60, Kd 2),
and wheels use velocity control (Kp 0, Kd 0.5). The reference pose is:

| Joint group | Left (rad) | Right (rad) |
|-------------|------------|-------------|
| Hip         | 0.0        | 0.0         |
| Thigh       | 0.2        | -0.2        |
| Calf        | -0.4       | 0.4         |
| Wheel       | 0.0        | 0.0         |

This pose is derived from the CAD link frames, with `base_link` initially 0.42 m
above ground; it is not a measured hardware calibration. Left/right wheel action
signs follow the opposite URDF axes. Tasks start upright, bound action targets,
penalize loss of height/orientation, and terminate falls. Confirm the actual motor,
transmission, supply voltage, mechanical limits and encoder-to-URDF calibration
before using trained policies on hardware. Short training verifies the pipeline,
not policy convergence.

### Other robots and tasks

Tasks follow the Gymnasium API. Common velocity configuration and MDP terms belong to the task
family; robot-specific scenes, registrations and agent configurations belong to `config`:

```text
source/robot_lab/robot_lab/
├── assets/
│   ├── data/Robots/               # Packaged project-owned URDFs, meshes and textures
│   └── unitree.py                # Asset configuration
└── tasks/manager_based/locomotion/velocity/
    ├── velocity_env_cfg.py        # Shared task and upstream physics presets
    ├── mdp/                      # Project-specific terms and upstream lazy exports
    └── config/quadruped/unitree_a1/
        ├── __init__.py           # Gymnasium registrations
        ├── rough_env_cfg.py
        ├── flat_env_cfg.py
        └── agents/
```

Use `ROBOT_LAB_ASSETS_DIR` from `robot_lab.assets` for asset paths. It resolves correctly in an
editable checkout and an installed wheel. Register new tasks in their package's `__init__.py` with
module-path environment and agent entry points, following the existing registrations. The
`isaaclab.tasks` package entry point discovers them without copying launchers.

Use one mechanical asset for both backends and composable upstream schema configurations for
backend-specific properties. Keep `run_asset_transformer=True` and `run_multi_physics_conversion=True`
for URDF/MJCF import. Set solver capacities directly in the task's
`self.sim.physics.newton_mjwarp.solver_cfg`. Follow the official
[asset preparation guide](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/prepare_asset_for_newton.rst)
and [2.x to 3.0 migration guide](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/migrating.html).

Assets use each backend's native public joint/body ordering by default. For checkpoint transfer,
configure `ArticulationCfg.joint_ordering` and `body_ordering` through the upstream ordering API.
Checkpoints trained with the previous explicit ordering profiles or unmerged body layouts require
their original conventions. A1 velocity tasks randomize physical trunk/link masses while preserving the foot bodies.
G1 uses its physical pelvis as the root; velocity tasks merge fixed sensor frames.
New BeyondMimic archives store joint/body names and resolve motion data by name. Reconvert legacy
archives if their body layout differs, using an output FPS matching the task control rate.

Validate initial states, contacts, observations, rewards and repeated resets in both backends before
training. Smoke tests and short training do not establish policy convergence or transfer quality.

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
uv run --extra importers isaaclab --editor
```

The editor command selects the active interpreter and writes the local Pyright configuration. The
`setup_python_env` VS Code task invokes it. Select that interpreter in VS Code or Cursor; see the
upstream [editor setup](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/installation.html).
If ROS injects incompatible pytest plugins into the environment, run tests with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest`.

The optional `robot_lab.ui_extension_example` module is a Kit UI example; task discovery and
training do not load it.

## Docker

Build an Isaac Lab v3.0.0 base image using its Docker workflow, then set `ISAACLAB_BASE_IMAGE`
in `docker/.env.base` to that image. The extension image installs this project into the base
image's provisioned environment:

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build robot-lab
docker compose --env-file .env.base --file docker-compose.yaml run --rm robot-lab
# Inside the container, use the existing environment:
uv run --active --no-project isaaclab list_envs
```

## Citation

Please cite the following if you use this code or parts of it:

```
@software{fan-ziqi2024robot_lab,
  author = {Ziqi Fan},
  title = {robot_lab: RL Extension Library for Robots, Based on IsaacLab.},
  url = {https://github.com/fan-ziqi/robot_lab},
  year = {2024}
}
```

## Acknowledgements

The project uses some code from the following open-source code repositories:

- [linden713/humanoid_amp](https://github.com/linden713/humanoid_amp)
- [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
