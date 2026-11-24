# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Single-robot keyboard evaluation through the Newton visualizer's input API."""

from __future__ import annotations

import argparse
import contextlib
import os
import time

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    video_playback_steps,
)
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


class KeyboardVelocityCommand(UniformVelocityCommand):
    """Feed held keys into the same command term used by observations and rewards."""

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.0

    def _update_command(self):
        visualizers = self._env.sim.visualizers

        def held(key):
            return any(getattr(viz, "is_key_down", lambda _: False)(key) for viz in visualizers)

        # I/K: forward/back; J/L: turn left/right; U/O: strafe left/right.
        # These keys leave Newton's WASD camera controls available.
        ranges = self.cfg.ranges
        self.vel_command_b[:, 0] = ranges.lin_vel_x[1] * held("i") + ranges.lin_vel_x[0] * held("k")
        self.vel_command_b[:, 1] = ranges.lin_vel_y[1] * held("u") + ranges.lin_vel_y[0] * held("o")
        self.vel_command_b[:, 2] = ranges.ang_vel_z[1] * held("j") + ranges.ang_vel_z[0] * held("l")
        if held("x"):
            self.vel_command_b.zero_()


def configure_keyboard(env_cfg, args):
    """Keep the training observation layout while replacing random velocity sampling."""
    env_cfg.scene.num_envs = 1
    command = env_cfg.commands.base_velocity
    command.class_type = KeyboardVelocityCommand
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (-args.speed, args.speed)
    command.ranges.lin_vel_y = (-args.strafe_speed, args.strafe_speed)
    command.ranges.ang_vel_z = (-args.turn_speed, args.turn_speed)
    command.debug_vis = True
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.time_out = None
    for name in ("randomize_push_robot", "randomize_apply_external_force_torque"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    for name in ("command_levels_lin_vel", "command_levels_ang_vel"):
        if hasattr(env_cfg.curriculum, name):
            setattr(env_cfg.curriculum, name, None)
    # Face the same direction on every reset, with no initial drift.
    reset = env_cfg.events.randomize_reset_base.params
    reset["pose_range"] = dict.fromkeys(reset["pose_range"], (0.0, 0.0))
    reset["velocity_range"] = dict.fromkeys(reset["velocity_range"], (0.0, 0.0))


def positive_float(value):
    """Parse a finite positive speed."""
    import math

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and greater than zero.")
    return number


def run(argv: list[str]) -> int:
    """Load a checkpoint and run interactive, real-time policy inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_play_args(parser, agent_default="rsl_rl_cfg_entry_point", agent_help="Agent configuration entry point.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    parser.add_argument("--speed", type=positive_float, default=0.5, help="Forward/backward speed in m/s.")
    parser.add_argument("--strafe-speed", type=positive_float, default=0.2, help="Lateral speed in m/s.")
    parser.add_argument("--turn-speed", type=positive_float, default=0.6, help="Yaw speed in rad/s.")
    parser.add_argument("--no-follow-camera", action="store_true", help="Allow free camera movement.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after a bounded number of evaluation steps.")
    parser.set_defaults(num_envs=1, checkpoint="latest", real_time=True, visualizer=["newton_gl"])
    args, overrides = setup_preset_cli(parser, argv)
    if args.num_envs != 1:
        parser.error("Keyboard evaluation requires --num_envs 1.")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be greater than zero.")
    if args.frontend != "torch":
        parser.error("Keyboard evaluation requires --frontend torch.")

    version = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=True, overrides=overrides)
    apply_env_overrides(args, env_cfg)
    configure_keyboard(env_cfg, args)
    enable_cameras_for_video(args)
    pre_launch_video_config(env_cfg, args)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version)
    env_cfg.seed = agent_cfg.seed
    root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args.checkpoint in ("latest", "best"):
        checkpoint = resolve_checkpoint_selector(
            root,
            args.checkpoint,
            library="rsl_rl",
            task=normalize_task_name(args.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args.agent},
        )
    else:
        checkpoint = retrieve_file_path(args.checkpoint)
    env_cfg.log_dir = os.path.dirname(checkpoint)
    apply_video_recording(env_cfg, env_cfg.log_dir, args, subdir="play", checkpoint_path=checkpoint)

    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base = env.unwrapped
        has_keyboard = any(callable(getattr(viz, "is_key_down", None)) for viz in base.sim.visualizers)
        if not has_keyboard and args.max_steps is None:
            raise RuntimeError("Keyboard control needs a Newton window. Run with --viz newton_gl.")
        runner = create_rsl_rl_runner(env, agent_cfg)
        print(f"[INFO] Loading keyboard evaluation checkpoint: {checkpoint}")
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=base.device)
        obs = env.get_observations()
        command = base.command_manager.get_term("base_velocity")
        reset_held = False
        limits = [limit for limit in (args.max_steps, video_playback_steps(args, env_cfg)) if limit is not None]
        max_steps = min(limits) if limits else None
        step_count = 0
        print("[Controls] Click viewer: I/K forward/back, J/L turn, U/O strafe; release or hold X = zero target.")
        print("[Controls] B resets robot. Ctrl+C exits. Speeds and measured body velocities print once per second.")
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while base.sim.is_running() and (max_steps is None or step_count < max_steps):
                started = time.monotonic()
                reset_now = any(getattr(viz, "is_key_down", lambda _: False)("b") for viz in base.sim.visualizers)
                if reset_now and not reset_held:
                    obs, _ = env.reset()
                    policy.reset(torch.ones(1, dtype=torch.long, device=base.device))
                reset_held = reset_now
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                step_count += 1
                if not args.no_follow_camera:
                    position = base.scene["robot"].data.root_pos_w.torch[0].cpu().tolist()
                    base.sim.set_camera_view(
                        eye=(position[0] + 2.5, position[1] + 2.5, position[2] + 1.8), target=position
                    )
                if step_count % max(1, round(1.0 / base.step_dt)) == 0:
                    robot = base.scene["robot"].data
                    target = command.command[0].cpu().tolist()
                    linear = robot.root_lin_vel_b.torch[0].cpu().tolist()
                    yaw = robot.root_ang_vel_b.torch[0, 2].item()
                    print(
                        f"[Velocity] target vx/vy/wz={target[0]:+.2f}/{target[1]:+.2f}/{target[2]:+.2f} | "
                        f"actual={linear[0]:+.2f}/{linear[1]:+.2f}/{yaw:+.2f}"
                    )
                if dones.any():
                    print("[INFO] Robot reset after termination (fall/contact/orientation).")
                remaining = base.step_dt - (time.monotonic() - started)
                if args.real_time and remaining > 0:
                    time.sleep(remaining)
        print(f"[INFO] Keyboard evaluation finished: {step_count} steps.")
    return 0
