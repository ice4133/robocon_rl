# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Newton viewer panel comparing commanded and measured body-frame velocities."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np


class VelocityTrackingPanel:
    """Cache one simulation sample per step and render it without GPU reads in UI callbacks."""

    AXES = (("Forward vx", "m/s"), ("Sideways vy", "m/s"), ("Turn wz", "rad/s"))
    TARGET_COLOR = (1.0, 0.72, 0.30, 1.0)
    ACTUAL_COLOR = (0.30, 0.85, 0.92, 1.0)

    def __init__(self, history_seconds: float = 5.0):
        self.history_seconds = history_seconds
        self.show_history = False
        self.elapsed = 0.0
        self.target = np.zeros(3, dtype=np.float64)
        self.actual = np.zeros(3, dtype=np.float64)
        self.history = deque()

    @property
    def error(self) -> np.ndarray:
        """Signed tracking error in the same units as each velocity component."""
        return self.actual - self.target

    def reset(self) -> None:
        """Start a fresh history after a manual reset or a terminated episode."""
        self.elapsed = 0.0
        self.target.fill(0.0)
        self.actual.fill(0.0)
        self.history.clear()

    def update(self, target: Sequence[float], actual: Sequence[float], dt: float) -> None:
        """Record a matched target/actual sample at the simulation control rate."""
        self.target = np.array(target, dtype=np.float64, copy=True)
        self.actual = np.array(actual, dtype=np.float64, copy=True)
        self.elapsed += dt
        self.history.append((self.elapsed, *self.target, *self.actual))
        while self.history and self.history[0][0] < self.elapsed - self.history_seconds:
            self.history.popleft()

    def draw(self, imgui) -> None:
        """Render a movable panel through Newton's public UI callback interface."""
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(
            imgui.ImVec2(viewport.work_pos.x + viewport.work_size.x - 12, viewport.work_pos.y + 12),
            imgui.Cond_.first_use_ever,
            imgui.ImVec2(1.0, 0.0),
        )
        width = min(490.0, viewport.work_size.x - 24.0)
        imgui.set_next_window_size(imgui.ImVec2(width, 0.0), imgui.Cond_.always)
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(width, 0.0), imgui.ImVec2(width, max(120.0, viewport.work_size.y - 24.0))
        )
        expanded, _ = imgui.begin("Velocity tracking", flags=imgui.WindowFlags_.always_auto_resize)
        try:
            if expanded:
                self._draw_values(imgui)
                _, self.show_history = imgui.checkbox("Show last 5 seconds", self.show_history)
                if self.show_history:
                    self._draw_history(imgui)
        finally:
            imgui.end()

    def _draw_values(self, imgui) -> None:
        imgui.text("Robot body frame | live velocities")
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.sizing_stretch_prop
        if imgui.begin_table("velocity_values", 4, flags):
            try:
                imgui.table_setup_column("Motion / unit", init_width_or_weight=1.8)
                for name in ("Target", "Actual", "Delta"):
                    imgui.table_setup_column(name, init_width_or_weight=1.0)
                imgui.table_headers_row()
                for axis, (label, unit) in enumerate(self.AXES):
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    imgui.text(f"{label} [{unit}]")
                    imgui.table_set_column_index(1)
                    imgui.text_colored(imgui.ImVec4(*self.TARGET_COLOR), f"{self.target[axis]:+.3f}")
                    imgui.table_set_column_index(2)
                    imgui.text_colored(imgui.ImVec4(*self.ACTUAL_COLOR), f"{self.actual[axis]:+.3f}")
                    imgui.table_set_column_index(3)
                    imgui.text(f"{self.error[axis]:+.3f}")
            finally:
                imgui.end_table()
        imgui.text("Delta = Actual - Target (closer to 0 is better)")
        imgui.text_disabled("+vx forward / -vx reverse; +vy left / -vy right")
        imgui.text_disabled("+wz left turn / -wz right turn")

    def _draw_history(self, imgui) -> None:
        # Newton GL already owns the ImPlot context; use its active context.
        from imgui_bundle import implot

        if len(self.history) < 2 or implot.get_current_context() is None:
            imgui.text_disabled("Waiting for velocity history...")
            return
        samples = np.asarray(self.history, dtype=np.float64)
        times = np.ascontiguousarray(samples[:, 0] - self.elapsed)
        target_style = implot.Spec(line_color=imgui.ImVec4(*self.TARGET_COLOR), line_weight=2.0)
        actual_style = implot.Spec(line_color=imgui.ImVec4(*self.ACTUAL_COLOR), line_weight=2.0)
        plot_height = min(160.0, max(120.0, (imgui.get_main_viewport().work_size.y - 300.0) / 3.0))
        for axis, (label, unit) in enumerate(self.AXES):
            if implot.begin_plot(f"{label} [{unit}]##velocity_{axis}", imgui.ImVec2(-1.0, plot_height)):
                try:
                    target = np.ascontiguousarray(samples[:, 1 + axis])
                    actual = np.ascontiguousarray(samples[:, 4 + axis])
                    extent = max(0.1, float(np.abs(target).max()), float(np.abs(actual).max())) * 1.2
                    implot.setup_axes("Time [s, 0 = now]", unit)
                    implot.setup_axes_limits(-self.history_seconds, 0.0, -extent, extent, implot.Cond_.always)
                    implot.setup_legend(
                        implot.Location_.north, implot.LegendFlags_.horizontal | implot.LegendFlags_.outside
                    )
                    implot.plot_line("Target", times, target, target_style)
                    implot.plot_line("Actual", times, actual, actual_style)
                finally:
                    implot.end_plot()
