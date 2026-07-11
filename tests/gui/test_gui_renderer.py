from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QLabel

from scripts import render_gui_preview as renderer


_ROS_IMPORT_PREFIXES = ("rclpy", "ament_index_python")


def _loaded_ros_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in _ROS_IMPORT_PREFIXES or name.startswith(
            tuple(f"{prefix}." for prefix in _ROS_IMPORT_PREFIXES)
        )
    }


def test_build_preview_does_not_load_ros_or_ament(qtbot) -> None:
    window = renderer.build_preview(scenario="idle")
    qtbot.addWidget(window)

    assert _loaded_ros_modules() == set()


@pytest.mark.parametrize(
    (
        "scenario",
        "phase",
        "runtime",
        "preview",
        "colors",
        "driver_state",
        "execution_state",
    ),
    (
        (
            "idle",
            "\u9636\u6bb5: IDLE",
            "ROS: \u5f85\u673a",
            "\u9884\u89c8\u6e90: \u5173\u95ed",
            ("#5F6B76", "#5F6B76", "#5F6B76"),
            (False, True),
            (False, False, False),
        ),
        (
            "running",
            "\u9636\u6bb5: TELEOP",
            "ROS: \u8fd0\u884c\u4e2d",
            "\u9884\u89c8\u6e90: 2 \u8def\u5728\u7ebf",
            ("#2563EB", "#16794B", "#16794B"),
            (True, True),
            (True, True, True),
        ),
        (
            "error",
            "\u9636\u6bb5: SAFE STOP",
            "ROS: \u901a\u4fe1\u5f02\u5e38",
            "\u9884\u89c8\u6e90: 1 \u8def\u4e2d\u65ad",
            ("#B42318", "#B42318", "#B15C00"),
            (False, False),
            (False, False, False),
        ),
    ),
)
def test_build_preview_applies_key_scenario_states(
    qtbot,
    scenario: str,
    phase: str,
    runtime: str,
    preview: str,
    colors: tuple[str, str, str],
    driver_state: tuple[bool, bool],
    execution_state: tuple[bool, bool, bool],
) -> None:
    window = renderer.build_preview(scenario=scenario)
    qtbot.addWidget(window)

    status_labels = (
        window.findChild(QLabel, "workspacePhaseStatus"),
        window.findChild(QLabel, "workspaceRuntimeStatus"),
        window.findChild(QLabel, "workspacePreviewStatus"),
    )
    assert all(status_labels)
    assert window.property("previewScenario") == scenario
    assert tuple(label.text() for label in status_labels) == (phase, runtime, preview)
    assert tuple(label.property("previewStatusColor") for label in status_labels) == colors

    system_panel = window.findChild(
        renderer.SystemControlPanel,
        "systemControlPanel",
    )
    inference_panel = window.findChild(
        renderer.InferencePanel,
        "inferencePanel",
    )
    assert system_panel is not None
    assert inference_panel is not None
    assert (
        system_panel.btn_robot_driver.isChecked(),
        system_panel.btn_robot_driver.isEnabled(),
    ) == driver_state
    assert (
        inference_panel.btn_execute_inference.isChecked(),
        inference_panel.btn_execute_inference.isEnabled(),
        inference_panel.btn_inference_estop.isEnabled(),
    ) == execution_state


def test_render_preview_writes_readable_nonblank_png(qtbot, tmp_path) -> None:
    window = renderer.build_preview(scenario="idle")
    qtbot.addWidget(window)
    output_path = tmp_path / "teleop-preview-idle-1280x800.png"

    rendered_path = renderer.render_preview(window, output_path, (1280, 800))

    assert rendered_path == output_path.resolve()
    assert rendered_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert rendered_path.stat().st_size > 1024

    image = QImage(str(rendered_path))
    assert not image.isNull()
    assert image.size() == QSize(1280, 800)
    sampled_colors = {
        image.pixelColor(x, y).rgba()
        for x in range(0, image.width(), 64)
        for y in range(0, image.height(), 50)
    }
    assert len(sampled_colors) >= 4


@pytest.mark.parametrize("scenario", renderer.SCENARIOS)
def test_scenario_preview_has_no_scroll_overflow_at_1280_by_800(
    qtbot,
    tmp_path,
    scenario: str,
) -> None:
    window = renderer.build_preview(scenario=scenario)
    qtbot.addWidget(window)

    renderer.render_preview(
        window,
        tmp_path / f"teleop-preview-{scenario}-1280x800.png",
        (1280, 800),
    )
    shell = window.findChild(renderer.WorkspaceShell)

    assert shell is not None
    assert shell.left_scroll.verticalScrollBar().maximum() == 0
    assert shell.right_scroll.verticalScrollBar().maximum() == 0
