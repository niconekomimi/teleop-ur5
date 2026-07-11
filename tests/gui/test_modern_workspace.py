from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("QT_SCALE_FACTOR", "1")
os.environ.setdefault("QT_FONT_DPI", "96")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from teleop_control_py.gui.panels import (
    DataRecordingPanel,
    InferencePanel,
    StatusOverviewPanel,
    SystemControlPanel,
    WorkspaceShell,
)
from teleop_control_py.gui.theme import (
    APP_STYLESHEET,
    SECTION_STYLE,
    configure_application,
    set_widget_role,
)


_TEST_FONT_LOADED = False


def _configure_test_application(qapp) -> None:
    global _TEST_FONT_LOADED
    if not _TEST_FONT_LOADED:
        windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
        candidates = (
            windows_root / "Fonts" / "msyh.ttc",
            windows_root / "Fonts" / "simhei.ttf",
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        )
        for font_path in candidates:
            if font_path.is_file() and QFontDatabase.addApplicationFont(str(font_path)) >= 0:
                break
        _TEST_FONT_LOADED = True
    configure_application(qapp)


def _settings() -> SimpleNamespace:
    """Return the smallest settings object needed by the ROS-free panels."""

    return SimpleNamespace(
        default_input_type="joy",
        joy_profiles=("auto",),
        default_joy_profile="auto",
        ur_type="ur5e",
        mediapipe_camera_options=("d435",),
        default_mediapipe_camera="d435",
        default_mediapipe_input_topic="/camera/image_raw",
        default_robot_ip="192.168.1.10",
        default_gripper_type="robotiq",
        camera_driver_options=("disabled",),
        default_camera_driver="disabled",
        default_hdf5_output_dir="data",
        default_hdf5_filename="demo.hdf5",
        default_openpi_host="127.0.0.1",
        default_openpi_port=8000,
        default_openpi_prompt="pick up the object",
        collect_inference_action_logs=False,
    )


def _assemble_workspace(qtbot, qapp):
    _configure_test_application(qapp)
    shell = WorkspaceShell()
    qtbot.addWidget(shell)

    module_labels: dict[str, QLabel] = {}
    hardware_labels: dict[str, QLabel] = {}
    settings = _settings()
    system_panel = SystemControlPanel(
        settings,
        local_ip="127.0.0.1",
        section_style=SECTION_STYLE,
        button_height=30,
    )
    recording_panel = DataRecordingPanel(
        settings,
        section_style=SECTION_STYLE,
        button_height=30,
    )
    status_panel = StatusOverviewPanel(
        section_style=SECTION_STYLE,
        module_status_labels=module_labels,
        hardware_status_labels=hardware_labels,
    )
    inference_panel = InferencePanel(
        settings,
        section_style=SECTION_STYLE,
        button_height=30,
        emphasis_spin_height=30,
    )

    shell.left_layout.addWidget(system_panel)
    shell.left_layout.addWidget(recording_panel, 1)
    shell.right_layout.addWidget(status_panel)
    shell.right_layout.addWidget(inference_panel)
    shell.right_layout.addStretch(1)
    return shell, system_panel, recording_panel, status_panel, inference_panel


def test_application_theme_and_widget_roles(qapp, qtbot) -> None:
    _configure_test_application(qapp)

    assert qapp.styleSheet() == APP_STYLESHEET
    assert 'QPushButton[role="primary"]' in qapp.styleSheet()
    qapp.setStyleSheet("")
    try:
        assert qapp.style().objectName().casefold() == "fusion"
    finally:
        qapp.setStyleSheet(APP_STYLESHEET)

    button = QPushButton("Run")
    qtbot.addWidget(button)
    set_widget_role(button, " primary ")
    assert button.property("role") == "primary"

    set_widget_role(button, "runtime_log")
    assert button.property("role") == "runtimeLog"


def test_workspace_has_status_header_and_independent_scrolling(qtbot, qapp) -> None:
    _configure_test_application(qapp)
    shell = WorkspaceShell()
    qtbot.addWidget(shell)

    assert (
        shell.phase_label.text(),
        shell.runtime_label.text(),
        shell.preview_label.text(),
    ) == ("阶段: IDLE", "ROS: 待机", "预览源: 关闭")
    assert all(
        label.property("role") == "status"
        for label in (shell.phase_label, shell.runtime_label, shell.preview_label)
    )
    assert shell.preview_button.text() == "实时预览"
    assert shell.preview_button.toolTip() == "打开实时预览与状态窗口"
    assert shell.preview_button.property("role") == "secondary"
    assert not shell.preview_button.icon().isNull()
    assert shell.preview_button.parent() is shell.top_bar

    assert shell.splitter.orientation() == Qt.Orientation.Horizontal
    assert not shell.splitter.childrenCollapsible()
    assert not shell.splitter.isCollapsible(0)
    assert not shell.splitter.isCollapsible(1)
    assert shell.left_scroll.sizePolicy().horizontalStretch() == 48
    assert shell.right_scroll.sizePolicy().horizontalStretch() == 52
    assert shell.left_scroll is not shell.right_scroll
    assert shell.left_scroll.widget() is shell.left_content
    assert shell.right_scroll.widget() is shell.right_content
    assert shell.left_scroll.widgetResizable()
    assert shell.right_scroll.widgetResizable()
    assert shell.left_scroll.verticalScrollBar() is not shell.right_scroll.verticalScrollBar()

    tall_left_content = QWidget()
    tall_left_content.setMinimumHeight(1200)
    shell.left_layout.addWidget(tall_left_content)
    shell.resize(1000, 600)
    shell.show()
    qapp.processEvents()

    sizes = shell.splitter.sizes()
    assert abs(sizes[0] / sum(sizes) - 0.48) < 0.02
    assert shell.left_scroll.verticalScrollBar().maximum() > 0
    assert shell.right_scroll.verticalScrollBar().maximum() == 0
    shell.left_scroll.verticalScrollBar().setValue(
        shell.left_scroll.verticalScrollBar().maximum()
    )
    assert shell.right_scroll.verticalScrollBar().value() == 0


def test_assembled_panels_expose_semantic_command_buttons(qtbot, qapp) -> None:
    shell, system, recording, status, inference = _assemble_workspace(qtbot, qapp)

    assert shell.left_layout.indexOf(system) >= 0
    assert shell.left_layout.indexOf(recording) >= 0
    assert shell.right_layout.indexOf(status) >= 0
    assert shell.right_layout.indexOf(inference) >= 0

    expected_roles = (
        (system.btn_refresh_topics, "secondary"),
        (system.btn_teleop_settings, "secondary"),
        (system.btn_robot_driver, "secondary"),
        (system.btn_teleop, "primary"),
        (system.btn_go_home, "warning"),
        (system.btn_go_home_zone, "warning"),
        (system.btn_set_home_current, "secondary"),
        (recording.btn_choose_record_dir, "secondary"),
        (recording.btn_preview_hdf5, "secondary"),
        (recording.btn_collector, "secondary"),
        (recording.btn_start_record, "primary"),
        (recording.btn_stop_record, "secondary"),
        (recording.btn_discard_record, "danger-secondary"),
        (shell.preview_button, "secondary"),
        (inference.btn_browse_inference_model, "secondary"),
        (inference.btn_refresh_inference_options, "secondary"),
        (inference.btn_browse_inference_embedding, "secondary"),
        (inference.btn_auto_match_embedding, "secondary"),
        (inference.btn_inference, "secondary"),
        (inference.btn_execute_inference, "primary"),
        (inference.btn_inference_estop, "warning"),
    )
    for button, role in expected_roles:
        assert button.property("role") == role, button.text()

    icon_buttons = tuple(
        button for button, _role in expected_roles if button is not system.btn_teleop_settings
    )
    for button in icon_buttons:
        assert not button.icon().isNull(), button.text()
        assert button.iconSize().width() == 16
        assert button.iconSize().height() == 16

    assert status.findChildren(QPushButton) == []
    assert inference.btn_inference_estop.text() == "停止推理输出"


def test_panels_do_not_overflow_horizontally_at_1280_by_800(qtbot, qapp) -> None:
    shell, *_panels = _assemble_workspace(qtbot, qapp)
    shell.resize(1280, 800)
    shell.show()
    qapp.processEvents()

    horizontal_overflow = (
        shell.left_scroll.horizontalScrollBar().maximum(),
        shell.right_scroll.horizontalScrollBar().maximum(),
    )
    assert horizontal_overflow == (0, 0)


def test_log_and_inference_panels_share_a_bottom_edge(qtbot, qapp) -> None:
    shell, _system, recording, _status, inference = _assemble_workspace(qtbot, qapp)

    for width, height in ((1280, 800), (1440, 900)):
        shell.resize(width, height)
        shell.show()
        qapp.processEvents()

        log_bottom = recording.preview_group.mapTo(
            shell,
            recording.preview_group.rect().bottomLeft(),
        ).y()
        inference_bottom = inference.mapTo(
            shell,
            inference.rect().bottomLeft(),
        ).y()

        assert log_bottom == inference_bottom
        assert shell.left_scroll.verticalScrollBar().maximum() == 0
        assert shell.right_scroll.verticalScrollBar().maximum() == 0
