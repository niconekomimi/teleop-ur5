"""Render the ROS-independent operator workspace to deterministic PNG files."""

from __future__ import annotations

import os


# These must be set before importing PySide6. They keep local and CI renders at
# the same logical scale without requiring a desktop session.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "1")
os.environ.setdefault("QT_FONT_DPI", "96")

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "teleop_control_py"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from PySide6.QtCore import QEventLoop, QSize
from PySide6.QtGui import QFontDatabase, QImage
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QMainWindow,
    QPushButton,
    QStyle,
    QWidget,
)

from teleop_control_py.gui.panels import (
    DataRecordingPanel,
    InferencePanel,
    StatusOverviewPanel,
    SystemControlPanel,
)
from teleop_control_py.gui.panels.workspace import WorkspaceShell
from teleop_control_py.gui.theme import (
    SECTION_STYLE,
    configure_application,
    set_standard_icon,
    status_indicator_style,
)


DEFAULT_VIEWPORTS = ((1440, 900), (1280, 800))
SCENARIOS = ("idle", "running", "error")

_NEUTRAL = "#5F6B76"
_INFO = "#2563EB"
_SUCCESS = "#16794B"
_WARNING = "#B15C00"
_DANGER = "#B42318"

_OWNED_APPLICATION: QApplication | None = None
_PREVIEW_FONT_LOADED = False


@dataclass(frozen=True)
class PreviewSettings:
    """Small settings object satisfying all four panel protocols."""

    default_input_type: str = "quest3"
    joy_profiles: tuple[str, ...] = ("auto", "xbox", "dualshock4")
    default_joy_profile: str = "auto"
    ur_type: str = "ur5e"
    mediapipe_camera_options: tuple[str, ...] = ("d455", "d435", "oakd")
    default_mediapipe_camera: str = "d455"
    default_mediapipe_input_topic: str = "/d455/camera/color/image_raw"
    default_robot_ip: str = "192.168.1.211"
    default_gripper_type: str = "robotiq"
    camera_driver_options: tuple[str, ...] = ("realsense", "oakd")
    default_camera_driver: str = "realsense"
    default_hdf5_output_dir: str = "D:/teleop/datasets"
    default_hdf5_filename: str = "pick_place_session.hdf5"
    default_inference_backend: str = "real_il"
    default_inference_device: str = "cuda"
    default_inference_hz: float = 10.0
    default_openpi_host: str = "127.0.0.1"
    default_openpi_port: int = 18000
    default_openpi_prompt: str = "pick up the red block and place it in the basket"
    collect_inference_action_logs: bool = True


@dataclass(frozen=True)
class _ScenarioState:
    phase: tuple[str, str]
    runtime: tuple[str, str]
    preview: tuple[str, str]
    modules: dict[str, tuple[str, str]]
    hardware: dict[str, tuple[str, str]]
    demo_status: tuple[str, str]
    record_stats: str
    inference_status: tuple[str, str]
    execution_status: tuple[str, str]
    log_lines: tuple[str, ...]
    action_text: str


_SCENARIO_STATES = {
    "idle": _ScenarioState(
        phase=("阶段: IDLE", _NEUTRAL),
        runtime=("ROS: 待机", _NEUTRAL),
        preview=("预览源: 关闭", _NEUTRAL),
        modules={
            "robot_driver": ("未启动", _NEUTRAL),
            "teleop": ("未启动", _NEUTRAL),
            "data_collector": ("未启动", _NEUTRAL),
            "inference": ("未启动", _NEUTRAL),
            "preview": ("关闭", _NEUTRAL),
        },
        hardware={
            "joystick": ("未检测到", _WARNING),
            "camera_1": ("D455 可用", _SUCCESS),
            "camera_2": ("OAK-D 可用", _SUCCESS),
            "camera_3": ("未检测到", _NEUTRAL),
            "robot": ("待连接", _NEUTRAL),
            "gripper": ("待连接", _NEUTRAL),
        },
        demo_status=("无（未录制）", _INFO),
        record_stats="录制时长: 00:00 | 帧数: 0",
        inference_status=("未启动", _NEUTRAL),
        execution_status=("未使能", _NEUTRAL),
        log_lines=(
            "[09:41:02] 已加载离线 GUI 预览配置。",
            "[09:41:02] 检测到 2 路模拟相机；未连接 ROS 图。",
            "[09:41:03] 工作站处于安全待机状态。",
        ),
        action_text="等待推理动作...",
    ),
    "running": _ScenarioState(
        phase=("阶段: TELEOP", _INFO),
        runtime=("ROS: 运行中", _SUCCESS),
        preview=("预览源: 2 路在线", _SUCCESS),
        modules={
            "robot_driver": ("运行中", _SUCCESS),
            "teleop": ("运行中", _SUCCESS),
            "data_collector": ("正在录制", _INFO),
            "inference": ("已就绪", _SUCCESS),
            "preview": ("2 路在线", _SUCCESS),
        },
        hardware={
            "joystick": ("Quest 3 已连接", _SUCCESS),
            "camera_1": ("D455 · 30 FPS", _SUCCESS),
            "camera_2": ("OAK-D · 30 FPS", _SUCCESS),
            "camera_3": ("未使用", _NEUTRAL),
            "robot": ("UR5e 已连接", _SUCCESS),
            "gripper": ("Robotiq 已连接", _SUCCESS),
        },
        demo_status=("demo_0007 · 录制中", _INFO),
        record_stats="录制时长: 01:24 | 帧数: 2520 | 30.0 Hz",
        inference_status=("策略已加载 · CUDA", _SUCCESS),
        execution_status=("任务执行中", _INFO),
        log_lines=(
            "[09:43:16] 机械臂驱动已启动，UR5e 状态正常。",
            "[09:43:17] D455 与 OAK-D 时间同步完成。",
            "[09:43:18] 数据采集已开始: demo_0007。",
            "[09:43:19] Real-IL 策略加载完成，执行频率 10.0 Hz。",
        ),
        action_text="Δx +0.014  Δy -0.006  Δz +0.003  grip 0.82",
    ),
    "error": _ScenarioState(
        phase=("阶段: SAFE STOP", _DANGER),
        runtime=("ROS: 通信异常", _DANGER),
        preview=("预览源: 1 路中断", _WARNING),
        modules={
            "robot_driver": ("连接超时", _DANGER),
            "teleop": ("安全停止", _DANGER),
            "data_collector": ("已暂停", _WARNING),
            "inference": ("动作已抑制", _WARNING),
            "preview": ("腕部相机离线", _WARNING),
        },
        hardware={
            "joystick": ("Quest 3 已连接", _SUCCESS),
            "camera_1": ("D455 可用", _SUCCESS),
            "camera_2": ("OAK-D 无响应", _DANGER),
            "camera_3": ("未使用", _NEUTRAL),
            "robot": ("状态超时 1.2 s", _DANGER),
            "gripper": ("状态未知", _WARNING),
        },
        demo_status=("demo_0007 · 已暂停", _WARNING),
        record_stats="录制暂停 | 已保留 1872 帧",
        inference_status=("等待机器人状态恢复", _WARNING),
        execution_status=("安全锁定", _DANGER),
        log_lines=(
            "[09:45:32] 警告: OAK-D 帧流超时 850 ms。",
            "[09:45:33] 错误: 机器人状态超过安全时限。",
            "[09:45:33] 已输出零动作并暂停当前录制。",
            "[09:45:33] 请检查网络和设备电源后重新连接。",
        ),
        action_text="动作已抑制：等待状态恢复",
    ),
}


def _load_preview_font() -> None:
    """Make CJK text available when the Windows offscreen plugin lists no fonts."""

    global _PREVIEW_FONT_LOADED
    if _PREVIEW_FONT_LOADED:
        return

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
    _PREVIEW_FONT_LOADED = True


def _application() -> QApplication:
    global _OWNED_APPLICATION

    app = QApplication.instance()
    if app is None:
        _OWNED_APPLICATION = QApplication([sys.argv[0]])
        app = _OWNED_APPLICATION
    _load_preview_font()
    configure_application(app)
    return app


def _set_combo_items(
    combo: QComboBox,
    entries: Sequence[tuple[str, Any]],
    selected_index: int = 0,
) -> None:
    combo.clear()
    for text, data in entries:
        combo.addItem(text, data)
    if entries:
        combo.setCurrentIndex(max(0, min(selected_index, len(entries) - 1)))


def _seed_preview_data(
    system_panel: SystemControlPanel,
    recording_panel: DataRecordingPanel,
    inference_panel: InferencePanel,
) -> None:
    cameras = (
        (
            "D455 · 全局",
            {"source": "realsense", "model": "d455", "serial": "SIM-D455-01"},
        ),
        (
            "D435 · 备用",
            {"source": "realsense", "model": "d435", "serial": "SIM-D435-02"},
        ),
        (
            "OAK-D · 腕部",
            {"source": "oakd", "model": "oakd", "serial": "SIM-OAKD-01"},
        ),
    )
    _set_combo_items(system_panel.mediapipe_camera_combo, cameras, 0)
    _set_combo_items(recording_panel.global_camera_source_combo, cameras, 0)
    _set_combo_items(recording_panel.wrist_camera_source_combo, cameras, 2)
    _set_combo_items(inference_panel.inference_global_camera_combo, cameras, 0)
    _set_combo_items(inference_panel.inference_wrist_camera_combo, cameras, 2)

    _set_combo_items(
        inference_panel.inference_env_combo,
        (("LIBERO Object", "libero_object"), ("Drawer", "drawer")),
        0,
    )
    _set_combo_items(
        inference_panel.inference_task_combo,
        (
            ("Pick up the red block", "pick_up_red_block"),
            ("Place the white mug in the drawer", "place_mug_in_drawer"),
        ),
        0,
    )
    inference_panel.inference_model_dir_input.setText(
        "models/real_il/diffusion_policy_2026-07-10"
    )
    inference_panel.inference_embedding_input.setText(
        "models/embeddings/libero_object/pick_up_red_block.pt"
    )
    inference_panel.inference_backend_combo.setCurrentIndex(
        max(0, inference_panel.inference_backend_combo.findData("real_il"))
    )
    inference_panel.inference_device_combo.setCurrentIndex(
        max(0, inference_panel.inference_device_combo.findData("cuda"))
    )
    inference_panel.inference_hz_spin.setValue(10.0)
    for widget in inference_panel._real_il_widgets:
        widget.setEnabled(True)
    for widget in inference_panel._openpi_widgets:
        widget.setEnabled(False)


def _set_status(widget: QWidget, text: str, color: str) -> None:
    widget.setProperty("previewStatusColor", color)
    if hasattr(widget, "setText"):
        widget.setText(text)
    widget.setStyleSheet(status_indicator_style(color))


def _set_button_state(
    button: QPushButton,
    *,
    checked: bool = False,
    enabled: bool = True,
    text: str | None = None,
) -> None:
    button.setEnabled(enabled)
    if button.isCheckable():
        button.setChecked(checked)
        set_standard_icon(
            button,
            QStyle.StandardPixmap.SP_MediaStop
            if checked
            else QStyle.StandardPixmap.SP_MediaPlay,
        )
    if text is not None:
        button.setText(text)


def _apply_scenario(
    scenario: str,
    shell: WorkspaceShell,
    system_panel: SystemControlPanel,
    recording_panel: DataRecordingPanel,
    inference_panel: InferencePanel,
    module_labels: dict[str, Any],
    hardware_labels: dict[str, Any],
) -> None:
    try:
        state = _SCENARIO_STATES[scenario]
    except KeyError as exc:
        choices = ", ".join(SCENARIOS)
        raise ValueError(f"Unknown preview scenario {scenario!r}; choose one of: {choices}") from exc

    _set_status(shell.phase_label, *state.phase)
    _set_status(shell.runtime_label, *state.runtime)
    _set_status(shell.preview_label, *state.preview)
    for key, label in module_labels.items():
        _set_status(label, *state.modules[key])
    for key, label in hardware_labels.items():
        _set_status(label, *state.hardware[key])

    _set_status(recording_panel.lbl_demo_status, *state.demo_status)
    recording_panel.lbl_main_record_stats.setText(state.record_stats)
    recording_panel.log_output.setPlainText("\n".join(state.log_lines))
    recording_panel.log_output.setObjectName("runtimeLog")
    recording_panel.log_output.setProperty("role", "runtimeLog")

    _set_status(inference_panel.lbl_inference_status, *state.inference_status)
    _set_status(inference_panel.lbl_inference_execute_status, *state.execution_status)
    inference_panel.inference_action_output.setPlainText(state.action_text)
    inference_panel.inference_action_output.setObjectName("actionOutput")
    inference_panel.inference_action_output.setProperty("role", "actionOutput")

    is_running = scenario == "running"
    is_error = scenario == "error"
    _set_button_state(
        system_panel.btn_robot_driver,
        checked=is_running,
        enabled=not is_error,
        text="停止机械臂驱动" if is_running else "启动机械臂驱动",
    )
    _set_button_state(
        system_panel.btn_teleop,
        checked=is_running,
        enabled=not is_error,
        text="停止遥操作系统" if is_running else "启动遥操作系统",
    )
    _set_button_state(
        recording_panel.btn_collector,
        checked=is_running,
        enabled=not is_error,
        text="停止采集节点" if is_running else "启动采集节点",
    )
    _set_button_state(
        recording_panel.btn_start_record,
        enabled=scenario == "idle",
    )
    _set_button_state(
        recording_panel.btn_stop_record,
        enabled=is_running,
    )
    _set_button_state(
        recording_panel.btn_discard_record,
        enabled=scenario != "idle",
    )
    _set_button_state(
        inference_panel.btn_inference,
        checked=is_running,
        enabled=not is_error,
        text="停止推理" if is_running else "启动推理",
    )
    _set_button_state(
        inference_panel.btn_execute_inference,
        checked=is_running,
        enabled=is_running,
        text="停止执行任务" if is_running else "开始执行任务",
    )
    _set_button_state(inference_panel.btn_inference_estop, enabled=is_running)


def build_preview(
    settings: Any | None = None,
    scenario: str = "idle",
) -> QMainWindow:
    """Build a hardware-free preview matching the production workspace."""

    _application()
    effective_settings = settings if settings is not None else PreviewSettings()

    window = QMainWindow()
    window.setObjectName("teleopPreviewWindow")
    window.setWindowTitle("Teleop Control · Offline UI Preview")

    shell = WorkspaceShell(window)
    window.setCentralWidget(shell)

    module_labels: dict[str, Any] = {}
    hardware_labels: dict[str, Any] = {}
    system_panel = SystemControlPanel(
        effective_settings,
        local_ip="192.168.1.10",
        section_style=SECTION_STYLE,
        button_height=30,
    )
    system_panel.setObjectName("systemControlPanel")
    recording_panel = DataRecordingPanel(
        effective_settings,
        section_style=SECTION_STYLE,
        button_height=30,
    )
    recording_panel.setObjectName("dataRecordingPanel")
    status_panel = StatusOverviewPanel(
        section_style=SECTION_STYLE,
        module_status_labels=module_labels,
        hardware_status_labels=hardware_labels,
    )
    status_panel.setObjectName("statusOverviewPanel")
    inference_panel = InferencePanel(
        effective_settings,
        section_style=SECTION_STYLE,
        button_height=30,
        emphasis_spin_height=30,
    )
    inference_panel.setObjectName("inferencePanel")

    shell.left_layout.addWidget(system_panel)
    shell.left_layout.addWidget(recording_panel, 1)
    shell.right_layout.addWidget(status_panel)
    shell.right_layout.addWidget(inference_panel)
    shell.right_layout.addStretch(1)

    _seed_preview_data(system_panel, recording_panel, inference_panel)
    _apply_scenario(
        scenario,
        shell,
        system_panel,
        recording_panel,
        inference_panel,
        module_labels,
        hardware_labels,
    )
    window.setProperty("previewScenario", scenario)
    return window


def _sampled_color_count(image: QImage) -> int:
    x_step = max(1, image.width() // 24)
    y_step = max(1, image.height() // 16)
    return len(
        {
            image.pixelColor(x, y).rgba()
            for x in range(0, image.width(), x_step)
            for y in range(0, image.height(), y_step)
        }
    )


def render_preview(
    window: QMainWindow,
    output_path: str | Path,
    size: tuple[int, int],
) -> Path:
    """Render a preview window at an exact logical viewport and save a PNG."""

    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Preview size must be positive, got {width}x{height}")

    app = _application()
    window.setFixedSize(QSize(width, height))
    window.show()
    window.ensurePolished()
    for widget in window.findChildren(QWidget):
        widget.ensurePolished()
        layout = widget.layout()
        if layout is not None:
            layout.activate()
    app.sendPostedEvents()
    app.processEvents(QEventLoop.AllEvents, 100)
    window.repaint()
    app.processEvents(QEventLoop.AllEvents, 100)

    pixmap = window.grab()
    if pixmap.isNull():
        raise RuntimeError("Qt returned a null pixmap for the preview window")
    logical_size = pixmap.deviceIndependentSize().toSize()
    expected_size = QSize(width, height)
    if logical_size != expected_size:
        raise RuntimeError(
            "Preview raster has the wrong logical size: "
            f"expected {width}x{height}, got {logical_size.width()}x{logical_size.height()}"
        )

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(destination), "PNG"):
        raise RuntimeError(f"Failed to save preview PNG: {destination}")

    image = QImage(str(destination))
    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        raise RuntimeError(f"Saved preview PNG is unreadable: {destination}")
    if _sampled_color_count(image) < 4:
        raise RuntimeError(f"Saved preview PNG appears blank: {destination}")

    window.hide()
    app.processEvents(QEventLoop.AllEvents, 50)
    return destination


_SIZE_PATTERN = re.compile(r"^(?P<width>[1-9]\d*)[xX](?P<height>[1-9]\d*)$")


def _parse_size(value: str) -> tuple[int, int]:
    match = _SIZE_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid viewport {value!r}; expected WIDTHxHEIGHT, for example 1440x900"
        )
    return int(match.group("width")), int(match.group("height"))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the ROS-independent teleoperation GUI preview.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gui",
        help="Directory for generated PNG files (default: artifacts/gui).",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=_parse_size,
        default=list(DEFAULT_VIEWPORTS),
        metavar="WIDTHxHEIGHT",
        help="One or more logical viewport sizes.",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="idle",
        help="Fake runtime state to display (default: idle).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    app = _application()
    for width, height in args.sizes:
        window = build_preview(PreviewSettings(), args.scenario)
        try:
            filename = f"teleop-preview-{args.scenario}-{width}x{height}.png"
            output_path = render_preview(
                window,
                args.output_dir / filename,
                (width, height),
            )
            print(output_path)
        finally:
            window.close()
            window.deleteLater()
            app.processEvents(QEventLoop.AllEvents, 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
