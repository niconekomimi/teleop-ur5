"""Runtime status overview panel for the main window."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..theme import set_widget_role


class StatusOverviewPanel(QGroupBox):
    MODULE_ENTRIES = (
        ("robot_driver", "机械臂 ROS2 驱动"),
        ("teleop", "遥操作系统"),
        ("data_collector", "采集节点"),
        ("inference", "模型推理"),
        ("preview", "实时预览"),
    )
    HARDWARE_ENTRIES = (
        ("joystick", "手柄设备"),
        ("camera_1", "相机1"),
        ("camera_2", "相机2"),
        ("camera_3", "相机3"),
        ("robot", "机械臂"),
        ("gripper", "末端执行器"),
    )

    def __init__(
        self,
        *,
        section_style: str,
        module_status_labels: MutableMapping[str, QLabel],
        hardware_status_labels: MutableMapping[str, QLabel],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("状态总览", parent)
        self.setStyleSheet(section_style)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(18)
        layout.addWidget(
            self._build_status_section("模块状态", self.MODULE_ENTRIES, module_status_labels),
            1,
        )
        layout.addWidget(
            self._build_status_section("硬件状态", self.HARDWARE_ENTRIES, hardware_status_labels),
            1,
        )

    @staticmethod
    def _build_status_section(
        title: str,
        entries: Sequence[tuple[str, str]],
        target: MutableMapping[str, QLabel],
    ) -> QWidget:
        section = QWidget()
        section.setProperty("role", "status-section")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(5)

        title_label = QLabel(title)
        set_widget_role(title_label, "subsection-title")
        section_layout.addWidget(title_label)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        name_header = QLabel("名称")
        status_header = QLabel("状态")
        set_widget_role(name_header, "table-header")
        set_widget_role(status_header, "table-header")
        grid.addWidget(name_header, 0, 0)
        grid.addWidget(status_header, 0, 1)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 3)
        for row, (key, label_text) in enumerate(entries, start=1):
            grid.addWidget(QLabel(label_text), row, 0)
            value_label = QLabel("未知")
            value_label.setAlignment(Qt.AlignCenter)
            set_widget_role(value_label, "status-neutral")
            target[key] = value_label
            grid.addWidget(value_label, row, 1)
        section_layout.addWidget(grid_host)
        section_layout.addStretch(1)
        return section
