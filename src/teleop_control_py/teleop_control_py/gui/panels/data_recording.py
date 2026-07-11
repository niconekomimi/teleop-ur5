"""Data recording and runtime log controls for the main window."""

from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import set_standard_icon, set_widget_role


class DataRecordingSettings(Protocol):
    default_hdf5_output_dir: str
    default_hdf5_filename: str


class DataRecordingPanel(QWidget):
    """Build the data recording controls and runtime log area."""

    def __init__(
        self,
        settings: DataRecordingSettings,
        *,
        section_style: str,
        button_height: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.record_group = self._build_record_group(
            settings,
            section_style=section_style,
            button_height=button_height,
        )
        self.preview_group = self._build_log_group(
            section_style=section_style,
        )
        layout.addWidget(self.record_group)
        layout.addWidget(self.preview_group)
        layout.addStretch(1)

    def _build_record_group(
        self,
        settings: DataRecordingSettings,
        *,
        section_style: str,
        button_height: int,
    ) -> QGroupBox:
        group = QGroupBox("数据录制")
        group.setStyleSheet(section_style)
        layout = QGridLayout(group)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)
        for column in range(1, 5):
            layout.setColumnStretch(column, 1)

        layout.addWidget(QLabel("HDF5 保存目录:"), 0, 0)
        self.record_dir_input = QLineEdit(settings.default_hdf5_output_dir)
        self.record_dir_input.setToolTip(settings.default_hdf5_output_dir)
        self.record_dir_input.setCursorPosition(0)
        layout.addWidget(self.record_dir_input, 0, 1, 1, 3)

        self.btn_choose_record_dir = QPushButton("选择目录")
        self.btn_choose_record_dir.setMinimumHeight(button_height)
        set_widget_role(self.btn_choose_record_dir, "secondary")
        set_standard_icon(self.btn_choose_record_dir, QStyle.StandardPixmap.SP_DirOpenIcon)
        layout.addWidget(self.btn_choose_record_dir, 0, 4)

        layout.addWidget(QLabel("HDF5 文件名:"), 1, 0)
        self.record_name_input = QLineEdit(settings.default_hdf5_filename)
        self.record_name_input.setToolTip(settings.default_hdf5_filename)
        self.record_name_input.setPlaceholderText("例如: libero_demos.hdf5")
        layout.addWidget(self.record_name_input, 1, 1, 1, 3)

        self.btn_preview_hdf5 = QPushButton("预览HDF5内容")
        self.btn_preview_hdf5.setMinimumHeight(button_height)
        set_widget_role(self.btn_preview_hdf5, "secondary")
        set_standard_icon(
            self.btn_preview_hdf5,
            QStyle.StandardPixmap.SP_FileDialogContentsView,
        )
        layout.addWidget(self.btn_preview_hdf5, 1, 4)

        layout.addWidget(self._build_camera_row(), 2, 0, 1, 5)
        layout.addWidget(self._build_record_actions_row(button_height), 3, 0, 1, 5)

        self.camera_binding_hint_label = QLabel("相机按型号选择，系统会自动绑定对应设备。")
        self.camera_binding_hint_label.setWordWrap(True)
        set_widget_role(self.camera_binding_hint_label, "hint")
        self.camera_binding_hint_label.setVisible(False)
        layout.addWidget(self.camera_binding_hint_label, 4, 0, 1, 5)

        layout.addWidget(QLabel("当前录制序列:"), 5, 0)
        self.lbl_demo_status = QLabel("无 (未录制)")
        self.lbl_demo_status.setAlignment(Qt.AlignCenter)
        self.lbl_demo_status.setFixedHeight(30)
        set_widget_role(self.lbl_demo_status, "status-neutral")
        layout.addWidget(self.lbl_demo_status, 5, 1)

        self.lbl_main_record_stats = QLabel("录制时长: 00:00 | 帧数: 0")
        self.lbl_main_record_stats.setAlignment(Qt.AlignCenter)
        self.lbl_main_record_stats.setFixedHeight(30)
        set_widget_role(self.lbl_main_record_stats, "status-neutral")
        layout.addWidget(self.lbl_main_record_stats, 5, 2, 1, 3)
        return group

    def _build_camera_row(self) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        global_camera_column = QWidget()
        global_camera_layout = QHBoxLayout(global_camera_column)
        global_camera_layout.setContentsMargins(0, 0, 0, 0)
        global_camera_layout.setSpacing(8)
        global_camera_layout.addWidget(QLabel("录制全局相机:"))
        self.global_camera_source_combo = QComboBox()
        global_camera_layout.addWidget(self.global_camera_source_combo, 1)

        wrist_camera_column = QWidget()
        wrist_camera_layout = QHBoxLayout(wrist_camera_column)
        wrist_camera_layout.setContentsMargins(0, 0, 0, 0)
        wrist_camera_layout.setSpacing(8)
        wrist_camera_layout.addWidget(QLabel("录制局部相机:"))
        self.wrist_camera_source_combo = QComboBox()
        wrist_camera_layout.addWidget(self.wrist_camera_source_combo, 1)

        row_layout.addWidget(global_camera_column, 1)
        row_layout.addWidget(wrist_camera_column, 1)
        return row

    def _build_record_actions_row(self, button_height: int) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        self.btn_collector = QPushButton("启动采集节点")
        self.btn_collector.setFixedHeight(button_height)
        self.btn_collector.setCheckable(True)
        set_widget_role(self.btn_collector, "secondary")
        set_standard_icon(self.btn_collector, QStyle.StandardPixmap.SP_MediaPlay)
        row_layout.addWidget(self.btn_collector, 1)

        self.btn_start_record = QPushButton("开始录制")
        self.btn_start_record.setFixedHeight(button_height)
        set_widget_role(self.btn_start_record, "primary")
        set_standard_icon(self.btn_start_record, QStyle.StandardPixmap.SP_MediaPlay)
        row_layout.addWidget(self.btn_start_record, 1)

        self.btn_stop_record = QPushButton("停止录制")
        self.btn_stop_record.setFixedHeight(button_height)
        set_widget_role(self.btn_stop_record, "secondary")
        set_standard_icon(self.btn_stop_record, QStyle.StandardPixmap.SP_MediaStop)
        row_layout.addWidget(self.btn_stop_record, 1)

        self.btn_discard_record = QPushButton("弃用当前 Demo")
        self.btn_discard_record.setFixedHeight(button_height)
        set_widget_role(self.btn_discard_record, "danger-secondary")
        set_standard_icon(self.btn_discard_record, QStyle.StandardPixmap.SP_TrashIcon)
        row_layout.addWidget(self.btn_discard_record, 1)
        return row

    def _build_log_group(
        self,
        *,
        section_style: str,
    ) -> QGroupBox:
        group = QGroupBox("日志")
        group.setStyleSheet(section_style)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(4)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(130)
        self.log_output.setObjectName("runtimeLog")
        set_widget_role(self.log_output, "runtime-log")
        layout.addWidget(self.log_output)
        return group
