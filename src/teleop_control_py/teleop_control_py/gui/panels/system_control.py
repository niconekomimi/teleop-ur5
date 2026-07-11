"""System configuration and command controls for the main window."""

from __future__ import annotations

from typing import Protocol, Sequence

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..theme import set_standard_icon, set_widget_role


class SystemControlSettings(Protocol):
    default_input_type: str
    joy_profiles: Sequence[str]
    default_joy_profile: str
    ur_type: str
    mediapipe_camera_options: Sequence[str]
    default_mediapipe_camera: str
    default_mediapipe_input_topic: str
    default_robot_ip: str
    default_gripper_type: str
    camera_driver_options: Sequence[str]
    default_camera_driver: str


class SystemControlPanel(QWidget):
    """Build the system settings, startup, and Home control groups."""

    def __init__(
        self,
        settings: SystemControlSettings,
        *,
        local_ip: str,
        section_style: str,
        button_height: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.settings_group = self._build_settings_group(
            settings,
            local_ip=local_ip,
            section_style=section_style,
            button_height=button_height,
        )
        self.startup_group = self._build_startup_group(
            settings,
            section_style=section_style,
            button_height=button_height,
        )
        self.home_group = self._build_home_group(
            section_style=section_style,
            button_height=button_height,
        )
        layout.addWidget(self.settings_group)
        layout.addWidget(self.startup_group)
        layout.addWidget(self.home_group)

    def _build_settings_group(
        self,
        settings: SystemControlSettings,
        *,
        local_ip: str,
        section_style: str,
        button_height: int,
    ) -> QGroupBox:
        group = QGroupBox("系统配置")
        group.setStyleSheet(section_style)
        layout = QGridLayout(group)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

        input_backend_label = QLabel("输入后端:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("joy (手柄)", "joy")
        self.mode_combo.addItem("mediapipe (手势输入)", "mediapipe")
        self.mode_combo.addItem("quest3 (VR 控制器)", "quest3")
        index = max(0, self.mode_combo.findData(settings.default_input_type))
        self.mode_combo.setCurrentIndex(index)

        joystick_profile_label = QLabel("手柄型号:")
        self.joy_profile_combo = QComboBox()
        for profile in settings.joy_profiles:
            self.joy_profile_combo.addItem(str(profile), str(profile))
        index = max(0, self.joy_profile_combo.findData(settings.default_joy_profile))
        self.joy_profile_combo.setCurrentIndex(index)

        ur_type_label = QLabel("UR 类型:")
        self.ur_type_input = QLineEdit(settings.ur_type or "ur5")
        self.ur_type_input.setPlaceholderText("例如: ur5, ur10e, ur16e")

        for widget in (self.mode_combo, self.joy_profile_combo, self.ur_type_input):
            widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        input_row = QWidget()
        input_row_layout = QHBoxLayout(input_row)
        input_row_layout.setContentsMargins(0, 0, 0, 0)
        input_row_layout.setSpacing(8)
        input_row_layout.addWidget(input_backend_label)
        input_row_layout.addWidget(self.mode_combo, 3)
        input_row_layout.addWidget(joystick_profile_label)
        input_row_layout.addWidget(self.joy_profile_combo, 2)
        input_row_layout.addWidget(ur_type_label)
        input_row_layout.addWidget(self.ur_type_input, 2)
        layout.addWidget(input_row, 0, 0, 1, 4)

        layout.addWidget(QLabel("手势识别输入相机:"), 1, 0)
        self.mediapipe_camera_combo = QComboBox()
        for option in settings.mediapipe_camera_options:
            normalized = str(option).strip().lower() or "d435"
            self.mediapipe_camera_combo.addItem(str(option), normalized)
        index = max(0, self.mediapipe_camera_combo.findData(settings.default_mediapipe_camera))
        self.mediapipe_camera_combo.setCurrentIndex(index)
        layout.addWidget(self.mediapipe_camera_combo, 1, 1)

        self.mediapipe_topic_combo = QComboBox()
        self.mediapipe_topic_combo.setEditable(True)
        self.mediapipe_topic_combo.setCurrentText(settings.default_mediapipe_input_topic)
        self.mediapipe_topic_combo.setVisible(False)
        layout.addWidget(self.mediapipe_topic_combo, 1, 2)

        self.btn_refresh_topics = QPushButton("刷新SDK相机")
        self.btn_refresh_topics.setMinimumHeight(button_height)
        set_widget_role(self.btn_refresh_topics, "secondary")
        set_standard_icon(self.btn_refresh_topics, QStyle.StandardPixmap.SP_BrowserReload)
        layout.addWidget(self.btn_refresh_topics, 1, 3)

        layout.addWidget(QLabel("机器人 IP:"), 2, 0)
        self.ip_input = QLineEdit(settings.default_robot_ip)
        layout.addWidget(self.ip_input, 2, 1)

        layout.addWidget(QLabel("本机 IP:"), 2, 2)
        self.local_ip_label = QLabel(str(local_ip))
        set_widget_role(self.local_ip_label, "value")
        layout.addWidget(self.local_ip_label, 2, 3)

        layout.addWidget(QLabel("末端执行器:"), 3, 0)
        self.ee_combo = QComboBox()
        self.ee_combo.addItem("robotiq", "robotiq")
        self.ee_combo.addItem("qbsofthand", "qbsofthand")
        index = max(0, self.ee_combo.findData(settings.default_gripper_type))
        self.ee_combo.setCurrentIndex(index)
        layout.addWidget(self.ee_combo, 3, 1)

        self.btn_teleop_settings = QPushButton("遥操作设置")
        self.btn_teleop_settings.setMinimumHeight(button_height)
        set_widget_role(self.btn_teleop_settings, "secondary")
        layout.addWidget(self.btn_teleop_settings, 3, 2, 1, 2)

        self.input_hint_label = QLabel(self)
        self.input_hint_label.setWordWrap(True)
        set_widget_role(self.input_hint_label, "hint")
        self.input_hint_label.setVisible(False)
        layout.addWidget(self.input_hint_label, 4, 0, 1, 4)
        return group

    def _build_startup_group(
        self,
        settings: SystemControlSettings,
        *,
        section_style: str,
        button_height: int,
    ) -> QGroupBox:
        group = QGroupBox("启动节点")
        group.setStyleSheet(section_style)
        layout = QGridLayout(group)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)

        self.camera_driver_combo = QComboBox(self)
        for option in settings.camera_driver_options:
            self.camera_driver_combo.addItem(str(option), str(option))
        index = max(0, self.camera_driver_combo.findData(settings.default_camera_driver))
        self.camera_driver_combo.setCurrentIndex(index)
        self.camera_driver_combo.setVisible(False)

        self.btn_camera_driver = QPushButton("相机 ROS2 驱动（已停用）", self)
        self.btn_camera_driver.setCheckable(True)
        self.btn_camera_driver.setEnabled(False)
        self.btn_camera_driver.setVisible(False)
        set_widget_role(self.btn_camera_driver, "secondary")
        self.camera_module_hint_label = QLabel("相机 ROS2 驱动入口已停用。", self)
        self.camera_module_hint_label.setWordWrap(True)
        set_widget_role(self.camera_module_hint_label, "hint")
        self.camera_module_hint_label.setVisible(False)
        layout.addWidget(self.camera_module_hint_label, 2, 0, 1, 3)

        layout.addWidget(QLabel("机械臂 ROS2 驱动:"), 0, 0)
        self.btn_robot_driver = QPushButton("启动机械臂驱动")
        self.btn_robot_driver.setFixedHeight(button_height)
        self.btn_robot_driver.setCheckable(True)
        set_widget_role(self.btn_robot_driver, "secondary")
        set_standard_icon(self.btn_robot_driver, QStyle.StandardPixmap.SP_MediaPlay)
        layout.addWidget(self.btn_robot_driver, 0, 1, 1, 2)

        layout.addWidget(QLabel("遥操作系统:"), 1, 0)
        self.btn_teleop = QPushButton("启动遥操作系统")
        self.btn_teleop.setFixedHeight(button_height)
        self.btn_teleop.setCheckable(True)
        set_widget_role(self.btn_teleop, "primary")
        set_standard_icon(self.btn_teleop, QStyle.StandardPixmap.SP_MediaPlay)
        layout.addWidget(self.btn_teleop, 1, 1, 1, 2)

        self.startup_hint_label = QLabel(
            "当遥操作系统启动时，会接管机械臂驱动；GUI 会显示机械臂驱动为运行中，但不允许单独关闭。",
            self,
        )
        self.startup_hint_label.setWordWrap(True)
        set_widget_role(self.startup_hint_label, "hint")
        self.startup_hint_label.setVisible(False)
        layout.addWidget(self.startup_hint_label, 3, 0, 1, 3)
        return group

    def _build_home_group(
        self,
        *,
        section_style: str,
        button_height: int,
    ) -> QGroupBox:
        group = QGroupBox("回home操作")
        group.setStyleSheet(section_style)
        layout = QGridLayout(group)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)
        for column in range(3):
            layout.setColumnStretch(column, 1)

        self.btn_go_home = QPushButton("回 Home 点")
        self.btn_go_home.setFixedHeight(button_height)
        set_widget_role(self.btn_go_home, "warning")
        set_standard_icon(self.btn_go_home, QStyle.StandardPixmap.SP_DirHomeIcon)
        layout.addWidget(self.btn_go_home, 0, 0)

        self.btn_go_home_zone = QPushButton("回 Home Zone")
        self.btn_go_home_zone.setFixedHeight(button_height)
        set_widget_role(self.btn_go_home_zone, "warning")
        set_standard_icon(self.btn_go_home_zone, QStyle.StandardPixmap.SP_DialogResetButton)
        layout.addWidget(self.btn_go_home_zone, 0, 1)

        self.btn_set_home_current = QPushButton("设当前姿态为 Home")
        self.btn_set_home_current.setFixedHeight(button_height)
        set_widget_role(self.btn_set_home_current, "secondary")
        set_standard_icon(self.btn_set_home_current, QStyle.StandardPixmap.SP_DialogSaveButton)
        layout.addWidget(self.btn_set_home_current, 0, 2)
        return group
