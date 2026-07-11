"""Teleoperation settings dialogs kept separate from the main window."""

from __future__ import annotations

from typing import Optional, Protocol

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme import set_standard_icon, set_widget_role


class TeleopSettingsStorage(Protocol):
    def load_teleop_params(self, current_file: str):
        ...

    def save_teleop_params_overrides(self, current_file: str, teleop_updates, moveit_updates):
        ...

    def save_gui_settings_overrides(self, current_file: str, updates):
        ...

    def load_gui_settings(self, current_file: str):
        ...


class DefaultTeleopSettingsStorage:
    def load_teleop_params(self, current_file: str):
        from ..support import load_teleop_params

        return load_teleop_params(current_file)

    def save_teleop_params_overrides(self, current_file: str, teleop_updates, moveit_updates):
        from ..support import save_teleop_params_overrides

        return save_teleop_params_overrides(current_file, teleop_updates, moveit_updates)

    def save_gui_settings_overrides(self, current_file: str, updates):
        from ..support import save_gui_settings_overrides

        return save_gui_settings_overrides(current_file, updates)

    def load_gui_settings(self, current_file: str):
        from ..support import load_gui_settings

        return load_gui_settings(current_file)


class TeleopBindingCaptureDialog(QDialog):
    """Capture a joystick axis/button index for integer teleop mapping fields."""

    def __init__(self, parent: QWidget, *, capture_kind: str, joy_profile: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("录制控制器输入")
        self.setModal(True)
        self.resize(420, 150)
        self._capture_kind = str(capture_kind).strip().lower()
        self._joy_profile = str(joy_profile).strip().lower() or "auto"
        self._result_value: Optional[int] = None
        self._devices = []

        layout = QVBoxLayout(self)
        self._message = QLabel("按下手柄按钮或推动摇杆/扳机。也可以直接按数字键。")
        self._message.setWordWrap(True)
        layout.addWidget(self._message)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        cancel_button.setText("取消")
        set_widget_role(cancel_button, "secondary")
        set_standard_icon(cancel_button, QStyle.StandardPixmap.SP_DialogCancelButton)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._open_joystick_devices()
        if not self._devices:
            self._message.setText("没有读到可用手柄事件设备；可以直接按数字键填入索引。")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_joystick_devices)
        self._timer.start(30)
        self.setFocusPolicy(Qt.StrongFocus)

    @property
    def result_value(self) -> Optional[int]:
        return self._result_value

    def _open_joystick_devices(self) -> None:
        try:
            from evdev import InputDevice, ecodes, list_devices  # type: ignore
            from teleop_control_py.hardware.input.joy_device_profiles import build_profiles, infer_profile_key
        except Exception as exc:
            self._message.setText(f"无法启用手柄录制: {exc}")
            return

        profiles = build_profiles(0.05)
        requested = self._joy_profile
        for device_path in list_devices():
            try:
                device = InputDevice(device_path)
                capabilities = device.capabilities()
                has_button = ecodes.EV_KEY in capabilities
                has_axis = ecodes.EV_ABS in capabilities
                if not has_button and not has_axis:
                    device.close()
                    continue
                profile_key = infer_profile_key(device.name, requested)
                profile = profiles.get(profile_key, profiles.get(requested, profiles["generic"]))
                self._devices.append((device, profile, ecodes))
            except Exception:
                continue

    def _poll_joystick_devices(self) -> None:
        for device, profile, ecodes in list(self._devices):
            try:
                events = list(device.read())
            except BlockingIOError:
                continue
            except Exception:
                continue
            for event in events:
                if self._capture_kind == "button" and event.type == ecodes.EV_KEY and int(event.value) == 1:
                    index = profile.button_indices.get(event.code)
                    if index is not None:
                        self._accept_value(int(index))
                        return
                if self._capture_kind == "axis" and event.type == ecodes.EV_ABS:
                    spec = profile.axis_specs.get(event.code)
                    if spec is not None:
                        self._accept_value(int(spec.index))
                        return

    def _accept_value(self, value: int) -> None:
        self._result_value = int(value)
        self.accept()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        text = event.text().strip()
        if text.lstrip("-").isdigit():
            self._accept_value(int(text))
            return
        self._message.setText("当前字段需要整数索引；按数字键，或按下手柄按钮/推动摇杆。")

    def _close_devices(self) -> None:
        for device, _profile, _ecodes in self._devices:
            try:
                device.close()
            except Exception:
                pass
        self._devices = []

    def done(self, result: int) -> None:  # type: ignore[override]
        self._close_devices()
        super().done(result)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._close_devices()
        super().closeEvent(event)


class TeleopSettingsDialog(QDialog):
    MODES = ("joy", "mediapipe", "quest3")
    MODE_LABELS = {
        "joy": "Joy 手柄",
        "mediapipe": "MediaPipe 手势",
        "quest3": "Quest 3 VR",
    }

    FIELD_GROUPS = {
        "joy": [
            (
                "速度与滤波",
                [
                    {"key": "max_linear_vel", "label": "XY 最大速度", "type": "float", "min": 0.05, "max": 5.0, "step": 0.05, "decimals": 2, "default": 2.0},
                    {"key": "max_angular_vel", "label": "旋转最大速度", "type": "float", "min": 0.05, "max": 10.0, "step": 0.05, "decimals": 2, "default": 3.0},
                    {"key": "max_linear_accel", "label": "平移加速度", "type": "float", "min": 0.05, "max": 30.0, "step": 0.05, "decimals": 2, "default": 4.5},
                    {"key": "max_angular_accel", "label": "旋转加速度", "type": "float", "min": 0.05, "max": 50.0, "step": 0.1, "decimals": 2, "default": 12.0},
                    {"key": "zero_return_accel_scale", "label": "松手停止倍率", "type": "float", "min": 0.01, "max": 2.0, "step": 0.05, "decimals": 2, "default": 0.65},
                    {"key": "linear_z_scale", "label": "Z 轴速度倍率", "type": "float", "min": 0.05, "max": 2.0, "step": 0.05, "decimals": 2, "default": 0.5},
                    {"key": "linear_z_zero_return_accel_scale", "label": "Z 松手停止倍率", "type": "float", "min": 0.01, "max": 2.0, "step": 0.05, "decimals": 2, "default": 0.65},
                    {"key": "joy_deadzone", "label": "摇杆死区", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "decimals": 3, "default": 0.08},
                    {"key": "joy_curve", "label": "摇杆曲线", "type": "combo", "options": [("cubic", "cubic"), ("linear", "linear")], "default": "cubic"},
                    {"key": "moveit_servo.low_pass_filter_coeff", "label": "Servo 低通系数", "type": "float", "min": 0.0, "max": 100.0, "step": 0.5, "decimals": 2, "default": 10.0},
                ],
            ),
            (
                "轴与按钮",
                [
                    {"key": "linear_x_axis", "label": "X 平移轴", "type": "int", "capture": "axis", "default": 0},
                    {"key": "linear_y_axis", "label": "Y 平移轴", "type": "int", "capture": "axis", "default": 1},
                    {"key": "linear_z_axis", "label": "Z 平移轴", "type": "int", "capture": "axis", "default": -1},
                    {"key": "linear_z_up_axis", "label": "Z 上升轴", "type": "int", "capture": "axis", "default": 4},
                    {"key": "linear_z_down_axis", "label": "Z 下降轴", "type": "int", "capture": "axis", "default": 5},
                    {"key": "angular_x_axis", "label": "Rx 旋转轴", "type": "int", "capture": "axis", "default": 3},
                    {"key": "angular_y_axis", "label": "Ry 旋转轴", "type": "int", "capture": "axis", "default": 2},
                    {"key": "angular_z_axis", "label": "Rz 旋转轴", "type": "int", "capture": "axis", "default": -1},
                    {"key": "angular_z_positive_button", "label": "Rz 正向按钮", "type": "int", "capture": "button", "default": 1},
                    {"key": "angular_z_negative_button", "label": "Rz 反向按钮", "type": "int", "capture": "button", "default": 0},
                    {"key": "gripper_close_button", "label": "夹爪闭合按钮", "type": "int", "capture": "button", "default": 5},
                    {"key": "gripper_open_button", "label": "夹爪打开按钮", "type": "int", "capture": "button", "default": 4},
                    {"key": "gripper_axis", "label": "夹爪轴", "type": "int", "capture": "axis", "default": -1},
                    {"key": "joy_deadman_enabled", "label": "启用 Deadman", "type": "bool", "default": False},
                    {"key": "deadman_button", "label": "Deadman 按钮", "type": "int", "capture": "button", "default": -1},
                    {"key": "deadman_axis", "label": "Deadman 轴", "type": "int", "capture": "axis", "default": 4},
                ],
            ),
        ],
        "mediapipe": [
            (
                "速度与姿态",
                [
                    {"key": "mediapipe_motion_mode", "label": "运动模式", "type": "combo", "options": [("target_pose", "target_pose"), ("velocity", "velocity")], "default": "target_pose"},
                    {"key": "mediapipe_max_linear_vel", "label": "最大平移速度", "type": "float", "min": 0.05, "max": 5.0, "step": 0.05, "decimals": 2, "default": 2.8},
                    {"key": "mediapipe_max_angular_vel", "label": "最大旋转速度", "type": "float", "min": 0.05, "max": 10.0, "step": 0.05, "decimals": 2, "default": 5.0},
                    {"key": "mediapipe_max_linear_accel", "label": "平移加速度", "type": "float", "min": 0.05, "max": 30.0, "step": 0.1, "decimals": 2, "default": 10.0},
                    {"key": "mediapipe_max_angular_accel", "label": "旋转加速度", "type": "float", "min": 0.05, "max": 50.0, "step": 0.1, "decimals": 2, "default": 18.0},
                    {"key": "mediapipe_linear_scale", "label": "平移缩放", "type": "float", "min": 0.05, "max": 8.0, "step": 0.05, "decimals": 2, "default": 2.8},
                    {"key": "mediapipe_angular_scale", "label": "旋转缩放", "type": "float", "min": 0.05, "max": 8.0, "step": 0.05, "decimals": 2, "default": 1.0},
                    {"key": "mediapipe_position_linear_gain", "label": "位置平移增益", "type": "float", "min": 0.1, "max": 30.0, "step": 0.1, "decimals": 2, "default": 7.0},
                    {"key": "mediapipe_position_angular_gain", "label": "位置旋转增益", "type": "float", "min": 0.1, "max": 30.0, "step": 0.1, "decimals": 2, "default": 5.0},
                    {"key": "mediapipe_deadzone", "label": "手势死区", "type": "float", "min": 0.0, "max": 0.2, "step": 0.005, "decimals": 3, "default": 0.02},
                    {"key": "mediapipe_smoothing_alpha", "label": "输入平滑 alpha", "type": "float", "min": 0.01, "max": 1.0, "step": 0.01, "decimals": 2, "default": 0.45},
                ],
            ),
            (
                "手势与夹爪",
                [
                    {"key": "mediapipe_hand_position_source", "label": "手部位置来源", "type": "combo", "options": [("depth", "depth"), ("normalized", "normalized"), ("hybrid", "hybrid")], "default": "depth"},
                    {"key": "mediapipe_orientation_mode", "label": "姿态模式", "type": "combo", "options": [("lock", "lock"), ("hand_relative", "hand_relative"), ("absolute", "absolute")], "default": "lock"},
                    {"key": "mediapipe_gripper_open_dist_m", "label": "夹爪打开距离(m)", "type": "float", "min": 0.001, "max": 0.3, "step": 0.005, "decimals": 3, "default": 0.10},
                    {"key": "mediapipe_gripper_close_dist_m", "label": "夹爪闭合距离(m)", "type": "float", "min": 0.001, "max": 0.3, "step": 0.005, "decimals": 3, "default": 0.03},
                    {"key": "mediapipe_gripper_metric_hold_sec", "label": "夹爪保持时间", "type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "decimals": 2, "default": 0.25},
                    {"key": "mediapipe_gripper_requires_deadman", "label": "夹爪需要 Deadman", "type": "bool", "default": True},
                    {"key": "mediapipe_deadman_filter_enabled", "label": "Deadman 滤波", "type": "bool", "default": True},
                    {"key": "mediapipe_deadman_engage_confirm_sec", "label": "Deadman 触发确认", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2, "default": 0.10},
                    {"key": "mediapipe_deadman_release_confirm_sec", "label": "Deadman 释放确认", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2, "default": 0.03},
                    {"key": "mediapipe_space_deadman_hold_sec", "label": "空格 Deadman 保持", "type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "decimals": 2, "default": 0.3},
                    {"key": "mediapipe_show_debug_window", "label": "显示调试窗口", "type": "bool", "default": True},
                ],
            ),
        ],
        "quest3": [
            (
                "速度与位姿",
                [
                    {"key": "quest3_active_hand", "label": "控制手", "type": "combo", "options": [("right", "right"), ("left", "left")], "default": "right"},
                    {"key": "quest3_motion_mode", "label": "运动模式", "type": "combo", "options": [("target_pose", "target_pose"), ("velocity", "velocity")], "default": "target_pose"},
                    {"key": "quest3_max_linear_vel", "label": "最大平移速度", "type": "float", "min": 0.05, "max": 5.0, "step": 0.05, "decimals": 2, "default": 2.8},
                    {"key": "quest3_max_angular_vel", "label": "最大旋转速度", "type": "float", "min": 0.05, "max": 10.0, "step": 0.05, "decimals": 2, "default": 5.0},
                    {"key": "quest3_max_linear_accel", "label": "平移加速度", "type": "float", "min": 0.05, "max": 40.0, "step": 0.1, "decimals": 2, "default": 14.0},
                    {"key": "quest3_max_angular_accel", "label": "旋转加速度", "type": "float", "min": 0.05, "max": 60.0, "step": 0.1, "decimals": 2, "default": 24.0},
                    {"key": "quest3_linear_scale", "label": "平移缩放", "type": "float", "min": 0.05, "max": 5.0, "step": 0.05, "decimals": 2, "default": 1.0},
                    {"key": "quest3_angular_scale", "label": "旋转缩放", "type": "float", "min": 0.05, "max": 5.0, "step": 0.05, "decimals": 2, "default": 1.0},
                    {"key": "quest3_position_linear_gain", "label": "位置平移增益", "type": "float", "min": 0.1, "max": 30.0, "step": 0.1, "decimals": 2, "default": 10.0},
                    {"key": "quest3_position_angular_gain", "label": "位置旋转增益", "type": "float", "min": 0.1, "max": 30.0, "step": 0.1, "decimals": 2, "default": 8.0},
                    {"key": "quest3_deadzone", "label": "位移死区", "type": "float", "min": 0.0, "max": 0.1, "step": 0.001, "decimals": 3, "default": 0.005},
                    {"key": "quest3_enable_input_smoothing", "label": "输入平滑", "type": "bool", "default": False},
                    {"key": "quest3_smoothing_alpha", "label": "平滑 alpha", "type": "float", "min": 0.01, "max": 1.0, "step": 0.01, "decimals": 2, "default": 0.45},
                    {"key": "quest3_max_relative_orientation_deg", "label": "最大相对姿态(deg)", "type": "float", "min": 1.0, "max": 180.0, "step": 1.0, "decimals": 1, "default": 90.0},
                ],
            ),
            (
                "Clutch 与按钮",
                [
                    {"key": "quest3_clutch_filter_enabled", "label": "Clutch 滤波", "type": "bool", "default": True},
                    {"key": "quest3_clutch_engage_confirm_sec", "label": "Clutch 触发确认", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2, "default": 0.02},
                    {"key": "quest3_clutch_release_confirm_sec", "label": "Clutch 释放确认", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2, "default": 0.02},
                    {"key": "quest3_clutch_axis_threshold", "label": "Clutch 轴阈值", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2, "default": 0.15},
                    {"key": "quest3_left_clutch_axis", "label": "左手 Clutch 轴", "type": "int", "default": 1},
                    {"key": "quest3_right_clutch_axis", "label": "右手 Clutch 轴", "type": "int", "default": 7},
                    {"key": "quest3_left_clutch_button", "label": "左手 Clutch 按钮", "type": "int", "default": 1},
                    {"key": "quest3_right_clutch_button", "label": "右手 Clutch 按钮", "type": "int", "default": 7},
                    {"key": "quest3_left_trigger_axis", "label": "左扳机轴", "type": "int", "default": 0},
                    {"key": "quest3_right_trigger_axis", "label": "右扳机轴", "type": "int", "default": 6},
                    {"key": "quest3_left_trigger_button", "label": "左扳机按钮", "type": "int", "default": 0},
                    {"key": "quest3_right_trigger_button", "label": "右扳机按钮", "type": "int", "default": 6},
                    {"key": "quest3_gripper_requires_clutch", "label": "夹爪需要 Clutch", "type": "bool", "default": True},
                    {"key": "quest3_frame_reset_enabled", "label": "启用 Frame Reset", "type": "bool", "default": False},
                    {"key": "quest3_frame_reset_scope", "label": "Frame Reset 范围", "type": "combo", "options": [("active_hand", "active_hand"), ("both", "both")], "default": "active_hand"},
                    {"key": "quest3_frame_reset_hold_sec", "label": "Frame Reset 长按", "type": "float", "min": 0.0, "max": 3.0, "step": 0.05, "decimals": 2, "default": 0.75},
                    {"key": "quest3_left_frame_reset_buttons", "label": "左手 Reset 按钮", "type": "list_int", "default": [4, 5]},
                    {"key": "quest3_right_frame_reset_buttons", "label": "右手 Reset 按钮", "type": "list_int", "default": [10, 11]},
                ],
            ),
        ],
    }

    def __init__(
        self,
        parent: QWidget,
        *,
        initial_mode: str,
        storage: Optional[TeleopSettingsStorage] = None,
    ) -> None:
        super().__init__(parent)
        self._parent_window = parent
        self._storage = storage or DefaultTeleopSettingsStorage()
        self.setWindowTitle("遥操作设置")
        self.setModal(True)
        self.setMinimumSize(480, 360)
        self.resize(760, 680)

        self._teleop_params, self._moveit_params = self._storage.load_teleop_params(__file__)
        self._field_widgets: dict[str, dict[str, QWidget]] = {mode: {} for mode in self.MODES}
        self._preset_combos: dict[str, QComboBox] = {}
        self._delete_buttons: dict[str, QPushButton] = {}
        self._preset_store = self._normalize_preset_store(getattr(parent, "gui_settings").teleop_settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("teleopSettingsTabs")
        layout.addWidget(self.tabs, 1)

        for mode in self.MODES:
            self.tabs.addTab(self._build_mode_page(mode), self.MODE_LABELS[mode])

        initial_index = self.MODES.index(initial_mode) if initial_mode in self.MODES else 0
        self.tabs.setCurrentIndex(initial_index)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close)
        self.close_button = self.button_box.button(QDialogButtonBox.Close)
        self.close_button.setText("关闭")
        set_widget_role(self.close_button, "secondary")
        set_standard_icon(self.close_button, QStyle.StandardPixmap.SP_DialogCloseButton)

        self.apply_button = self.button_box.addButton("应用", QDialogButtonBox.ApplyRole)
        set_widget_role(self.apply_button, "primary")
        set_standard_icon(self.apply_button, QStyle.StandardPixmap.SP_DialogApplyButton)
        self.apply_button.clicked.connect(self.apply_settings)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _all_specs(self, mode: str) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for _title, group_specs in self.FIELD_GROUPS.get(mode, []):
            specs.extend(group_specs)
        return specs

    def _source_value(self, key: str, default: object) -> object:
        if key.startswith("moveit_servo."):
            return self._moveit_params.get(key.split(".", 1)[1], default)
        return self._teleop_params.get(key, default)

    def _as_bool(self, value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(default)

    def _normalize_value(self, spec: dict[str, object], value: object) -> object:
        field_type = str(spec.get("type", "float"))
        default = spec.get("default", 0.0)
        try:
            if field_type == "float":
                return float(value)
            if field_type == "int":
                return int(value)
            if field_type == "bool":
                return self._as_bool(value, bool(default))
            if field_type == "list_int":
                if isinstance(value, (list, tuple)):
                    return [int(v) for v in value]
                return [int(v.strip()) for v in str(value).split(",") if v.strip()]
            if field_type == "combo":
                valid_values = {str(item[1]) for item in spec.get("options", [])}  # type: ignore[arg-type]
                normalized = str(value).strip()
                return normalized if normalized in valid_values else str(default)
        except Exception:
            return default
        return value

    def _default_params_for_mode(self, mode: str) -> dict[str, object]:
        defaults: dict[str, object] = {}
        for spec in self._all_specs(mode):
            key = str(spec["key"])
            defaults[key] = self._normalize_value(spec, self._source_value(key, spec.get("default")))
        return defaults

    def _merge_known_params(self, mode: str, values: object) -> dict[str, object]:
        merged = self._default_params_for_mode(mode)
        if not isinstance(values, dict):
            return merged
        for spec in self._all_specs(mode):
            key = str(spec["key"])
            if key in values:
                merged[key] = self._normalize_value(spec, values.get(key))
        return merged

    def _normalize_preset_store(self, raw: object) -> dict[str, object]:
        raw_store = raw if isinstance(raw, dict) else {}
        raw_presets = raw_store.get("presets", {}) if isinstance(raw_store.get("presets", {}), dict) else {}
        raw_active = raw_store.get("active_preset_by_input", {}) if isinstance(raw_store.get("active_preset_by_input", {}), dict) else {}
        presets: dict[str, dict[str, dict[str, object]]] = {}
        active: dict[str, str] = {}

        for mode in self.MODES:
            mode_presets: dict[str, dict[str, object]] = {}
            raw_mode_presets = raw_presets.get(mode, {}) if isinstance(raw_presets, dict) else {}
            if isinstance(raw_mode_presets, dict):
                for name, entry in raw_mode_presets.items():
                    preset_name = str(name).strip()
                    if not preset_name:
                        continue
                    entry_dict = entry if isinstance(entry, dict) else {}
                    params_source = entry_dict.get("params", entry_dict)
                    mode_presets[preset_name] = {
                        "locked": bool(entry_dict.get("locked", preset_name == "default")),
                        "params": self._merge_known_params(mode, params_source),
                    }
            if "default" not in mode_presets:
                mode_presets["default"] = {"locked": True, "params": self._default_params_for_mode(mode)}
            else:
                mode_presets["default"]["locked"] = True

            active_name = str(raw_active.get(mode, "default")).strip() if isinstance(raw_active, dict) else "default"
            if active_name not in mode_presets:
                active_name = "default"
            presets[mode] = mode_presets
            active[mode] = active_name

        return {"presets": presets, "active_preset_by_input": active}

    def _build_mode_page(self, mode: str) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName(f"{mode}_settings_scroll")
        scroll.setWidgetResizable(True)
        scroll.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        preset_group = QGroupBox("参数方案")
        preset_layout = QGridLayout(preset_group)
        preset_layout.setColumnStretch(1, 1)
        preset_layout.addWidget(QLabel("当前方案:"), 0, 0)
        combo = QComboBox()
        self._preset_combos[mode] = combo
        preset_layout.addWidget(combo, 0, 1)

        save_as_button = QPushButton("另存方案")
        set_widget_role(save_as_button, "secondary")
        set_standard_icon(save_as_button, QStyle.StandardPixmap.SP_DialogSaveButton)
        save_as_button.clicked.connect(lambda _checked=False, m=mode: self._save_preset_as(m))
        preset_layout.addWidget(save_as_button, 0, 2)

        delete_button = QPushButton("删除")
        set_widget_role(delete_button, "danger-secondary")
        set_standard_icon(delete_button, QStyle.StandardPixmap.SP_TrashIcon)
        delete_button.clicked.connect(lambda _checked=False, m=mode: self._delete_current_preset(m))
        self._delete_buttons[mode] = delete_button
        preset_layout.addWidget(delete_button, 0, 3)
        layout.addWidget(preset_group)

        for title, specs in self.FIELD_GROUPS.get(mode, []):
            group = QGroupBox(title)
            grid = QGridLayout(group)
            grid.setColumnStretch(1, 1)
            for row, spec in enumerate(specs):
                grid.addWidget(QLabel(str(spec.get("label", spec.get("key", "")))), row, 0)
                editor, value_widget = self._make_editor(mode, spec)
                self._field_widgets[mode][str(spec["key"])] = value_widget
                grid.addWidget(editor, row, 1)
            layout.addWidget(group)

        layout.addStretch(1)
        self._refresh_preset_combo(mode)
        combo.currentIndexChanged.connect(lambda _index, m=mode: self._on_preset_changed(m))
        scroll.setWidget(page)
        return scroll

    def _make_editor(self, mode: str, spec: dict[str, object]) -> tuple[QWidget, QWidget]:
        field_type = str(spec.get("type", "float"))
        if field_type == "float":
            spin = QDoubleSpinBox()
            spin.setRange(float(spec.get("min", -9999.0)), float(spec.get("max", 9999.0)))
            spin.setDecimals(int(spec.get("decimals", 2)))
            spin.setSingleStep(float(spec.get("step", 0.1)))
            return spin, spin
        if field_type == "int":
            spin = QSpinBox()
            spin.setRange(int(spec.get("min", -1)), int(spec.get("max", 64)))
            capture_kind = str(spec.get("capture", "")).strip()
            if capture_kind:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                row_layout.addWidget(spin, 1)
                capture_button = QPushButton("录制")
                set_widget_role(capture_button, "secondary")
                set_standard_icon(capture_button, QStyle.StandardPixmap.SP_MediaPlay)
                capture_button.clicked.connect(lambda _checked=False, s=spin, k=capture_kind: self._capture_binding(s, k))
                row_layout.addWidget(capture_button)
                return row, spin
            return spin, spin
        if field_type == "bool":
            checkbox = QCheckBox()
            checkbox.setAccessibleName(str(spec.get("label", spec.get("key", "布尔选项"))))
            return checkbox, checkbox
        if field_type == "combo":
            combo = QComboBox()
            for label, value in spec.get("options", []):  # type: ignore[assignment]
                combo.addItem(str(label), str(value))
            return combo, combo
        if field_type == "list_int":
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("例如: 4,5")
            return line_edit, line_edit
        line_edit = QLineEdit()
        return line_edit, line_edit

    def _capture_binding(self, target: QSpinBox, capture_kind: str) -> None:
        joy_profile = "auto"
        selected = getattr(self._parent_window, "_selected_joy_profile", None)
        if callable(selected):
            try:
                joy_profile = str(selected())
            except Exception:
                joy_profile = "auto"
        dialog = TeleopBindingCaptureDialog(self, capture_kind=capture_kind, joy_profile=joy_profile)
        if dialog.exec() == QDialog.Accepted and dialog.result_value is not None:
            target.setValue(int(dialog.result_value))

    def _refresh_preset_combo(self, mode: str) -> None:
        combo = self._preset_combos[mode]
        combo.blockSignals(True)
        combo.clear()
        presets = self._preset_store["presets"][mode]  # type: ignore[index]
        for name in presets.keys():
            combo.addItem(str(name), str(name))
        active = self._preset_store["active_preset_by_input"].get(mode, "default")  # type: ignore[index]
        index = combo.findData(active)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)
        self._load_current_preset_to_widgets(mode)
        self._update_delete_button(mode)

    def _current_preset_name(self, mode: str) -> str:
        combo = self._preset_combos[mode]
        value = combo.currentData()
        return str(value).strip() if value is not None else "default"

    def _current_preset_entry(self, mode: str) -> dict[str, object]:
        presets = self._preset_store["presets"][mode]  # type: ignore[index]
        name = self._current_preset_name(mode)
        return presets.get(name, presets["default"])

    def _on_preset_changed(self, mode: str) -> None:
        self._preset_store["active_preset_by_input"][mode] = self._current_preset_name(mode)  # type: ignore[index]
        self._load_current_preset_to_widgets(mode)
        self._update_delete_button(mode)

    def _update_delete_button(self, mode: str) -> None:
        entry = self._current_preset_entry(mode)
        self._delete_buttons[mode].setEnabled(not bool(entry.get("locked", False)))

    def _load_current_preset_to_widgets(self, mode: str) -> None:
        entry = self._current_preset_entry(mode)
        params = entry.get("params", {}) if isinstance(entry, dict) else {}
        self._load_params_to_widgets(mode, params)

    def _load_params_to_widgets(self, mode: str, params: object) -> None:
        values = params if isinstance(params, dict) else {}
        for spec in self._all_specs(mode):
            key = str(spec["key"])
            widget = self._field_widgets[mode].get(key)
            if widget is None:
                continue
            value = self._normalize_value(spec, values.get(key, spec.get("default")))
            field_type = str(spec.get("type", "float"))
            if field_type == "float" and isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif field_type == "int" and isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif field_type == "bool" and isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif field_type == "combo" and isinstance(widget, QComboBox):
                index = widget.findData(str(value))
                widget.setCurrentIndex(index if index >= 0 else 0)
            elif field_type == "list_int" and isinstance(widget, QLineEdit):
                widget.setText(",".join(str(v) for v in value))

    def _collect_mode_params(self, mode: str) -> dict[str, object]:
        params: dict[str, object] = {}
        for spec in self._all_specs(mode):
            key = str(spec["key"])
            widget = self._field_widgets[mode].get(key)
            field_type = str(spec.get("type", "float"))
            if isinstance(widget, QDoubleSpinBox):
                value: object = float(widget.value())
            elif isinstance(widget, QSpinBox):
                value = int(widget.value())
            elif isinstance(widget, QCheckBox):
                value = bool(widget.isChecked())
            elif isinstance(widget, QComboBox):
                value = str(widget.currentData())
            elif isinstance(widget, QLineEdit) and field_type == "list_int":
                value = [int(v.strip()) for v in widget.text().split(",") if v.strip()]
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
            else:
                value = spec.get("default")
            params[key] = self._normalize_value(spec, value)
        return params

    def _save_preset_as(self, mode: str) -> None:
        name, ok = QInputDialog.getText(self, "新增遥操作方案", "方案名称:")
        if not ok:
            return
        preset_name = str(name).strip()
        if not preset_name:
            return
        presets = self._preset_store["presets"][mode]  # type: ignore[index]
        if preset_name in presets:
            reply = QMessageBox.question(self, "覆盖方案", f"方案 `{preset_name}` 已存在，是否覆盖？")
            if reply != QMessageBox.Yes:
                return
        presets[preset_name] = {"locked": preset_name == "default", "params": self._collect_mode_params(mode)}
        self._preset_store["active_preset_by_input"][mode] = preset_name  # type: ignore[index]
        self._persist_preset_store()
        self._refresh_preset_combo(mode)

    def _delete_current_preset(self, mode: str) -> None:
        name = self._current_preset_name(mode)
        entry = self._current_preset_entry(mode)
        if bool(entry.get("locked", False)):
            return
        reply = QMessageBox.question(self, "删除遥操作方案", f"确定删除 `{name}`？")
        if reply != QMessageBox.Yes:
            return
        presets = self._preset_store["presets"][mode]  # type: ignore[index]
        presets.pop(name, None)
        self._preset_store["active_preset_by_input"][mode] = "default"  # type: ignore[index]
        self._persist_preset_store()
        self._refresh_preset_combo(mode)

    def _persist_preset_store(self) -> None:
        self._storage.save_gui_settings_overrides(__file__, {"teleop_settings": self._preset_store})
        parent = self._parent_window
        if hasattr(parent, "gui_settings"):
            parent.gui_settings = self._storage.load_gui_settings(__file__)

    def apply_settings(self) -> None:
        teleop_updates: dict[str, object] = {}
        moveit_updates: dict[str, object] = {}
        for mode in self.MODES:
            params = self._collect_mode_params(mode)
            preset_name = self._current_preset_name(mode)
            presets = self._preset_store["presets"][mode]  # type: ignore[index]
            if preset_name not in presets:
                preset_name = "default"
            presets[preset_name]["params"] = params
            self._preset_store["active_preset_by_input"][mode] = preset_name  # type: ignore[index]
            for key, value in params.items():
                if key.startswith("moveit_servo."):
                    moveit_updates[key.split(".", 1)[1]] = value
                else:
                    teleop_updates[key] = value

        config_path = self._storage.save_teleop_params_overrides(__file__, teleop_updates, moveit_updates)
        self._persist_preset_store()

        parent = self._parent_window
        if hasattr(parent, "log"):
            parent.log(f"已保存遥操作设置: {config_path}")
        runtime = getattr(parent, "runtime_facade", None)
        teleop_running = bool(runtime is not None and runtime.is_process_running("teleop"))
        if teleop_running:
            QMessageBox.information(self, "遥操作设置", "设置已保存。当前遥操作进程运行中，重启遥操作系统后生效。")
        else:
            QMessageBox.information(self, "遥操作设置", "设置已保存，下次启动遥操作系统时生效。")

__all__ = [
    "DefaultTeleopSettingsStorage",
    "TeleopBindingCaptureDialog",
    "TeleopSettingsDialog",
    "TeleopSettingsStorage",
]
