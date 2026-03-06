#!/usr/bin/env python3
"""输入策略层：不同输入源统一输出标准 Twist 与夹爪值。"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Sequence

from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

from .transform_utils import _clamp, apply_deadzone, map_axis_linear, map_axis_nonlinear


def _zero_twist() -> Twist:
    twist = Twist()
    twist.linear.x = 0.0
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0
    return twist


class InputHandlerBase(ABC):
    """输入策略统一接口。"""

    def __init__(self, node: Node) -> None:
        self.node = node
        self._lock = threading.Lock()
        self._latest_twist = _zero_twist()
        self._latest_gripper = 0.0

    @abstractmethod
    def get_command(self) -> tuple[Twist, float]:
        """返回标准化控制命令：(twist, gripper_value)。"""


class JoyInputHandler(InputHandlerBase):
    """Joy 手柄输入策略，内部完成映射和缓存。"""

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self._deadzone = float(node.get_parameter("joy_deadzone").value)
        self._curve = str(node.get_parameter("joy_curve").value).strip().lower()
        self._joy_topic = str(node.get_parameter("joy_topic").value)

        self._max_linear = float(node.get_parameter("max_linear_vel").value)
        self._max_angular = float(node.get_parameter("max_angular_vel").value)

        self._linear_axes = [
            int(node.get_parameter("linear_x_axis").value),
            int(node.get_parameter("linear_y_axis").value),
            int(node.get_parameter("linear_z_axis").value),
        ]
        self._linear_z_up_button = int(node.get_parameter("linear_z_up_button").value)
        self._linear_z_down_button = int(node.get_parameter("linear_z_down_button").value)
        self._angular_axes = [
            int(node.get_parameter("angular_x_axis").value),
            int(node.get_parameter("angular_y_axis").value),
            int(node.get_parameter("angular_z_axis").value),
        ]
        self._angular_z_positive_button = int(node.get_parameter("angular_z_positive_button").value)
        self._angular_z_negative_button = int(node.get_parameter("angular_z_negative_button").value)
        self._linear_sign = [float(value) for value in node.get_parameter("linear_axis_sign").value]
        self._angular_sign = [float(value) for value in node.get_parameter("angular_axis_sign").value]

        self._deadman_button = int(node.get_parameter("deadman_button").value)
        self._deadman_axis = int(node.get_parameter("deadman_axis").value)
        self._deadman_axis_threshold = float(node.get_parameter("deadman_axis_threshold").value)
        self._deadman_enabled = bool(node.get_parameter("joy_deadman_enabled").value)

        self._gripper_axis = int(node.get_parameter("gripper_axis").value)
        self._gripper_close_button = int(node.get_parameter("gripper_close_button").value)
        self._gripper_open_button = int(node.get_parameter("gripper_open_button").value)
        self._gripper_axis_inverted = bool(node.get_parameter("gripper_axis_inverted").value)

        node.create_subscription(Joy, self._joy_topic, self._joy_callback, 20)

    def _axis_value(self, values: Sequence[float], index: int) -> float:
        if index < 0 or index >= len(values):
            return 0.0
        raw = float(values[index])
        if self._curve == "cubic":
            return map_axis_nonlinear(raw, deadzone=self._deadzone, exponent=3.0, scale=1.0)
        return map_axis_linear(raw, deadzone=self._deadzone, scale=1.0)

    def _button_value(self, values: Sequence[int], index: int) -> bool:
        return 0 <= index < len(values) and bool(values[index])

    def _button_axis(self, buttons: Sequence[int], positive_button: int, negative_button: int) -> float:
        positive = self._button_value(buttons, positive_button)
        negative = self._button_value(buttons, negative_button)
        if positive and not negative:
            return 1.0
        if negative and not positive:
            return -1.0
        return 0.0

    def _deadman_active(self, axes: Sequence[float], buttons: Sequence[int]) -> bool:
        if not self._deadman_enabled:
            return True
        if self._deadman_button >= 0 and self._button_value(buttons, self._deadman_button):
            return True
        if self._deadman_axis >= 0 and self._deadman_axis < len(axes):
            return float(axes[self._deadman_axis]) >= self._deadman_axis_threshold
        return False

    def _joy_callback(self, msg: Joy) -> None:
        axes = list(msg.axes)
        buttons = list(msg.buttons)
        twist = _zero_twist()

        if self._deadman_active(axes, buttons):
            linear_z = self._axis_value(axes, self._linear_axes[2])
            if self._linear_axes[2] < 0:
                linear_z = self._button_axis(buttons, self._linear_z_up_button, self._linear_z_down_button)

            angular_z = self._axis_value(axes, self._angular_axes[2])
            if self._angular_axes[2] < 0:
                angular_z = self._button_axis(
                    buttons,
                    self._angular_z_positive_button,
                    self._angular_z_negative_button,
                )

            linear = [
                self._axis_value(axes, self._linear_axes[0]) * self._linear_sign[0] * self._max_linear,
                self._axis_value(axes, self._linear_axes[1]) * self._linear_sign[1] * self._max_linear,
                linear_z * self._linear_sign[2] * self._max_linear,
            ]
            angular = [
                self._axis_value(axes, self._angular_axes[0]) * self._angular_sign[0] * self._max_angular,
                self._axis_value(axes, self._angular_axes[1]) * self._angular_sign[1] * self._max_angular,
                angular_z * self._angular_sign[2] * self._max_angular,
            ]
            twist.linear.x, twist.linear.y, twist.linear.z = linear
            twist.angular.x, twist.angular.y, twist.angular.z = angular

        with self._lock:
            self._latest_twist = twist
            if self._gripper_axis >= 0 and self._gripper_axis < len(axes):
                axis_val = self._axis_value(axes, self._gripper_axis)
                gripper = axis_val if axis_val >= 0.0 else 0.5 * (axis_val + 1.0)
                if self._gripper_axis_inverted:
                    gripper = 1.0 - gripper
                self._latest_gripper = float(_clamp(gripper, 0.0, 1.0))
            else:
                if self._button_value(buttons, self._gripper_close_button):
                    self._latest_gripper = 1.0
                elif self._button_value(buttons, self._gripper_open_button):
                    self._latest_gripper = 0.0

    def get_command(self) -> tuple[Twist, float]:
        with self._lock:
            twist = Twist()
            twist.linear.x = self._latest_twist.linear.x
            twist.linear.y = self._latest_twist.linear.y
            twist.linear.z = self._latest_twist.linear.z
            twist.angular.x = self._latest_twist.angular.x
            twist.angular.y = self._latest_twist.angular.y
            twist.angular.z = self._latest_twist.angular.z
            return twist, float(self._latest_gripper)


class MediaPipeInputHandler(InputHandlerBase):
    """MediaPipe 输入策略。

    默认订阅 `Float32MultiArray`，约定数据格式为：
    `[vx, vy, vz, wx, wy, wz, gripper]`
    可选第 8 个值作为 deadman 标志，<=0 时当前 Twist 会被清零。
    """

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self._topic = str(node.get_parameter("mediapipe_topic").value)
        self._deadzone = float(node.get_parameter("mediapipe_deadzone").value)
        self._linear_scale = float(node.get_parameter("mediapipe_linear_scale").value)
        self._angular_scale = float(node.get_parameter("mediapipe_angular_scale").value)
        self._linear_mapping = [int(v) for v in node.get_parameter("mediapipe_linear_axis_mapping").value]
        self._angular_mapping = [int(v) for v in node.get_parameter("mediapipe_angular_axis_mapping").value]
        self._linear_sign = [float(v) for v in node.get_parameter("mediapipe_linear_axis_sign").value]
        self._angular_sign = [float(v) for v in node.get_parameter("mediapipe_angular_axis_sign").value]
        node.create_subscription(Float32MultiArray, self._topic, self._callback, 10)

    def _select_axis(self, values: list[float], mapping: list[int], signs: list[float], scale: float) -> list[float]:
        out = [0.0, 0.0, 0.0]
        for i in range(3):
            src_idx = mapping[i]
            raw = values[src_idx] if 0 <= src_idx < len(values) else 0.0
            out[i] = map_axis_linear(raw, deadzone=self._deadzone, scale=scale * signs[i])
        return out

    def _callback(self, msg: Float32MultiArray) -> None:
        values = [float(v) for v in msg.data]
        if len(values) < 7:
            return

        twist = _zero_twist()
        linear = self._select_axis(values[:3], self._linear_mapping, self._linear_sign, self._linear_scale)
        angular = self._select_axis(values[3:6], self._angular_mapping, self._angular_sign, self._angular_scale)
        twist.linear.x, twist.linear.y, twist.linear.z = linear
        twist.angular.x, twist.angular.y, twist.angular.z = angular

        if len(values) >= 8 and values[7] <= 0.0:
            twist = _zero_twist()

        with self._lock:
            self._latest_twist = twist
            self._latest_gripper = float(_clamp(values[6], 0.0, 1.0))

    def get_command(self) -> tuple[Twist, float]:
        with self._lock:
            twist = Twist()
            twist.linear.x = self._latest_twist.linear.x
            twist.linear.y = self._latest_twist.linear.y
            twist.linear.z = self._latest_twist.linear.z
            twist.angular.x = self._latest_twist.angular.x
            twist.angular.y = self._latest_twist.angular.y
            twist.angular.z = self._latest_twist.angular.z
            return twist, float(self._latest_gripper)
