#!/usr/bin/env python3
"""夹爪策略层：不同末端执行器统一暴露 set_gripper(value) 接口。"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray

from .transform_utils import _clamp

try:
    from qbsofthand_control.srv import SetClosure
except Exception:  # pragma: no cover - 运行环境未安装时保持模块可导入
    SetClosure = None


class GripperControllerBase(ABC):
    """夹爪控制器统一接口。"""

    def __init__(self, node: Node) -> None:
        self.node = node
        self._lock = threading.Lock()
        self._last_value: Optional[float] = None
        self._state_topic = str(node.get_parameter("gripper_cmd_topic").value)
        self._state_pub = node.create_publisher(Float32, self._state_topic, 10)

    @abstractmethod
    def set_gripper(self, value: float) -> None:
        """设置夹爪开合度，0.0 为全开，1.0 为全闭。"""

    def stop(self) -> None:
        """可选释放资源。"""

    def _publish_state(self, value: float) -> None:
        msg = Float32()
        msg.data = float(_clamp(value, 0.0, 1.0))
        self._state_pub.publish(msg)


class RobotiqController(GripperControllerBase):
    """Robotiq 控制器，兼容 confidence/binary 两种话题接口。"""

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self._command_interface = str(node.get_parameter("robotiq_command_interface").value).strip().lower()
        self._confidence_topic = str(node.get_parameter("robotiq_confidence_topic").value)
        self._binary_topic = str(node.get_parameter("robotiq_binary_topic").value)
        self._binary_threshold = float(node.get_parameter("robotiq_binary_threshold").value)
        self._min_delta = float(node.get_parameter("gripper_command_delta").value)
        self._confidence_pub = node.create_publisher(Float32MultiArray, self._confidence_topic, 10)
        self._binary_pub = node.create_publisher(Float32MultiArray, self._binary_topic, 10)

    def set_gripper(self, value: float) -> None:
        closure = float(_clamp(value, 0.0, 1.0))
        with self._lock:
            if self._last_value is not None and abs(closure - self._last_value) < self._min_delta:
                return
            self._last_value = closure

        self._publish_state(closure)

        if self._command_interface in {"binary", "binary_topic"}:
            msg = Float32MultiArray()
            msg.data = [-1.0 if closure >= self._binary_threshold else 1.0]
            self._binary_pub.publish(msg)
            return

        # confidence_command: positive=open, negative=close, 0=neutral
        msg = Float32MultiArray()
        msg.data = [float(1.0 - 2.0 * closure)]
        self._confidence_pub.publish(msg)


class QbSoftHandController(GripperControllerBase):
    """qbSoftHand 控制器，优先调用服务，失败时退化到话题。"""

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self._service_name = str(node.get_parameter("qbsofthand_service_name").value)
        self._topic_name = str(node.get_parameter("gripper_cmd_topic").value)
        self._duration_sec = float(node.get_parameter("qbsofthand_duration_sec").value)
        self._speed_ratio = float(node.get_parameter("qbsofthand_speed_ratio").value)
        self._min_delta = float(node.get_parameter("gripper_command_delta").value)
        self._topic_pub = node.create_publisher(Float32, self._topic_name, 10)
        self._service_client = None if SetClosure is None else node.create_client(SetClosure, self._service_name)

    def set_gripper(self, value: float) -> None:
        closure = float(_clamp(value, 0.0, 1.0))
        with self._lock:
            if self._last_value is not None and abs(closure - self._last_value) < self._min_delta:
                return
            self._last_value = closure

        self._publish_state(closure)

        if self._service_client is not None and self._service_client.service_is_ready():
            request = SetClosure.Request()
            request.closure = closure
            request.duration_sec = self._duration_sec
            request.speed_ratio = self._speed_ratio
            self._service_client.call_async(request)
            return

        topic_msg = Float32()
        topic_msg.data = closure
        self._topic_pub.publish(topic_msg)
