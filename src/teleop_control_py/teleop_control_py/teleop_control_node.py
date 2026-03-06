#!/usr/bin/env python3
"""遥操作主节点：只负责参数装配、策略实例化和高频控制循环。"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .gripper_controllers import QbSoftHandController, RobotiqController
from .input_handlers import JoyInputHandler, MediaPipeInputHandler
from .servo_pose_follower import ServoPoseFollower
from .transform_utils import apply_velocity_limits


class TeleopControlNode(Node):
    """主调度节点，遵循 DIP：只依赖抽象协议，不依赖具体设备细节。"""

    def __init__(self) -> None:
        super().__init__("teleop_control_node")
        self._declare_parameters()

        self._input_type = str(self.get_parameter("input_type").value).strip().lower()
        self._gripper_type = str(self.get_parameter("gripper_type").value).strip().lower()
        self._control_hz = max(1.0, float(self.get_parameter("control_hz").value))

        self._max_velocity = np.array(
            [
                float(self.get_parameter("max_linear_vel").value),
                float(self.get_parameter("max_linear_vel").value),
                float(self.get_parameter("max_linear_vel").value),
                float(self.get_parameter("max_angular_vel").value),
                float(self.get_parameter("max_angular_vel").value),
                float(self.get_parameter("max_angular_vel").value),
            ],
            dtype=np.float64,
        )
        self._max_acceleration = np.array(
            [
                float(self.get_parameter("max_linear_accel").value),
                float(self.get_parameter("max_linear_accel").value),
                float(self.get_parameter("max_linear_accel").value),
                float(self.get_parameter("max_angular_accel").value),
                float(self.get_parameter("max_angular_accel").value),
                float(self.get_parameter("max_angular_accel").value),
            ],
            dtype=np.float64,
        )
        self._last_twist_vec = np.zeros(6, dtype=np.float64)
        self._last_loop_time = time.monotonic()

        self.input_handler = self._build_input_handler(self._input_type)
        self.gripper_ctrl = self._build_gripper_controller(self._gripper_type)
        self.arm_ctrl = ServoPoseFollower(self)

        self._timer = self.create_timer(1.0 / self._control_hz, self._control_loop)
        self.get_logger().info(
            f"TeleopControlNode ready. input_type={self._input_type}, gripper_type={self._gripper_type}, "
            f"control_hz={self._control_hz:.1f}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_type", "joy")
        self.declare_parameter("gripper_type", "robotiq")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("target_frame_id", "base")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("auto_switch_controllers", True)
        self.declare_parameter("controller_manager_ns", "/controller_manager")
        self.declare_parameter("teleop_controller", "forward_position_controller")
        self.declare_parameter("trajectory_controller", "scaled_joint_trajectory_controller")
        self.declare_parameter("startup_retry_period_sec", 1.0)
        self.declare_parameter("max_linear_vel", 1.5)
        self.declare_parameter("max_angular_vel", 3.0)
        self.declare_parameter("max_linear_accel", 4.0)
        self.declare_parameter("max_angular_accel", 8.0)

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("input_watchdog_timeout_sec", 0.2)
        self.declare_parameter("joy_deadzone", 0.05)
        self.declare_parameter("joy_curve", "linear")
        self.declare_parameter("joy_deadman_enabled", False)
        self.declare_parameter("deadman_button", -1)
        self.declare_parameter("deadman_axis", 4)
        self.declare_parameter("deadman_axis_threshold", 0.5)
        self.declare_parameter("linear_x_axis", 0)
        self.declare_parameter("linear_y_axis", 1)
        self.declare_parameter("linear_z_axis", -1)
        self.declare_parameter("linear_z_up_button", 1)
        self.declare_parameter("linear_z_down_button", 0)
        self.declare_parameter("angular_x_axis", 3)
        self.declare_parameter("angular_y_axis", 2)
        self.declare_parameter("angular_z_axis", -1)
        self.declare_parameter("angular_z_positive_button", 3)
        self.declare_parameter("angular_z_negative_button", 2)
        self.declare_parameter("linear_axis_sign", [-1.0, -1.0, 1.0])
        self.declare_parameter("angular_axis_sign", [-1.0, 1.0, 1.0])
        self.declare_parameter("gripper_close_button", 5)
        self.declare_parameter("gripper_open_button", 4)
        self.declare_parameter("gripper_axis", -1)
        self.declare_parameter("gripper_axis_inverted", False)

        self.declare_parameter("mediapipe_input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("mediapipe_topic", "")
        self.declare_parameter("mediapipe_depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("mediapipe_camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("mediapipe_deadzone", 0.02)
        self.declare_parameter("mediapipe_linear_scale", 1.0)
        self.declare_parameter("mediapipe_angular_scale", 1.0)
        self.declare_parameter("mediapipe_linear_axis_mapping", [0, 1, 2])
        self.declare_parameter("mediapipe_angular_axis_mapping", [0, 1, 2])
        self.declare_parameter("mediapipe_linear_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("mediapipe_angular_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("mediapipe_hand_position_source", "hybrid")
        self.declare_parameter("mediapipe_orientation_mode", "lock")
        self.declare_parameter("mediapipe_orientation_axis_mapping", [0, 1, 2])
        self.declare_parameter("mediapipe_orientation_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("mediapipe_depth_min_m", 0.1)
        self.declare_parameter("mediapipe_depth_max_m", 2.0)
        self.declare_parameter("mediapipe_depth_unit_scale", 0.001)
        self.declare_parameter("mediapipe_smoothing_alpha", 0.2)
        self.declare_parameter("mediapipe_gripper_open_dist_px", 100.0)
        self.declare_parameter("mediapipe_gripper_close_dist_px", 20.0)
        self.declare_parameter("mediapipe_gripper_open_dist_m", 0.12)
        self.declare_parameter("mediapipe_gripper_close_dist_m", 0.03)
        self.declare_parameter("mediapipe_gripper_metric_hold_sec", 0.25)
        self.declare_parameter("mediapipe_gripper_requires_deadman", True)
        self.declare_parameter("mediapipe_deadman_filter_enabled", True)
        self.declare_parameter("mediapipe_deadman_engage_confirm_sec", 0.10)
        self.declare_parameter("mediapipe_deadman_release_confirm_sec", 0.03)
        self.declare_parameter("mediapipe_space_deadman_backend", "opencv")
        self.declare_parameter("mediapipe_space_deadman_hold_sec", 0.3)
        self.declare_parameter("mediapipe_show_debug_window", True)

        self.declare_parameter("gripper_cmd_topic", "/gripper/cmd")
        self.declare_parameter("gripper_command_delta", 0.01)
        self.declare_parameter("robotiq_command_interface", "confidence_topic")
        self.declare_parameter("robotiq_confidence_topic", "/robotiq_2f_gripper/confidence_command")
        self.declare_parameter("robotiq_binary_topic", "/robotiq_2f_gripper/binary_command")
        self.declare_parameter("robotiq_binary_threshold", 0.5)
        self.declare_parameter("qbsofthand_service_name", "/qbsofthand_control_node/set_closure")
        self.declare_parameter("qbsofthand_duration_sec", 0.3)
        self.declare_parameter("qbsofthand_speed_ratio", 1.0)

    def _build_input_handler(self, input_type: str):
        strategies = {
            "joy": JoyInputHandler,
            "mediapipe": MediaPipeInputHandler,
        }
        handler_cls = strategies.get(input_type)
        if handler_cls is None:
            self.get_logger().warn(f"未知 input_type '{input_type}'，回退到 joy。")
            handler_cls = JoyInputHandler
        return handler_cls(self)

    def _build_gripper_controller(self, gripper_type: str):
        strategies = {
            "robotiq": RobotiqController,
            "qbsofthand": QbSoftHandController,
        }
        controller_cls = strategies.get(gripper_type)
        if controller_cls is None:
            self.get_logger().warn(f"未知 gripper_type '{gripper_type}'，回退到 robotiq。")
            controller_cls = RobotiqController
        return controller_cls(self)

    def _twist_to_vector(self, twist: Twist) -> np.ndarray:
        return np.array(
            [
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ],
            dtype=np.float64,
        )

    def _vector_to_twist(self, values: np.ndarray) -> Twist:
        twist = Twist()
        twist.linear.x = float(values[0])
        twist.linear.y = float(values[1])
        twist.linear.z = float(values[2])
        twist.angular.x = float(values[3])
        twist.angular.y = float(values[4])
        twist.angular.z = float(values[5])
        return twist

    def _control_loop(self) -> None:
        twist, gripper_val = self.input_handler.get_command()
        now = time.monotonic()
        dt = max(1e-4, now - self._last_loop_time)
        self._last_loop_time = now

        target_vec = self._twist_to_vector(twist)
        limited_vec = apply_velocity_limits(
            target=target_vec,
            previous=self._last_twist_vec,
            max_velocity=self._max_velocity,
            max_acceleration=self._max_acceleration,
            dt=dt,
        )

        # When the input strategy has already snapped an axis back to zero,
        # clear that axis immediately instead of letting accel limiting create coast.
        zero_axes = np.isclose(target_vec, 0.0, atol=1e-6)
        limited_vec[zero_axes] = 0.0
        self._last_twist_vec = limited_vec

        limited_twist = self._vector_to_twist(limited_vec)
        self.arm_ctrl.send_twist(limited_twist)
        self.gripper_ctrl.set_gripper(gripper_val)

    def destroy_node(self) -> bool:  # type: ignore[override]
        try:
            self.input_handler.stop()
        except Exception:
            pass
        try:
            self.arm_ctrl.stop()
        except Exception:
            pass
        try:
            self.gripper_ctrl.stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
