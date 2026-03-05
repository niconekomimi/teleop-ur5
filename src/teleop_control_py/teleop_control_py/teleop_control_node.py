#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import cv2
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, TwistStamped
from qbsofthand_control.srv import SetClosure
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray

from teleop_control_py.input_handlers import BaseInputHandler, HandInputHandler, JoyInputHandler


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class _NullInputHandler(BaseInputHandler):
    """Fallback strategy when an input backend is not implemented."""

    def get_command(self):
        return None

    def get_gripper_state(self):
        return None

    def is_active(self) -> bool:
        return False


class TeleopControlNode(Node):
    """Teleoperation bridge that delegates input processing via Strategy Pattern."""

    def __init__(self) -> None:
        super().__init__("teleop_control_node")

        # Parameters for easy runtime tuning
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("robot_pose_topic", "/tcp_pose_broadcaster/pose")
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("target_frame_id", "base")
        self.declare_parameter("gripper_cmd_topic", "/gripper/cmd")

        # End-effector selection:
        # - qbsofthand: use qbsofthand_control SetClosure service
        # - robotiq: use robotiq_2f_gripper action server (preferred) or topic commands (fallback)
        # NOTE: auto-detection removed at the system level; keep 'auto' as an alias for qbsofthand.
        self.declare_parameter("end_effector", "robotiq")  # qbsofthand|robotiq (auto alias supported)

        # Robotiq config (only used when end_effector=robotiq)
        # robotiq_command_interface:
        # - confidence_topic: publish [-1..1] to /robotiq_2f_gripper/confidence_command (best for trigger)
        # - binary_topic: publish +/-1 to /robotiq_2f_gripper/binary_command
        # - action: send MoveTwoFingerGripper goals (position control; can be chatty)
        self.declare_parameter("robotiq_command_interface", "confidence_topic")
        self.declare_parameter("robotiq_action_name", "/robotiq_2f_gripper_action")
        self.declare_parameter("robotiq_open_position_m", 0.14)  # [m] distance between fingers
        self.declare_parameter("robotiq_close_position_m", 0.0)  # [m]
        self.declare_parameter("robotiq_speed", 1.0)  # [0..1]
        self.declare_parameter("robotiq_force", 0.5)  # [0..1]
        self.declare_parameter("robotiq_min_goal_interval_sec", 0.10)
        self.declare_parameter("robotiq_delta_threshold", 0.02)
        # If True (and robotiq_command_interface is action), keep sending goals even if closure didn't change.
        # Default False to avoid spamming the action server when a button is held.
        self.declare_parameter("robotiq_action_force_republish", False)
        self.declare_parameter("robotiq_confidence_topic", "/robotiq_2f_gripper/confidence_command")
        # Fallback topic control (if action msg not available in python env)
        self.declare_parameter("robotiq_binary_topic", "/robotiq_2f_gripper/binary_command")
        self.declare_parameter("robotiq_binary_threshold", 0.5)  # closure >= -> close, else open
        self.declare_parameter("scale_factor", 0.5)
        self.declare_parameter("axis_mapping", [0, 1, 2])
        self.declare_parameter("smoothing_alpha", 0.2)
        self.declare_parameter("gripper_open_dist_px", 100.0)
        self.declare_parameter("gripper_close_dist_px", 20.0)
        self.declare_parameter("gripper_open_dist_m", 0.12)
        self.declare_parameter("gripper_close_dist_m", 0.03)
        self.declare_parameter("depth_unit_scale", 0.001)  # 16UC1 in mm -> m
        self.declare_parameter("hand_position_source", "hybrid")  # depth|normalized|hybrid
        # lock: keep robot orientation fixed
        # hand_relative: apply hand wrist delta rotation onto robot initial orientation
        self.declare_parameter("orientation_mode", "lock")  # lock|hand_relative
        # When using hand_relative, map the hand delta-rotation vector components into robot delta.
        # This helps fix axis mixing/inversion caused by camera/hand coordinate conventions.
        self.declare_parameter("orientation_axis_mapping", [0, 1, 2])
        self.declare_parameter("orientation_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("depth_min_m", 0.1)
        self.declare_parameter("depth_max_m", 2.0)
        # Keyboard deadman input backend:
        # - opencv: uses cv2.waitKey() events (can miss key-up; not ideal for true "hold")
        # - pynput: tracks press/release reliably (recommended for immediate stop)
        self.declare_parameter("space_deadman_backend", "opencv")  # opencv|pynput
        # opencv backend uses a latch window extended by key repeats
        self.declare_parameter("space_deadman_hold_sec", 0.3)

        # Deadman filtering: avoid flicker without adding noticeable stop latency.
        # Engage can be slightly delayed to avoid false triggers; release should be quick.
        self.declare_parameter("deadman_filter_enabled", True)
        self.declare_parameter("deadman_engage_confirm_sec", 0.10)
        self.declare_parameter("deadman_release_confirm_sec", 0.03)
        self.declare_parameter("gripper_requires_deadman", True)
        self.declare_parameter("gripper_step", 0.2)
        self.declare_parameter("gripper_duration_sec", 0.5)
        self.declare_parameter("gripper_speed_ratio", 1.0)
        # When depth-based metric gripper distance becomes temporarily unavailable,
        # hold the last valid metric distance for a short time to prevent oscillation.
        self.declare_parameter("gripper_metric_hold_sec", 0.25)
        self.declare_parameter("control_mode", "hand")  # hand|xbox
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("xbox_loop_hz", 60.0)
        self.declare_parameter("xbox_deadzone", 0.12)
        self.declare_parameter("xbox_linear_speed", 0.15)  # m/s
        self.declare_parameter("xbox_angular_speed", 0.8)  # rad/s
        self.declare_parameter("xbox_linear_axis", [0, 1, 4])
        self.declare_parameter("xbox_linear_sign", [1.0, -1.0, -1.0])
        self.declare_parameter("xbox_angular_axis", [7, 6, 3])
        self.declare_parameter("xbox_angular_sign", [1.0, 1.0, -1.0])
        self.declare_parameter("xbox_deadman_button", 5)
        self.declare_parameter("xbox_gripper_close_button", 0)
        self.declare_parameter("xbox_gripper_open_button", 1)

        gripper_cmd_topic = self.get_parameter("gripper_cmd_topic").get_parameter_value().string_value
        self.control_mode = self.get_parameter("control_mode").get_parameter_value().string_value.strip().lower()
        if self.control_mode not in {"hand", "xbox"}:
            self.get_logger().warn("control_mode must be 'hand' or 'xbox'; falling back to 'hand'")
            self.control_mode = "hand"

        self.target_frame_id = self.get_parameter("target_frame_id").get_parameter_value().string_value

        self.gripper_step = float(self.get_parameter("gripper_step").get_parameter_value().double_value)
        self.gripper_duration_sec = float(self.get_parameter("gripper_duration_sec").get_parameter_value().double_value)
        self.gripper_speed_ratio = float(self.get_parameter("gripper_speed_ratio").get_parameter_value().double_value)

        self._end_effector_param = str(self.get_parameter("end_effector").value).strip().lower()
        if self._end_effector_param == "auto":
            self.get_logger().info("end_effector=auto is deprecated; using qbsofthand (manual selection only)")
            self._end_effector_param = "qbsofthand"

        if self._end_effector_param not in {"qbsofthand", "robotiq"}:
            self.get_logger().warn(
                f"end_effector must be one of qbsofthand|robotiq; got '{self._end_effector_param}', falling back to qbsofthand"
            )
            self._end_effector_param = "qbsofthand"

        # Publishers (fixed topics as a central dispatcher)
        self.pose_pub = self.create_publisher(PoseStamped, "/pose_target_cmds", 10)
        self.twist_pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.gripper_pub = self.create_publisher(Float32, gripper_cmd_topic, 10)
        self._qb_client = self.create_client(SetClosure, "/qbsofthand_control_node/set_closure")
        if not self._qb_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().info("SoftHand service not available yet; will retry on demand")

        self._robotiq_binary_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("robotiq_binary_topic").value),
            10,
        )

        self._robotiq_confidence_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("robotiq_confidence_topic").value),
            10,
        )

        self._robotiq_action_client = None
        self._robotiq_action_goal_type = None
        self._setup_robotiq_action_client_best_effort()

        self._selected_end_effector: str = "publish_only"  # publish_only|qbsofthand|robotiq
        self._select_end_effector(initial=True)

        self.add_on_set_parameters_callback(self._on_params)
        self._last_gripper_cmd: Optional[float] = None
        self._last_gripper_call_ns = 0
        self._gripper_delta_threshold = 0.02
        self._gripper_min_interval_ns = int(0.1 * 1e9)

        self._last_gripper_pub_cmd: Optional[float] = None

        self._robotiq_last_cmd: Optional[float] = None
        self._robotiq_last_call_ns: int = 0
        self.input_handler: BaseInputHandler
        if self.control_mode == "hand":
            self.input_handler = HandInputHandler(self)
            self.get_logger().info("TeleopControlNode mode=hand (HandInputHandler)")
        else:
            self.input_handler = JoyInputHandler(self)
            self.get_logger().info("TeleopControlNode mode=xbox (JoyInputHandler)")

        self._tick_timer = self.create_timer(1.0 / 60.0, self._tick)

        # Manual selection only: no periodic probing.

    def destroy_node(self):  # type: ignore[override]
        try:
            if isinstance(self.input_handler, HandInputHandler):
                self.input_handler.destroy()
        except Exception:
            pass
        return super().destroy_node()

    def _tick(self) -> None:
        # Deadman first: immediate stop if not active
        if not self.input_handler.is_active():
            stop = TwistStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.header.frame_id = self.target_frame_id
            stop.twist.linear.x = 0.0
            stop.twist.linear.y = 0.0
            stop.twist.linear.z = 0.0
            stop.twist.angular.x = 0.0
            stop.twist.angular.y = 0.0
            stop.twist.angular.z = 0.0
            self.twist_pub.publish(stop)
            return

        cmd = self.input_handler.get_command()
        if isinstance(cmd, PoseStamped):
            self.pose_pub.publish(cmd)
        elif isinstance(cmd, TwistStamped):
            self.twist_pub.publish(cmd)

        gripper = self.input_handler.get_gripper_state()
        if gripper is not None:
            self._publish_gripper(float(gripper))

    def _publish_gripper(self, cmd: float) -> None:
        # Guard against invalid values from upstream (NaN/inf).
        try:
            cmd_f = float(cmd)
        except Exception:
            return
        if not math.isfinite(cmd_f):
            return

        cmd_clamped = float(clamp(cmd_f, 0.0, 1.0))
        if self.gripper_step and self.gripper_step > 0:
            step = float(self.gripper_step)
            quantized = round(cmd_clamped / step) * step
            quantized = float(clamp(quantized, 0.0, 1.0))
        else:
            # gripper_step<=0: continuous control (no quantization)
            quantized = cmd_clamped

        # Default behavior: if command didn't change, don't republish or re-dispatch.
        force_republish = False
        if self._selected_end_effector == "robotiq":
            iface = str(self.get_parameter("robotiq_command_interface").value).strip().lower()
            if iface in {"action", "position", "pos"}:
                force_republish = bool(self.get_parameter("robotiq_action_force_republish").value)

        if not force_republish:
            if self._last_gripper_pub_cmd is not None and abs(quantized - self._last_gripper_pub_cmd) < 1e-9:
                return
            self._last_gripper_pub_cmd = quantized

        msg = Float32()
        msg.data = quantized
        self.gripper_pub.publish(msg)
        self._dispatch_gripper_to_end_effector(quantized)

    def _setup_robotiq_action_client_best_effort(self) -> None:
        """Try to create a Robotiq action client if message types are available."""
        try:
            from robotiq_2f_gripper_msgs.action import MoveTwoFingerGripper  # type: ignore

            action_name = str(self.get_parameter("robotiq_action_name").value)
            self._robotiq_action_goal_type = MoveTwoFingerGripper
            self._robotiq_action_client = ActionClient(self, MoveTwoFingerGripper, action_name)
        except Exception:
            self._robotiq_action_goal_type = None
            self._robotiq_action_client = None

    def _robotiq_server_ready(self) -> bool:
        if self._robotiq_action_client is None:
            return False
        try:
            return bool(self._robotiq_action_client.server_is_ready())
        except Exception:
            return False

    def _robotiq_topic_available(self) -> bool:
        """Detect Robotiq presence by checking if its command topic exists in the ROS graph."""
        topic = str(self.get_parameter("robotiq_binary_topic").value)
        try:
            names_and_types = self.get_topic_names_and_types()
            return any(name == topic for name, _types in names_and_types)
        except Exception:
            return False

    def _qb_ready(self) -> bool:
        try:
            if self._qb_client.service_is_ready():
                return True
            return bool(self._qb_client.wait_for_service(timeout_sec=0.0))
        except Exception:
            return False

    def _select_end_effector(self, initial: bool = False) -> None:
        desired = self._end_effector_param

        selected = "publish_only"
        if desired == "robotiq":
            # Forced mode: keep selection even if backend isn't ready yet.
            selected = "robotiq"
        elif desired == "qbsofthand":
            # Forced mode: keep selection even if service isn't ready yet.
            selected = "qbsofthand"

        if selected != self._selected_end_effector:
            self._selected_end_effector = selected
            self.get_logger().info(
                f"End-effector backend selected: {self._selected_end_effector} (param={self._end_effector_param})"
            )

        if initial and self._selected_end_effector == "robotiq" and self._robotiq_action_client is None:
            self.get_logger().warn(
                "Robotiq action message type not available in this Python env; will use binary topic fallback. "
                "(Build & source the workspace overlay if you want action-based continuous control.)"
            )

    def _dispatch_gripper_to_end_effector(self, closure: float) -> None:
        if self._selected_end_effector == "qbsofthand":
            self._call_qb_service(closure)
            return
        if self._selected_end_effector == "robotiq":
            iface = str(self.get_parameter("robotiq_command_interface").value).strip().lower()
            if iface in {"confidence", "confidence_topic", "topic"}:
                self._pub_robotiq_confidence(closure)
                return
            if iface in {"binary", "binary_topic"}:
                self._pub_robotiq_binary(closure)
                return

            # action (position control)
            if iface in {"action", "position", "pos"}:
                if self._robotiq_action_client is not None and self._robotiq_action_goal_type is not None:
                    self._send_robotiq_action(closure)
                else:
                    # Best-effort fallback when action message type isn't available in this Python env.
                    self._pub_robotiq_binary(closure)
                return

    def _call_qb_service(self, closure: float) -> None:
        closure = float(clamp(closure, 0.0, 1.0))
        now_ns = self.get_clock().now().nanoseconds
        if self._last_gripper_cmd is not None:
            # Don't resend the same command just because time has passed.
            if abs(closure - self._last_gripper_cmd) < self._gripper_delta_threshold:
                return
            # If the command changed, still limit the update rate.
            if now_ns - self._last_gripper_call_ns < self._gripper_min_interval_ns:
                return
        if not self._qb_ready():
            return
        request = SetClosure.Request()
        request.closure = closure
        request.duration_sec = self.gripper_duration_sec
        request.speed_ratio = self.gripper_speed_ratio
        try:
            self._qb_client.call_async(request)
            self._last_gripper_cmd = closure
            self._last_gripper_call_ns = now_ns
        except Exception as exc:  # pragma: no cover - best effort
            self.get_logger().warn(f"SoftHand call failed: {exc}")

    def _send_robotiq_action(self, closure: float) -> None:
        if self._robotiq_action_client is None or self._robotiq_action_goal_type is None:
            return
        if not self._robotiq_server_ready():
            return

        closure = float(clamp(closure, 0.0, 1.0))
        now_ns = self.get_clock().now().nanoseconds

        min_interval_ns = int(float(self.get_parameter("robotiq_min_goal_interval_sec").value) * 1e9)
        delta_threshold = float(self.get_parameter("robotiq_delta_threshold").value)
        if self._robotiq_last_cmd is not None:
            # Don't resend the same goal just because the button is held.
            if abs(closure - self._robotiq_last_cmd) <= delta_threshold:
                return
            # If goal changed, still limit update rate.
            if now_ns - self._robotiq_last_call_ns < min_interval_ns:
                return

        open_pos = float(self.get_parameter("robotiq_open_position_m").value)
        close_pos = float(self.get_parameter("robotiq_close_position_m").value)
        speed = float(clamp(float(self.get_parameter("robotiq_speed").value), 0.0, 1.0))
        force = float(clamp(float(self.get_parameter("robotiq_force").value), 0.0, 1.0))

        # closure=0 -> open_pos, closure=1 -> close_pos
        target = open_pos + (close_pos - open_pos) * closure
        # Keep in-range even if user configured open/close inversely
        lo = min(open_pos, close_pos)
        hi = max(open_pos, close_pos)
        target = float(clamp(target, lo, hi))

        goal = self._robotiq_action_goal_type.Goal()  # type: ignore[attr-defined]
        goal.target_position = float(target)
        goal.target_speed = float(speed)
        goal.target_force = float(force)

        try:
            self._robotiq_action_client.send_goal_async(goal)
            self._robotiq_last_cmd = closure
            self._robotiq_last_call_ns = now_ns
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f"Robotiq action send_goal failed: {exc}")

    def _pub_robotiq_binary(self, closure: float) -> None:
        # Fallback: publish +/-1.0 to binary_command
        closure = float(clamp(closure, 0.0, 1.0))

        now_ns = self.get_clock().now().nanoseconds
        min_interval_ns = int(float(self.get_parameter("robotiq_min_goal_interval_sec").value) * 1e9)
        delta_threshold = float(self.get_parameter("robotiq_delta_threshold").value)
        if self._robotiq_last_cmd is not None:
            if abs(closure - self._robotiq_last_cmd) < delta_threshold:
                return
            if now_ns - self._robotiq_last_call_ns < min_interval_ns:
                return

        thr = float(clamp(float(self.get_parameter("robotiq_binary_threshold").value), 0.0, 1.0))
        val = -1.0 if closure >= thr else 1.0  # -1 close, +1 open
        msg = Float32MultiArray()
        msg.data = [float(val)]
        self._robotiq_binary_pub.publish(msg)

        self._robotiq_last_cmd = closure
        self._robotiq_last_call_ns = now_ns

    def _pub_robotiq_confidence(self, closure: float) -> None:
        # confidence_command expects a Float32MultiArray with one value in [-1, 1].
        # Positive -> open, negative -> close. Driver applies hysteresis.
        closure = float(clamp(closure, 0.0, 1.0))

        now_ns = self.get_clock().now().nanoseconds
        min_interval_ns = int(float(self.get_parameter("robotiq_min_goal_interval_sec").value) * 1e9)
        delta_threshold = float(self.get_parameter("robotiq_delta_threshold").value)
        if self._robotiq_last_cmd is not None:
            if abs(closure - self._robotiq_last_cmd) < delta_threshold:
                return
            if now_ns - self._robotiq_last_call_ns < min_interval_ns:
                return

        confidence = float(clamp(1.0 - 2.0 * closure, -1.0, 1.0))
        msg = Float32MultiArray()
        msg.data = [confidence]
        self._robotiq_confidence_pub.publish(msg)

        self._robotiq_last_cmd = closure
        self._robotiq_last_call_ns = now_ns

    def _on_params(self, params):
        # Allow runtime switching via `ros2 param set`.
        ee_updates = [p for p in params if p.name == "end_effector"]
        if ee_updates:
            v = ee_updates[0].value
            if isinstance(v, str):
                vv = v.strip().lower()
                if vv == "auto":
                    vv = "qbsofthand"
                if vv in {"qbsofthand", "robotiq"}:
                    self._end_effector_param = vv
                    self._select_end_effector()
                    return SetParametersResult(successful=True)
                return SetParametersResult(successful=False, reason="invalid end_effector")
            return SetParametersResult(successful=False, reason="end_effector must be string")
        return SetParametersResult(successful=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
