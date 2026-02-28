#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from qbsofthand_control.srv import SetClosure
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Joy
from std_msgs.msg import Float32


class LowPassFilter:
    """Simple exponential moving average for jitter reduction."""

    def __init__(self, alpha: float, initial: Optional[np.ndarray] = None) -> None:
        self.alpha = float(alpha)
        self.state = initial

    def reset(self, value: Optional[np.ndarray] = None) -> None:
        self.state = value

    def apply(self, value: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = value
            return value
        self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = (float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3]))
    x2, y2, z2, w2 = (float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3]))
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (q / norm).astype(np.float64)


def rotmat_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion in ROS (x,y,z,w) ordering."""
    m00, m01, m02 = float(r[0, 0]), float(r[0, 1]), float(r[0, 2])
    m10, m11, m12 = float(r[1, 0]), float(r[1, 1]), float(r[1, 2])
    m20, m21, m22 = float(r[2, 0]), float(r[2, 1]), float(r[2, 2])

    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return quat_normalize_xyzw(np.array([qx, qy, qz, qw], dtype=np.float64))


def quat_to_rotvec_xyzw(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> rotation vector (axis * angle)."""
    qn = quat_normalize_xyzw(q.astype(np.float64))
    x, y, z, w = (float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3]))
    w = clamp(w, -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([x / s, y / s, z / s], dtype=np.float64)
    return axis * angle


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) -> quaternion (x,y,z,w)."""
    rv = rotvec.astype(np.float64)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rv / angle
    half = 0.5 * angle
    s = math.sin(half)
    return quat_normalize_xyzw(np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float64))


class TeleopControlNode(Node):
    """
    Teleoperation bridge from MediaPipe hand tracking to UR5 + SoftHand.

    Key ideas:
    - Deadman switch: holding space bar keeps motion enabled.
    - Relative control: capture initial hand pose and robot pose, then apply deltas.
    - Low-pass filter on target position for smoother motion.
    - Gripper: thumb-index distance mapped to [0.0, 1.0].

    How to tune:
    - Scale factor: parameter `scale_factor` (default 0.5) multiplies hand deltas before sending.
    - Axis mapping: parameter `axis_mapping` defines which hand axis drives each robot axis, e.g. [0, 2, 1]
      maps hand X->robot X, hand Z->robot Y, hand Y->robot Z.
    """

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

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        robot_pose_topic = self.get_parameter("robot_pose_topic").get_parameter_value().string_value
        target_pose_topic = self.get_parameter("target_pose_topic").get_parameter_value().string_value
        self.target_frame_id = self.get_parameter("target_frame_id").get_parameter_value().string_value
        gripper_cmd_topic = self.get_parameter("gripper_cmd_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.scale_factor = float(self.get_parameter("scale_factor").get_parameter_value().double_value)
        self.axis_mapping = self._validate_axis_mapping(
            self.get_parameter("axis_mapping").get_parameter_value().integer_array_value
        )
        smoothing_alpha = float(self.get_parameter("smoothing_alpha").get_parameter_value().double_value)
        self.gripper_open_dist_px = float(
            self.get_parameter("gripper_open_dist_px").get_parameter_value().double_value
        )
        self.gripper_close_dist_px = float(
            self.get_parameter("gripper_close_dist_px").get_parameter_value().double_value
        )
        self.gripper_open_dist_m = float(
            self.get_parameter("gripper_open_dist_m").get_parameter_value().double_value
        )
        self.gripper_close_dist_m = float(
            self.get_parameter("gripper_close_dist_m").get_parameter_value().double_value
        )
        self.depth_unit_scale = float(
            self.get_parameter("depth_unit_scale").get_parameter_value().double_value
        )
        self.hand_position_source = (
            self.get_parameter("hand_position_source").get_parameter_value().string_value.strip().lower()
        )
        self.orientation_mode = (
            self.get_parameter("orientation_mode").get_parameter_value().string_value.strip().lower()
        )
        if self.orientation_mode not in {"lock", "hand_relative"}:
            self.get_logger().warn("orientation_mode must be 'lock' or 'hand_relative'; falling back to 'lock'")
            self.orientation_mode = "lock"

        self.orientation_axis_mapping = self._validate_axis_mapping(
            list(self.get_parameter("orientation_axis_mapping").value)
        )
        self.orientation_axis_sign = self._validate_axis_sign(
            list(self.get_parameter("orientation_axis_sign").value)
        )
        self.depth_min_m = float(self.get_parameter("depth_min_m").get_parameter_value().double_value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").get_parameter_value().double_value)
        self.space_deadman_backend = (
            self.get_parameter("space_deadman_backend").get_parameter_value().string_value.strip().lower()
        )
        if self.space_deadman_backend not in {"opencv", "pynput"}:
            self.get_logger().warn("space_deadman_backend must be 'opencv' or 'pynput'; falling back to 'opencv'")
            self.space_deadman_backend = "opencv"
        self.space_deadman_hold_sec = float(
            self.get_parameter("space_deadman_hold_sec").get_parameter_value().double_value
        )
        self.deadman_filter_enabled = bool(
            self.get_parameter("deadman_filter_enabled").get_parameter_value().bool_value
        )
        self.deadman_engage_confirm_sec = float(
            self.get_parameter("deadman_engage_confirm_sec").get_parameter_value().double_value
        )
        self.deadman_release_confirm_sec = float(
            self.get_parameter("deadman_release_confirm_sec").get_parameter_value().double_value
        )
        self.gripper_requires_deadman = bool(
            self.get_parameter("gripper_requires_deadman").get_parameter_value().bool_value
        )
        self.gripper_step = float(
            self.get_parameter("gripper_step").get_parameter_value().double_value
        )
        self.gripper_duration_sec = float(
            self.get_parameter("gripper_duration_sec").get_parameter_value().double_value
        )
        self.gripper_speed_ratio = float(
            self.get_parameter("gripper_speed_ratio").get_parameter_value().double_value
        )
        self.gripper_metric_hold_sec = float(self.get_parameter("gripper_metric_hold_sec").value)
        self.control_mode = self.get_parameter("control_mode").get_parameter_value().string_value.strip().lower()
        if self.control_mode not in {"hand", "xbox"}:
            self.get_logger().warn("control_mode must be 'hand' or 'xbox'; falling back to 'hand'")
            self.control_mode = "hand"
        joy_topic = self.get_parameter("joy_topic").get_parameter_value().string_value
        self.xbox_loop_hz = float(self.get_parameter("xbox_loop_hz").value)
        self.xbox_deadzone = float(self.get_parameter("xbox_deadzone").value)
        self.xbox_linear_speed = float(self.get_parameter("xbox_linear_speed").value)
        self.xbox_angular_speed = float(self.get_parameter("xbox_angular_speed").value)
        self.xbox_linear_axis = self._validate_axis_index_list(list(self.get_parameter("xbox_linear_axis").value))
        self.xbox_linear_sign = self._validate_axis_sign(list(self.get_parameter("xbox_linear_sign").value))
        self.xbox_angular_axis = self._validate_axis_index_list(list(self.get_parameter("xbox_angular_axis").value))
        self.xbox_angular_sign = self._validate_axis_sign(list(self.get_parameter("xbox_angular_sign").value))
        self.xbox_deadman_button = int(self.get_parameter("xbox_deadman_button").value)
        self.xbox_gripper_close_button = int(self.get_parameter("xbox_gripper_close_button").value)
        self.xbox_gripper_open_button = int(self.get_parameter("xbox_gripper_open_button").value)

        # ROS interfaces
        self.bridge = CvBridge()
        if self.control_mode == "hand":
            self.create_subscription(Image, image_topic, self._image_callback, 10)
            self.create_subscription(Image, depth_topic, self._depth_callback, 5)
            self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, 1)
        else:
            self.create_subscription(Joy, joy_topic, self._joy_callback, 20)
        self.create_subscription(PoseStamped, robot_pose_topic, self._pose_callback, 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, target_pose_topic, 10)
        self.gripper_pub = self.create_publisher(Float32, gripper_cmd_topic, 10)
        self.gripper_client = self.create_client(SetClosure, "/qbsofthand_control_node/set_closure")
        if not self.gripper_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("SoftHand service not available; will retry on demand")
        self._last_gripper_cmd: Optional[float] = None
        self._last_gripper_call_ns = 0
        self._gripper_delta_threshold = 0.02
        self._gripper_min_interval_ns = int(0.1 * 1e9)
        self._last_metric_gripper_dist_m: Optional[float] = None
        self._last_metric_gripper_dist_ns: int = 0

        # MediaPipe setup
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None
        if self.control_mode == "hand":
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self.mp_drawing = mp.solutions.drawing_utils

        # State
        self.latest_robot_pose: Optional[Pose] = None
        self.initial_hand_pos: Optional[np.ndarray] = None
        self.initial_hand_quat: Optional[np.ndarray] = None  # hand orientation at deadman engage (x,y,z,w)
        self.initial_robot_pos: Optional[np.ndarray] = None  # 3-vector
        self.initial_robot_orientation: Optional[np.ndarray] = None  # 4-vector
        self.deadman_active = False
        self._space_deadman_until_ns: int = 0
        self._space_down: bool = False
        self._keyboard_listener = None
        self._deadman_filtered: bool = False
        self._deadman_candidate: Optional[bool] = None
        self._deadman_candidate_since_ns: int = 0
        self._joy_axes: List[float] = []
        self._joy_buttons: List[int] = []
        self._joy_prev_buttons: List[int] = []
        self._xbox_target_pos: Optional[np.ndarray] = None
        self._xbox_target_quat: Optional[np.ndarray] = None
        self._xbox_last_tick_ns: int = 0
        self.depth_image: Optional[np.ndarray] = None
        self.depth_stamp_ns: int = 0
        self.depth_encoding: Optional[str] = None
        self.depth_info: Optional[CameraInfo] = None
        self.fx = self.fy = self.cx = self.cy = None

        # Filtering per-axis for the target position
        self.pos_filter = LowPassFilter(alpha=smoothing_alpha)

        # Optional: track true key press/release for immediate stop
        if self.control_mode == "hand" and self.space_deadman_backend == "pynput":
            try:
                from pynput import keyboard  # type: ignore

                def on_press(key):
                    try:
                        if key == keyboard.Key.space:
                            self._space_down = True
                    except Exception:
                        pass

                def on_release(key):
                    try:
                        if key == keyboard.Key.space:
                            self._space_down = False
                    except Exception:
                        pass

                self._keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
                self._keyboard_listener.daemon = True
                self._keyboard_listener.start()
            except Exception as exc:
                self.get_logger().warn(f"Failed to start pynput keyboard listener; falling back to opencv. Err: {exc}")
                self.space_deadman_backend = "opencv"

        self._xbox_timer = None
        if self.control_mode == "xbox":
            period = 1.0 / max(1.0, self.xbox_loop_hz)
            self._xbox_timer = self.create_timer(period, self._xbox_control_timer)
            self.get_logger().info("TeleopControlNode ready. Xbox mode enabled.")
        else:
            self.get_logger().info("TeleopControlNode ready. Hold space to move.")
        self.get_logger().info(
            "Orientation config: mode=%s, axis_mapping=%s, axis_sign=%s"
            % (
                self.orientation_mode,
                self.orientation_axis_mapping,
                self.orientation_axis_sign.tolist(),
            )
        )

    def destroy_node(self):  # type: ignore[override]
        try:
            if self._keyboard_listener is not None:
                self._keyboard_listener.stop()
        except Exception:
            pass
        return super().destroy_node()

    def _deadman_filter(self, desired: bool, now_ns: int) -> bool:
        if not self.deadman_filter_enabled:
            self._deadman_filtered = desired
            self._deadman_candidate = None
            return desired

        if self._deadman_candidate is None or desired != self._deadman_candidate:
            self._deadman_candidate = desired
            self._deadman_candidate_since_ns = now_ns
            return self._deadman_filtered

        if desired == self._deadman_filtered:
            return self._deadman_filtered

        confirm_sec = self.deadman_engage_confirm_sec if desired else self.deadman_release_confirm_sec
        confirm_ns = int(max(0.0, float(confirm_sec)) * 1e9)
        if now_ns - self._deadman_candidate_since_ns >= confirm_ns:
            self._deadman_filtered = desired
            self._deadman_candidate = None
        return self._deadman_filtered

    def _validate_axis_mapping(self, mapping: List[int]) -> List[int]:
        if len(mapping) != 3:
            self.get_logger().warn("axis_mapping must have 3 entries; falling back to [0, 1, 2]")
            return [0, 1, 2]
        if sorted(mapping) != [0, 1, 2]:
            self.get_logger().warn("axis_mapping must be a permutation of [0,1,2]; falling back to [0,1,2]")
            return [0, 1, 2]
        return list(mapping)

    def _validate_axis_sign(self, signs: List[float]) -> np.ndarray:
        if len(signs) != 3:
            self.get_logger().warn("orientation_axis_sign must have 3 entries; falling back to [1,1,1]")
            return np.array([1.0, 1.0, 1.0], dtype=np.float64)
        out = []
        for s in signs:
            try:
                sf = float(s)
            except Exception:
                sf = 1.0
            if sf == 0.0:
                sf = 1.0
            out.append(1.0 if sf > 0.0 else -1.0)
        return np.array(out, dtype=np.float64)

    def _validate_axis_index_list(self, axis_indices: List[int]) -> List[int]:
        if len(axis_indices) != 3:
            self.get_logger().warn("axis index list must have 3 entries; falling back to [0,1,2]")
            return [0, 1, 2]
        out: List[int] = []
        for i, idx in enumerate(axis_indices):
            try:
                axis = int(idx)
            except Exception:
                self.get_logger().warn(
                    f"axis index at position {i} is invalid ({idx}); falling back to [0,1,2]"
                )
                return [0, 1, 2]
            if axis < 0:
                self.get_logger().warn(
                    f"axis index at position {i} must be >= 0 (got {axis}); falling back to [0,1,2]"
                )
                return [0, 1, 2]
            out.append(axis)
        return out

    def _pose_callback(self, msg: PoseStamped) -> None:
        self.latest_robot_pose = msg.pose

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # pragma: no cover - defensive for camera errors
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        height, width, _ = cv_image.shape
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        status_text = "IDLE"
        status_color = (0, 0, 255)
        wrist_source = "-"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            wrist = hand_landmarks.landmark[0]
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]

            wrist_pos, wrist_source = self._get_hand_position(wrist, width, height)
            hand_quat, hand_ori_source = self._get_hand_orientation_quat(
                results, wrist, index_mcp, pinky_mcp, width, height
            )
            now_ns = self.get_clock().now().nanoseconds

            if self.space_deadman_backend == "pynput":
                key_deadman = self._space_down
            else:
                key_deadman = now_ns < self._space_deadman_until_ns

            desired_deadman = bool(key_deadman)
            deadman = self._deadman_filter(desired_deadman, now_ns)

            # Gripper command from pinch distance (prefers metric if depth+info available)
            gripper_cmd = self._gripper_from_distance(thumb_tip, index_tip, width, height)
            if not self.gripper_requires_deadman or deadman:
                self._publish_gripper(gripper_cmd)

            if deadman and self.latest_robot_pose is not None and wrist_pos is not None:
                if not self.deadman_active:
                    # Arm was idle; capture anchors
                    self.initial_hand_pos = wrist_pos.copy()
                    self.initial_hand_quat = hand_quat.copy() if hand_quat is not None else None
                    self.initial_robot_pos = self._position_to_array(self.latest_robot_pose.position)
                    self.initial_robot_orientation = self._orientation_to_array(
                        self.latest_robot_pose.orientation
                    )
                    self.pos_filter.reset(self.initial_robot_pos.copy())
                    self.deadman_active = True
                    self.get_logger().info("Deadman engaged; starting relative control")

                target_pose = self._compute_target_pose(wrist_pos, hand_quat)
                if target_pose is not None:
                    self._publish_target_pose(target_pose)
                    status_text = "CONTROLLING"
                    status_color = (0, 200, 0)
                else:
                    status_text = "WAITING_POSE"
                    status_color = (0, 165, 255)
            else:
                if self.deadman_active:
                    self.get_logger().info("Deadman released; holding position")
                self.deadman_active = False
                self.initial_hand_pos = None
                self.initial_hand_quat = None
                if self.latest_robot_pose is not None:
                    # Hold position by re-publishing the latest pose
                    hold_pose = PoseStamped()
                    hold_pose.header = msg.header
                    hold_pose.header.frame_id = self.target_frame_id
                    hold_pose.pose = self.latest_robot_pose
                    self.target_pose_pub.publish(hold_pose)

            if deadman and self.latest_robot_pose is not None and wrist_pos is None:
                status_text = "WAITING_DEPTH" if self.hand_position_source != "normalized" else "WAITING_HAND"
                status_color = (0, 165, 255)

            # Visual aids
            thumb_px = (int(thumb_tip.x * width), int(thumb_tip.y * height))
            index_px = (int(index_tip.x * width), int(index_tip.y * height))
            cv2.line(cv_image, thumb_px, index_px, (255, 255, 0), 2)
            cv2.circle(cv_image, thumb_px, 4, (255, 0, 255), -1)
            cv2.circle(cv_image, index_px, 4, (0, 255, 255), -1)
        else:
            # No hand detected
            if self.deadman_active:
                self.get_logger().info("Hand lost; stopping motion")
            self.deadman_active = False
            self.initial_hand_pos = None
            self.initial_hand_quat = None

        cv2.putText(
            cv_image,
            f"{status_text} ({wrist_source},{hand_ori_source})" if results.multi_hand_landmarks else status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            status_color,
            3,
        )
        cv2.imshow("Teleop", cv_image)

        # Keyboard handling (space deadman)
        key = cv2.waitKey(1) & 0xFF
        if self.space_deadman_backend == "opencv" and key == 32:  # space
            now_ns = self.get_clock().now().nanoseconds
            # latch window extended by key repeats
            hold_ns = int(max(0.0, self.space_deadman_hold_sec) * 1e9)
            self._space_deadman_until_ns = now_ns + hold_ns

    def _get_hand_position(self, wrist_landmark, width: int, height: int) -> Tuple[Optional[np.ndarray], str]:
        """Return hand position vector for relative control.

        - depth: metric 3D (meters) in camera coordinates via depth+intrinsics
        - normalized: mediapipe normalized (x,y in [0,1], z relative)
        - hybrid: prefer depth, fallback to normalized
        """

        source = self.hand_position_source
        if source not in {"depth", "normalized", "hybrid"}:
            source = "hybrid"

        if source in {"depth", "hybrid"}:
            p3d = self._landmark_3d_m(wrist_landmark, width, height)
            if p3d is not None:
                return p3d.astype(np.float32), "DEPTH"
            if source == "depth":
                return None, "DEPTH_NONE"

        return np.array([wrist_landmark.x, wrist_landmark.y, wrist_landmark.z], dtype=np.float32), "NORM"

    def _gripper_from_distance(self, thumb_tip, index_tip, width: int, height: int) -> float:
        dist_m = self._physical_distance_m(thumb_tip, index_tip, width, height)
        if dist_m is None and self._last_metric_gripper_dist_m is not None:
            now_ns = self.get_clock().now().nanoseconds
            hold_ns = int(max(0.0, self.gripper_metric_hold_sec) * 1e9)
            if now_ns - self._last_metric_gripper_dist_ns <= hold_ns:
                dist_m = float(self._last_metric_gripper_dist_m)

        if dist_m is not None:
            now_ns = self.get_clock().now().nanoseconds
            self._last_metric_gripper_dist_m = float(dist_m)
            self._last_metric_gripper_dist_ns = now_ns
            u = clamp(
                (dist_m - self.gripper_close_dist_m)
                / (self.gripper_open_dist_m - self.gripper_close_dist_m),
                0.0,
                1.0,
            )
        else:
            dist_px = self._distance_px(thumb_tip, index_tip, width, height)
            u = clamp(
                (dist_px - self.gripper_close_dist_px)
                / (self.gripper_open_dist_px - self.gripper_close_dist_px),
                0.0,
                1.0,
            )
        # u=0 -> closed, u=1 -> open; invert for command (1 closed)
        return 1.0 - u

    def _distance_px(self, p1, p2, width: int, height: int) -> float:
        x1, y1 = p1.x * width, p1.y * height
        x2, y2 = p2.x * width, p2.y * height
        return math.hypot(x2 - x1, y2 - y1)

    def _physical_distance_m(self, p1, p2, width: int, height: int) -> Optional[float]:
        if self.depth_image is None or self.depth_info is None or self.fx is None:
            return None
        uv1 = self._landmark_px(p1, width, height)
        uv2 = self._landmark_px(p2, width, height)
        d1 = self._get_depth_m(*uv1)
        d2 = self._get_depth_m(*uv2)
        if d1 is None and d2 is None:
            return None
        # Use a shared depth estimate to reduce noise/jitter from two independent depth samples.
        if d1 is None:
            depth = float(d2)
        elif d2 is None:
            depth = float(d1)
        else:
            depth = 0.5 * (float(d1) + float(d2))
        p3d_1 = self._deproject(uv1, depth)
        p3d_2 = self._deproject(uv2, depth)
        return float(np.linalg.norm(p3d_2 - p3d_1))

    def _landmark_px(self, landmark, width: int, height: int) -> Tuple[int, int]:
        u = int(landmark.x * width)
        v = int(landmark.y * height)
        return u, v

    def _get_depth_m(self, u: int, v: int) -> Optional[float]:
        # Use a small window median to reduce noise; ignore zeros/NaNs
        if self.depth_image is None:
            return None
        h, w = self.depth_image.shape[:2]
        if u < 0 or v < 0 or u >= w or v >= h:
            return None
        window = self.depth_image[max(v - 2, 0) : min(v + 3, h), max(u - 2, 0) : min(u + 3, w)]
        values = window.flatten()
        if values.dtype == np.uint16:
            vals = values.astype(np.float32) * self.depth_unit_scale
        else:
            vals = values.astype(np.float32)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size == 0:
            return None
        depth = float(np.median(vals))
        if self.depth_min_m > 0.0 and depth < self.depth_min_m:
            return None
        if self.depth_max_m > 0.0 and depth > self.depth_max_m:
            return None
        return depth

    def _landmark_3d_m(self, landmark, width: int, height: int) -> Optional[np.ndarray]:
        if self.depth_image is None or self.depth_info is None or self.fx is None:
            return None
        uv = self._landmark_px(landmark, width, height)
        depth_m = self._get_depth_m(*uv)
        if depth_m is None:
            return None
        return self._deproject(uv, depth_m)

    def _deproject(self, uv: Tuple[int, int], depth_m: float) -> np.ndarray:
        u, v = uv
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return np.array([x, y, z], dtype=np.float32)

    def _compute_target_pose(
        self, current_hand_pos: np.ndarray, current_hand_quat: Optional[np.ndarray]
    ) -> Optional[PoseStamped]:
        if self.initial_hand_pos is None or self.initial_robot_pos is None or self.initial_robot_orientation is None:
            return None

        delta_hand = current_hand_pos - self.initial_hand_pos
        mapped_delta = np.array(
            [delta_hand[self.axis_mapping[0]], delta_hand[self.axis_mapping[1]], delta_hand[self.axis_mapping[2]]],
            dtype=np.float32,
        )
        target_pos = self.initial_robot_pos + mapped_delta * self.scale_factor

        # Filter target position
        filtered_pos = self.pos_filter.apply(target_pos)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.target_frame_id
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(filtered_pos[0])
        pose_msg.pose.position.y = float(filtered_pos[1])
        pose_msg.pose.position.z = float(filtered_pos[2])

        q_robot_start = self.initial_robot_orientation.astype(np.float64)
        q_target = q_robot_start
        if (
            self.orientation_mode == "hand_relative"
            and self.initial_hand_quat is not None
            and current_hand_quat is not None
        ):
            q_hand_start = self.initial_hand_quat.astype(np.float64)
            q_hand_curr = current_hand_quat.astype(np.float64)
            # Qdelta = Qhand_current * Qhand_start^{-1}
            # NOTE: Using the opposite order typically results in reversed rotation directions
            # under common "orientation of frame" conventions.
            q_delta = quat_multiply_xyzw(q_hand_curr, quat_conjugate_xyzw(q_hand_start))
            q_delta = quat_normalize_xyzw(q_delta)

            # Map delta rotation axes if needed
            rotvec = quat_to_rotvec_xyzw(q_delta)
            mapped = np.array(
                [
                    rotvec[self.orientation_axis_mapping[0]] * self.orientation_axis_sign[0],
                    rotvec[self.orientation_axis_mapping[1]] * self.orientation_axis_sign[1],
                    rotvec[self.orientation_axis_mapping[2]] * self.orientation_axis_sign[2],
                ],
                dtype=np.float64,
            )
            q_delta_mapped = rotvec_to_quat_xyzw(mapped)

            # Qtarget = Qrobot_start * Qdelta_mapped
            q_target = quat_multiply_xyzw(q_robot_start, q_delta_mapped)
            q_target = quat_normalize_xyzw(q_target)

        pose_msg.pose.orientation.x = float(q_target[0])
        pose_msg.pose.orientation.y = float(q_target[1])
        pose_msg.pose.orientation.z = float(q_target[2])
        pose_msg.pose.orientation.w = float(q_target[3])
        return pose_msg

    def _get_hand_orientation_quat(
        self,
        results,
        wrist_lm,
        index_mcp_lm,
        pinky_mcp_lm,
        width: int,
        height: int,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Compute hand orientation quaternion from wrist/index_mcp/pinky_mcp.

        Priority:
        1) MediaPipe multi_hand_world_landmarks (more stable 3D geometry)
        2) Depth deprojection of the three landmarks (if depth+intrinsics available)
        """

        # 1) World landmarks (meters) if available
        try:
            if getattr(results, "multi_hand_world_landmarks", None):
                wlms = results.multi_hand_world_landmarks[0].landmark
                p_w = np.array([wlms[0].x, wlms[0].y, wlms[0].z], dtype=np.float64)
                p_i = np.array([wlms[5].x, wlms[5].y, wlms[5].z], dtype=np.float64)
                p_p = np.array([wlms[17].x, wlms[17].y, wlms[17].z], dtype=np.float64)
                q = self._hand_quat_from_points(p_w, p_i, p_p)
                return (q.astype(np.float64) if q is not None else None), "WLD" if q is not None else "WLD_NONE"
        except Exception:
            pass

        # 2) Depth-based 3D points in camera coords
        p_w = self._landmark_3d_m(wrist_lm, width, height)
        p_i = self._landmark_3d_m(index_mcp_lm, width, height)
        p_p = self._landmark_3d_m(pinky_mcp_lm, width, height)
        if p_w is not None and p_i is not None and p_p is not None:
            q = self._hand_quat_from_points(
                p_w.astype(np.float64), p_i.astype(np.float64), p_p.astype(np.float64)
            )
            return (q.astype(np.float64) if q is not None else None), "DEP" if q is not None else "DEP_NONE"

        return None, "ORI_NONE"

    def _hand_quat_from_points(
        self, p_wrist: np.ndarray, p_index: np.ndarray, p_pinky: np.ndarray
    ) -> Optional[np.ndarray]:
        x_axis = normalize_vector(p_index - p_wrist)
        y_hint = normalize_vector(p_pinky - p_wrist)
        z_axis = np.cross(x_axis, y_hint)
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-9:
            return None
        z_axis = z_axis / z_norm
        y_axis = np.cross(z_axis, x_axis)
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm < 1e-9:
            return None
        y_axis = y_axis / y_norm
        x_axis = normalize_vector(x_axis)

        r = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns
        return rotmat_to_quat_xyzw(r)

    def _position_to_array(self, position) -> np.ndarray:
        return np.array([position.x, position.y, position.z], dtype=np.float32)

    def _orientation_to_array(self, orientation) -> np.ndarray:
        return np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float32,
        )

    def _publish_target_pose(self, pose: PoseStamped) -> None:
        self.target_pose_pub.publish(pose)

    def _publish_gripper(self, cmd: float) -> None:
        cmd_clamped = float(clamp(cmd, 0.0, 1.0))
        step = self.gripper_step if self.gripper_step > 0 else 0.2
        quantized = round(cmd_clamped / step) * step
        quantized = float(clamp(quantized, 0.0, 1.0))
        msg = Float32()
        msg.data = quantized
        self.gripper_pub.publish(msg)
        self._call_gripper_service(quantized)

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_image = depth
            self.depth_encoding = msg.encoding
            self.depth_stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        except Exception as exc:  # pragma: no cover - defensive
            self.get_logger().warn(f"Depth conversion failed: {exc}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.depth_info = msg
        if len(msg.k) == 9:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def _call_gripper_service(self, closure: float) -> None:
        closure = float(clamp(closure, 0.0, 1.0))
        now_ns = self.get_clock().now().nanoseconds
        if (
            self._last_gripper_cmd is not None
            and abs(closure - self._last_gripper_cmd) < self._gripper_delta_threshold
            and now_ns - self._last_gripper_call_ns < self._gripper_min_interval_ns
        ):
            return
        if not self.gripper_client.service_is_ready():
            if not self.gripper_client.wait_for_service(timeout_sec=0.0):
                return
        request = SetClosure.Request()
        request.closure = closure
        request.duration_sec = self.gripper_duration_sec
        request.speed_ratio = self.gripper_speed_ratio
        try:
            self.gripper_client.call_async(request)
            self._last_gripper_cmd = closure
            self._last_gripper_call_ns = now_ns
        except Exception as exc:  # pragma: no cover - best effort
            self.get_logger().warn(f"SoftHand call failed: {exc}")


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
