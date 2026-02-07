#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.duration import Duration
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration as DurationMsg
from tf2_ros import Buffer, TransformException, TransformListener


def quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate for ROS ordering (x, y, z, w)."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication for ROS ordering (x, y, z, w)."""
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def quat_to_vec_angle(q) -> tuple[float, np.ndarray]:
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm == 0.0:
        return 0.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)
    qx, qy, qz, qw = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)
    angle = 2.0 * math.acos(clamp(qw, -1.0, 1.0))
    s = math.sqrt(1.0 - qw * qw)
    if s < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = np.array([qx / s, qy / s, qz / s], dtype=np.float32)
    return angle, axis


class ServoPoseFollower(Node):
    """Convert Pose targets to MoveIt Servo Twist commands and manage Servo state."""

    def __init__(self) -> None:
        super().__init__("servo_pose_follower")

        # Parameters
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("linear_gain", 2.0)
        self.declare_parameter("angular_gain", 1.5)
        self.declare_parameter("max_linear_speed", 0.5)
        self.declare_parameter("max_angular_speed", 1.5)
        self.declare_parameter("position_deadband", 0.002)
        self.declare_parameter("rotation_deadband", 0.01)
        self.declare_parameter("command_rate_hz", 100.0)

        # Servo Startup & Controller Switching
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("auto_switch_controllers", True)
        self.declare_parameter("controller_manager_ns", "/controller_manager")
        # 默认使用 forward_position_controller（与 Universal Robots 的 ur_servo.yaml 默认输出一致）
        self.declare_parameter("activate_controller", "forward_position_controller")
        self.declare_parameter(
            "deactivate_controllers",
            ["scaled_joint_trajectory_controller"],
        )
        self.declare_parameter("switch_strictness", "best_effort")
        self.declare_parameter("startup_retry_period_sec", 1.0)

        # Retrieve parameters
        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        self.servo_twist_topic = self.get_parameter("servo_twist_topic").value
        self.planning_frame = self.get_parameter("planning_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.linear_gain = self.get_parameter("linear_gain").value
        self.angular_gain = self.get_parameter("angular_gain").value
        self.max_linear_speed = self.get_parameter("max_linear_speed").value
        self.max_angular_speed = self.get_parameter("max_angular_speed").value
        self.position_deadband = self.get_parameter("position_deadband").value
        self.rotation_deadband = self.get_parameter("rotation_deadband").value
        self.command_period = 1.0 / self.get_parameter("command_rate_hz").value

        self.auto_start_servo = self.get_parameter("auto_start_servo").value
        self.start_servo_service = self.get_parameter("start_servo_service").value
        self.auto_switch_controllers = self.get_parameter("auto_switch_controllers").value
        self.controller_manager_ns = self.get_parameter("controller_manager_ns").value.rstrip("/")
        self.activate_controller = self.get_parameter("activate_controller").value
        self.deactivate_controllers = list(self.get_parameter("deactivate_controllers").value)
        self.switch_strictness = self.get_parameter("switch_strictness").value
        self.startup_retry_period_sec = self.get_parameter("startup_retry_period_sec").value

        # Publishers / Subscribers
        self.cmd_pub = self.create_publisher(TwistStamped, self.servo_twist_topic, 10)
        self.target_sub = self.create_subscription(
            PoseStamped, self.target_pose_topic, self._target_callback, 10
        )

        # TF
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # State
        self.latest_target: Optional[PoseStamped] = None
        self._servo_started = False
        self._controller_switched = False
        self._startup_inflight = False

        # Service Clients
        self._start_servo_client = self.create_client(Trigger, self.start_servo_service)
        self._switch_ctrl_client = self.create_client(
            SwitchController, f"{self.controller_manager_ns}/switch_controller"
        )
        self._list_ctrl_client = self.create_client(
            ListControllers, f"{self.controller_manager_ns}/list_controllers"
        )

        # Startup Timer
        if self.auto_start_servo or self.auto_switch_controllers:
            self.get_logger().info("Waiting for services to become available...")
            self._startup_timer = self.create_timer(self.startup_retry_period_sec, self._startup_tick)

        # Control Loop
        self.create_timer(self.command_period, self._publish_twist)
        self.get_logger().info(
            f"ServoPoseFollower initialized. Target: {self.target_pose_topic}, Output: {self.servo_twist_topic}"
        )

    def _startup_tick(self) -> None:
        """Periodic check to ensure Servo is started and controllers are switched."""
        if self._startup_inflight:
            return

        # 1. Start Servo
        if self.auto_start_servo and not self._servo_started:
            if self._start_servo_client.service_is_ready():
                self._startup_inflight = True
                self.get_logger().info(f"Calling service: {self.start_servo_service}")
                future = self._start_servo_client.call_async(Trigger.Request())
                future.add_done_callback(self._on_start_servo_done)
            return

        # 2. Switch Controllers
        if self.auto_switch_controllers and not self._controller_switched:
            if self._switch_ctrl_client.service_is_ready():
                self._try_switch_controllers()
            return

        # 3. All done
        if (not self.auto_start_servo or self._servo_started) and \
           (not self.auto_switch_controllers or self._controller_switched):
            self.get_logger().info("Startup sequence complete.")
            self._startup_timer.cancel()

    def _on_start_servo_done(self, future) -> None:
        self._startup_inflight = False
        try:
            resp = future.result()
            if resp.success:
                self._servo_started = True
                self.get_logger().info("SUCCESS: MoveIt Servo started.")
            else:
                self.get_logger().warn(f"FAILED: MoveIt Servo start returned false. Msg: {resp.message}")
        except Exception as exc:
            self.get_logger().warn(f"Service call failed: {exc}")

    def _try_switch_controllers(self) -> None:
        self._startup_inflight = True
        
        # Helper to send switch request
        def send_switch(deactivate_list):
            req = SwitchController.Request()
            req.activate_controllers = [self.activate_controller]
            req.deactivate_controllers = deactivate_list
            req.strictness = SwitchController.Request.BEST_EFFORT if self.switch_strictness == "best_effort" else SwitchController.Request.STRICT
            req.activate_asap = True
            req.timeout = DurationMsg(sec=2, nanosec=0)

            self.get_logger().info(f"Switching: +{self.activate_controller} -{deactivate_list}")
            future = self._switch_ctrl_client.call_async(req)
            future.add_done_callback(self._on_switch_done)

        # Optional: List controllers first to find conflicts intelligently
        if self._list_ctrl_client.service_is_ready():
            future = self._list_ctrl_client.call_async(ListControllers.Request())
            
            def on_list_done(fut):
                try:
                    res = fut.result()
                    to_deactivate = set(self.deactivate_controllers)
                    # 自动停用与目标控制器冲突的“关节命令控制器”（位置/速度/轨迹）。
                    # 这样无需在 YAML 里硬编码 forward_velocity_controller 等名字。
                    for c in res.controller:
                        if c.name == self.activate_controller:
                            continue
                        if c.state != "active":
                            continue
                        ctrl_type = getattr(c, "type", "") or ""
                        if (
                            "JointTrajectoryController" in ctrl_type
                            or "JointGroupPositionController" in ctrl_type
                            or "JointGroupVelocityController" in ctrl_type
                            or "JointGroupEffortController" in ctrl_type
                            or "joint_trajectory_controller" in c.name
                            or "forward_" in c.name
                        ):
                            to_deactivate.add(c.name)
                    send_switch(list(to_deactivate))
                except Exception:
                    send_switch(self.deactivate_controllers)
            
            future.add_done_callback(on_list_done)
        else:
            send_switch(self.deactivate_controllers)

    def _on_switch_done(self, future) -> None:
        self._startup_inflight = False
        try:
            resp = future.result()
            if resp.ok:
                self._controller_switched = True
                self.get_logger().info(f"SUCCESS: Switched to {self.activate_controller}")
            else:
                self.get_logger().warn("FAILED: Controller switch rejected.")
        except Exception as exc:
            self.get_logger().warn(f"Switch service call failed: {exc}")

    def _target_callback(self, msg: PoseStamped) -> None:
        self.latest_target = msg

    def _publish_twist(self) -> None:
        if self.latest_target is None:
            return
        
        # TF Lookup: Get Current Pose
        try:
            tf = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            # Squeal only occasionally or debug to reduce log noise
            return

        # TF Transform: Target -> Planning Frame
        target_in_plan = self._transform_target(self.latest_target)
        if target_in_plan is None:
            return

        # Calculate Errors
        current_pos = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        target_pos = np.array([target_in_plan.pose.position.x, target_in_plan.pose.position.y, target_in_plan.pose.position.z])
        lin_err = target_pos - current_pos

        q_curr = np.array(
            [
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            ],
            dtype=np.float64,
        )
        q_tgt = np.array(
            [
                target_in_plan.pose.orientation.x,
                target_in_plan.pose.orientation.y,
                target_in_plan.pose.orientation.z,
                target_in_plan.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        q_err = quat_multiply_xyzw(q_tgt, quat_conjugate_xyzw(q_curr))
        q_err = quat_normalize_xyzw(q_err)
        # Ensure shortest-arc representation
        if q_err[3] < 0.0:
            q_err *= -1.0
        ang_err_mag, ang_axis = quat_to_vec_angle(q_err)
        ang_err = ang_axis * ang_err_mag

        # Deadband
        if np.linalg.norm(lin_err) < self.position_deadband:
            lin_err[:] = 0.0
        if abs(ang_err_mag) < self.rotation_deadband:
            ang_err[:] = 0.0

        # Gain & Clip
        lin_cmd = lin_err * self.linear_gain
        ang_cmd = ang_err * self.angular_gain

        l_norm = np.linalg.norm(lin_cmd)
        if l_norm > self.max_linear_speed:
            lin_cmd *= (self.max_linear_speed / l_norm)
        
        a_norm = np.linalg.norm(ang_cmd)
        if a_norm > self.max_angular_speed:
            ang_cmd *= (self.max_angular_speed / a_norm)

        # Publish
        msg = TwistStamped()
        msg.header.frame_id = self.planning_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = float(lin_cmd[0]), float(lin_cmd[1]), float(lin_cmd[2])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = float(ang_cmd[0]), float(ang_cmd[1]), float(ang_cmd[2])
        self.cmd_pub.publish(msg)

    def _transform_target(self, target: PoseStamped) -> Optional[PoseStamped]:
        if target.header.frame_id == self.planning_frame:
            return target
        try:
            tf = self.tf_buffer.lookup_transform(
                self.planning_frame,
                target.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None
        
        out = PoseStamped()
        out.header = target.header
        out.header.frame_id = self.planning_frame
        
        # Position
        tx, ty, tz = tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z
        qx, qy, qz, qw = tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w
        q_tf = np.array([qx, qy, qz, qw])
        p_src = np.array([target.pose.position.x, target.pose.position.y, target.pose.position.z])
        
        # Rotate p_src by q_tf
        uv = np.cross(q_tf[:3], p_src)
        uuv = np.cross(q_tf[:3], uv)
        p_rot = p_src + 2.0 * (q_tf[3] * uv + uuv)
        
        out.pose.position.x = p_rot[0] + tx
        out.pose.position.y = p_rot[1] + ty
        out.pose.position.z = p_rot[2] + tz
        
        # Orientation
        q_src = np.array(
            [
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
                target.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        q_new = quat_multiply_xyzw(q_tf.astype(np.float64), q_src)
        q_new = quat_normalize_xyzw(q_new)
        out.pose.orientation.x = float(q_new[0])
        out.pose.orientation.y = float(q_new[1])
        out.pose.orientation.z = float(q_new[2])
        out.pose.orientation.w = float(q_new[3])
        return out


def main(args=None):
    rclpy.init(args=args)
    node = ServoPoseFollower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
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


if __name__ == "__main__":
    main()