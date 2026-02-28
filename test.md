# 遥操作（Teleop）

## 启动

```bash
# RealSense（用于手部追踪 可选：用 disparity/spatial/temporal 滤波）
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true filters:=disparity,spatial,temporal

# UR5 驱动
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.211 launch_rviz:=false reverse_ip:=192.168.1.10

# MoveIt Servo（teleop 依赖 servo）
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_rviz:=false launch_servo:=true

# 软手（如果你用 qbsofthand）
ros2 run qbsofthand_control qbsofthand_control_node

# teleop
ros2 launch teleop_control_py teleop_control.launch.py
```


# 录制（DataCollector / LIBERO HDF5）


##  运行 DataCollectorNode（用 YAML 参数）

先配置：`src/teleop_control_py/config/data_collector_params.yaml`
（相机映射、夹爪类型、home_joint_positions、输出路径）

```bash
ros2 run teleop_control_py data_collector_node --ros-args --params-file src/teleop_control_py/config/data_collector_params.yaml
```

## 4) 开始/停止录制 + go_home

```bash
# 开始
ros2 service call /data_collector/start std_srvs/srv/Trigger {}
# 停止
ros2 service call /data_collector/stop  std_srvs/srv/Trigger {}
# 回home点
ros2 service call /data_collector/go_home std_srvs/srv/Trigger {}

```