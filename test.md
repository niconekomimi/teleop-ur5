# realsence相机
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true filters:=spatial,temporal

# 手
ros2 run qbsofthand_control qbsofthand_control_node

# ur5
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.211 launch_rviz:=false reverse_ip:=192.168.1.10

# moveit
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_rviz:=false launch_servo:=true

# teleop
ros2 launch teleop_control_py teleop_control.launch.py

# 测试相机

