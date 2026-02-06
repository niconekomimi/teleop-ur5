#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _default_python_executable() -> str:
    # Prefer an activated venv/conda env if present; otherwise fall back to PATH lookup.
    for prefix_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        prefix = os.environ.get(prefix_var)
        if not prefix:
            continue
        candidate = os.path.join(prefix, "bin", "python3")
        if os.path.exists(candidate):
            return candidate
    return "python3"

def generate_launch_description():
    # 获取包路径
    pkg_share = get_package_share_directory("teleop_control_py")
    default_params = os.path.join(pkg_share, "config", "teleop_params.yaml")

    # 声明参数文件路径参数
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to teleop parameter file",
    )

    python_executable_arg = DeclareLaunchArgument(
        "python_executable",
        default_value=_default_python_executable(),
        description="Python executable used to run teleop_control_node (needs mediapipe)",
    )

    return LaunchDescription(
        [
            params_file_arg,
            python_executable_arg,
            
            # 1. 手势识别节点
            # executable 直接填写 setup.py 中 console_scripts 定义的名字
            # 注意：console_scripts 生成的脚本 shebang 可能固定为 /usr/bin/python3。
            # 这里用指定的 python 直接运行模块，确保使用你安装了 mediapipe 的解释器。
            ExecuteProcess(
                name="teleop_control_node",
                cmd=[
                    LaunchConfiguration("python_executable"),
                    "-m",
                    "teleop_control_py.teleop_control_node",
                    "--ros-args",
                    "-r",
                    "__node:=teleop_control_node",
                    "--params-file",
                    LaunchConfiguration("params_file"),
                ],
                output="screen",
            ),

            # 2. 伺服跟随节点
            Node(
                package="teleop_control_py",
                executable="servo_pose_follower",
                name="servo_pose_follower",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )