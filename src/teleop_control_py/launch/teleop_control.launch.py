#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import LogInfo
from launch.actions import OpaqueFunction
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


def _read_control_mode_from_params(params_file: str) -> str:
    mode = "hand"
    try:
        import yaml  # type: ignore
    except Exception:
        return mode

    try:
        with open(params_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return mode

    if not isinstance(data, dict):
        return mode

    for key in ("teleop_control_node", "/teleop_control_node"):
        block = data.get(key)
        if not isinstance(block, dict):
            continue
        params = block.get("ros__parameters")
        if not isinstance(params, dict):
            continue
        cm = params.get("control_mode")
        if isinstance(cm, str) and cm.strip():
            return cm.strip().lower()
    return mode


def _read_end_effector_from_params(params_file: str) -> str:
    ee = "robotiq"
    try:
        import yaml  # type: ignore
    except Exception:
        return ee

    try:
        with open(params_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return ee

    if not isinstance(data, dict):
        return ee

    for key in ("teleop_control_node", "/teleop_control_node"):
        block = data.get(key)
        if not isinstance(block, dict):
            continue
        params = block.get("ros__parameters")
        if not isinstance(params, dict):
            continue
        raw = params.get("end_effector")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    return ee


def _launch_teleop_node(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file").perform(context)
    override = LaunchConfiguration("control_mode").perform(context).strip().lower()
    if override in ("hand", "xbox"):
        resolved = override
    else:
        resolved = _read_control_mode_from_params(params_file)

    ee_override = LaunchConfiguration("end_effector").perform(context).strip().lower()
    if ee_override in ("auto", "qbsofthand", "robotiq"):
        resolved_ee = ee_override
    else:
        resolved_ee = _read_end_effector_from_params(params_file)

    return [
        LogInfo(msg=f"[teleop_control.launch] resolved control_mode: {resolved}"),
        LogInfo(msg=f"[teleop_control.launch] resolved end_effector: {resolved_ee}"),
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
                "-p",
                f"control_mode:={resolved}",
                "-p",
                f"end_effector:={resolved_ee}",
            ],
            output="screen",
        ),
    ]

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

    control_mode_arg = DeclareLaunchArgument(
        "control_mode",
        default_value="",
        description="Optional control mode override (hand|xbox). Empty means read from params_file.",
    )

    end_effector_arg = DeclareLaunchArgument(
        "end_effector",
        default_value="",
        description="Optional end effector override (qbsofthand|robotiq; 'auto' is a deprecated alias for qbsofthand). Empty means read from params_file.",
    )

    return LaunchDescription(
        [
            params_file_arg,
            python_executable_arg,
            control_mode_arg,
            end_effector_arg,
            
            # 1. 手势识别节点
            # executable 直接填写 setup.py 中 console_scripts 定义的名字
            # 注意：console_scripts 生成的脚本 shebang 可能固定为 /usr/bin/python3。
            # 这里用指定的 python 直接运行模块，确保使用你安装了 mediapipe 的解释器。
            OpaqueFunction(function=_launch_teleop_node),

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