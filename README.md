# teleop-ur5
# 本项目是一个基于 MediaPipe 的手势识别控制 UR5+SoftHand 的项目

## 快速启动
```bash
# 启动整个控制系统（手、机械臂、相机）
ros2 launch teleop_control_py control_system.launch.py
```

## 激活环境
    sudo apt install python3-venv
    python3 -m venv ~/clds --system-site-packages
    source ~/clds/bin/activate

**VSCode 自动激活**：本工作区已配置自动激活环境。新打开的终端会自动执行：
- Python 虚拟环境 `~/clds`
- ROS2 Humble
- 当前工作区的 `install/setup.bash`

## 需要安装
 - ur 驱动  
 - realsence 驱动   
 - oak 驱动 

### 依赖安装

#### ur驱动
参考https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/tree/humble?tab=readme-ov-file

#### realsence驱动

##### 第一步
```bash
    # 1. 安装必要的工具
    sudo apt-get install -y curl gnupg2 lsb-release

    # 2. 建立密钥目录
    sudo mkdir -p /etc/apt/keyrings

    # 3. 下载并注册 Intel 的公钥
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

    # 4. 将 Intel 仓库添加到你的源列表中
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/librealsense.list

    # 5. 更新源
    sudo apt-get update
```
##### 第二步
```bash
    # 1. 安装底层驱动和调试工具 (realsense-viewer)
    sudo apt-get install -y librealsense2-dkms librealsense2-utils

    # 2. 安装 ROS 2 Humble 的 RealSense 驱动包
    sudo apt-get install -y ros-humble-realsense2-camera
```
#### 其它依赖安装
```bash
    pip install -r requirements.txt
```

## Teleop 控制包
- 包路径：src/teleop_control_py
- 主要节点：teleop_control_node（输入 RealSense 图像，输出末端位姿 PoseStamped 到 /target_pose，夹爪 Float32 到 /gripper/cmd）
- 默认参数：config/teleop_params.yaml（话题名、缩放、轴映射、低通 alpha、捏合阈值、夹爪距离映射）

### 构建与运行
```bash
# 构建（只构建本包）
colcon build --packages-select teleop_control_py
source install/setup.bash

# 运行，自动加载默认参数
ros2 launch teleop_control_py teleop_control.launch.py 

# 如需覆盖参数，例如切换图像话题或缩放
ros2 launch teleop_control_py teleop_control.launch.py \
    params_file:=src/teleop_control_py/config/teleop_params.yaml \
    image_topic:=/camera/camera/color/image_raw \
    scale_factor:=0.3
```

### 控制逻辑概述
- Deadman：支持两种来源，默认“捏合或空格”。可在 YAML 将 `use_pinch_deadman` 设为 `false`，改为“仅空格”；此时捏合只控制夹爪，不再作为安全开关。
- 相对控制：Deadman 激活时记录初始手腕位置与机器人位姿，之后用手腕增量经轴映射与缩放得到 TCP 目标。
- 平滑：位置使用指数低通，alpha 越小越平滑。
- 夹爪：拇指-食指距离 20px→闭合，100px→张开，线性映射。

