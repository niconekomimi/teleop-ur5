# teleop-ur5
# 本项目是一个基于 MediaPipe 的手势识别控制 UR5+SoftHand 的项目

## 快速启动
```bash
# 启动整个控制系统（手、机械臂、相机）
ros2 launch teleop_control_py control_system.launch.py
```

## Robotiq（可选）：用 udev 固定串口名

本工程通过 `end_effector:=robotiq` 手动选择 Robotiq 驱动，并用 `robotiq_serial_port` 指定串口路径。推荐使用 `/dev/robotiq_gripper` 这种 udev 软链接（避免 `/dev/ttyUSB*` 因插拔顺序变化而漂移）。

### 1）找到你的 Robotiq USB 设备信息

先插上夹爪 USB，查看它对应的真实设备（例如 `/dev/ttyUSB0`）：

```bash
ls -l /dev/ttyUSB*
```

然后用 `udevadm` 查询属性（把 `/dev/ttyUSB0` 换成你的）：

```bash
udevadm info -a -n /dev/ttyUSB0 | head -n 80
```

通常你会用到 `idVendor`、`idProduct`，更稳一点可以再加序列号（不同硬件字段名可能不同，例如 `ATTRS{serial}`）。

### 2）写 udev rule，创建稳定软链接 `/dev/robotiq_gripper`

新建规则文件：

```bash
sudo nano /etc/udev/rules.d/99-robotiq-gripper.rules
```

示例（请把 `xxxx/yyyy/your_serial` 换成你机器上查到的值；若没有 serial 字段就删掉那一段）：

```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="yyyy", ATTRS{serial}=="your_serial", SYMLINK+="robotiq_gripper", MODE:="0666"
```

重载并触发：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/robotiq_gripper
```

### 3）启动时使用（手动选择）

- 显式指定：

```bash
ros2 launch teleop_control_py control_system.launch.py end_effector:=robotiq robotiq_serial_port:=/dev/robotiq_gripper
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
# 注意：即使你激活了 venv，`colcon` 也可能仍然指向系统的 /usr/bin/colcon（系统 Python）。
# 为了让生成的 Python 可执行脚本绑定到当前 venv（从而能用到 venv 里 pip 安装的依赖），建议统一用：
python -m colcon build --packages-select teleop_control_py
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

