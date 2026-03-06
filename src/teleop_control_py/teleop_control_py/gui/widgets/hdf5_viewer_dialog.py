import os

import h5py
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QMessageBox,
)


class HDF5ViewerDialog(QDialog):
    def __init__(self, initial_hdf5_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 数据集高级回放器")
        self.resize(1150, 750)

        self.hdf5_path = initial_hdf5_path
        self.file_handle = None
        self.current_demo_group = None

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.on_play_timeout)
        self.is_playing = False
        self.base_interval_ms = 100

        self.setup_ui()
        if os.path.exists(self.hdf5_path):
            self.open_hdf5_file(self.hdf5_path)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.btn_open_file = QPushButton("📂 选择 HDF5 文件")
        self.btn_open_file.setStyleSheet("font-weight: bold; background-color: #d0e8f1;")
        self.btn_open_file.clicked.connect(self.open_file_dialog)
        top_layout.addWidget(self.btn_open_file)

        self.lbl_file_path = QLabel(os.path.basename(self.hdf5_path) if self.hdf5_path else "未选择文件")
        self.lbl_file_path.setStyleSheet("color: #555; font-style: italic;")
        top_layout.addWidget(self.lbl_file_path)

        top_layout.addWidget(QLabel("  |  选择录制序列:"))
        self.demo_combo = QComboBox()
        self.demo_combo.currentIndexChanged.connect(self.load_demo)
        top_layout.addWidget(self.demo_combo)

        self.lbl_frame_info = QLabel("当前帧: 0 / 0")
        self.lbl_frame_info.setStyleSheet("font-weight: bold; color: blue;")
        top_layout.addWidget(self.lbl_frame_info)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        ctrl_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⏮ 上一帧")
        self.btn_prev.clicked.connect(self.step_prev)
        ctrl_layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setStyleSheet("font-weight: bold; color: green; min-width: 80px;")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)

        self.btn_next = QPushButton("⏭ 下一帧")
        self.btn_next.clicked.connect(self.step_next)
        ctrl_layout.addWidget(self.btn_next)

        ctrl_layout.addWidget(QLabel("  倍速:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "2.0x", "4.0x", "8.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        ctrl_layout.addWidget(self.speed_combo)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.sliderPressed.connect(self.pause_playback)
        ctrl_layout.addWidget(self.slider)
        main_layout.addLayout(ctrl_layout)

        content_layout = QHBoxLayout()
        cam_layout = QVBoxLayout()

        global_title = QLabel("【全局相机 (Agent View)】")
        global_title.setAlignment(Qt.AlignCenter)
        self.lbl_agent = QLabel("无画面")
        self.lbl_agent.setAlignment(Qt.AlignCenter)
        self.lbl_agent.setStyleSheet("background-color: black; color: white;")
        self.lbl_agent.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        wrist_title = QLabel("【手部相机 (Eye-in-Hand)】")
        wrist_title.setAlignment(Qt.AlignCenter)
        self.lbl_wrist = QLabel("无画面")
        self.lbl_wrist.setAlignment(Qt.AlignCenter)
        self.lbl_wrist.setStyleSheet("background-color: black; color: white;")
        self.lbl_wrist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        cam_layout.addWidget(global_title)
        cam_layout.addWidget(self.lbl_agent)
        cam_layout.addWidget(wrist_title)
        cam_layout.addWidget(self.lbl_wrist)
        content_layout.addLayout(cam_layout, stretch=2)

        self.text_state = QTextEdit()
        self.text_state.setReadOnly(True)
        self.text_state.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; background-color: #f5f5f5;")
        content_layout.addWidget(self.text_state, stretch=1)
        main_layout.addLayout(content_layout)

    def open_file_dialog(self):
        start_dir = os.path.dirname(self.hdf5_path) if self.hdf5_path else os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择已录制的 HDF5 数据集",
            start_dir,
            "HDF5 Files (*.hdf5 *.h5);;All Files (*)",
        )
        if file_path:
            self.open_hdf5_file(file_path)

    def open_hdf5_file(self, path):
        self.pause_playback()
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

        self.hdf5_path = path
        self.lbl_file_path.setText(os.path.basename(path))
        self.setWindowTitle(f"HDF5 数据集高级回放器 - {os.path.basename(path)}")
        self.demo_combo.clear()

        try:
            self.file_handle = h5py.File(path, "r")
            if "data" not in self.file_handle:
                raise KeyError("HDF5 文件中未找到 'data' 根组，格式不符合要求。")

            demos = list(self.file_handle["data"].keys())
            if not demos:
                raise ValueError("数据集中没有任何 demo 序列。")

            demos.sort(key=lambda item: int(item.split("_")[1]) if "_" in item else 0)
            self.demo_combo.addItems(demos)
        except Exception as exc:
            QMessageBox.critical(self, "HDF5 读取错误", str(exc))

    def load_demo(self):
        self.pause_playback()
        demo_name = self.demo_combo.currentText()
        if not demo_name or self.file_handle is None:
            return

        self.current_demo_group = self.file_handle["data"][demo_name]
        num_samples = self.current_demo_group.attrs.get("num_samples", 0)
        if num_samples == 0 and "actions" in self.current_demo_group:
            num_samples = self.current_demo_group["actions"].shape[0]

        if num_samples > 0:
            self.slider.blockSignals(True)
            self.slider.setMaximum(int(num_samples) - 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.update_frame_display()
        else:
            self.lbl_frame_info.setText("空序列")

    def toggle_play(self):
        if self.current_demo_group is None:
            return
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)

        self.is_playing = True
        self.btn_play.setText("⏸ 暂停")
        self.btn_play.setStyleSheet("font-weight: bold; color: red; min-width: 80px;")
        self.change_speed()

    def pause_playback(self):
        self.is_playing = False
        self.playback_timer.stop()
        self.btn_play.setText("▶ 播放")
        self.btn_play.setStyleSheet("font-weight: bold; color: green; min-width: 80px;")

    def change_speed(self):
        text = self.speed_combo.currentText().replace("x", "")
        try:
            factor = float(text)
        except Exception:
            factor = 1.0

        interval = int(self.base_interval_ms / factor)
        if self.is_playing:
            self.playback_timer.start(interval)

    def on_play_timeout(self):
        current = self.slider.value()
        if current < self.slider.maximum():
            self.slider.setValue(current + 1)
        else:
            self.pause_playback()

    def step_prev(self):
        self.pause_playback()
        current = self.slider.value()
        if current > 0:
            self.slider.setValue(current - 1)

    def step_next(self):
        self.pause_playback()
        current = self.slider.value()
        if current < self.slider.maximum():
            self.slider.setValue(current + 1)

    def on_slider_changed(self):
        self.update_frame_display()

    def update_frame_display(self):
        if self.current_demo_group is None:
            return

        idx = self.slider.value()
        total = self.slider.maximum() + 1
        self.lbl_frame_info.setText(f"当前帧: {idx + 1} / {total}")

        try:
            agent_rgb = self.current_demo_group["obs"]["agentview_rgb"][idx]
            wrist_rgb = self.current_demo_group["obs"]["eye_in_hand_rgb"][idx]
            height, width, channels = agent_rgb.shape
            bytes_per_line = channels * width

            qimg_agent = QImage(agent_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.lbl_agent.setPixmap(QPixmap.fromImage(qimg_agent).scaled(self.lbl_agent.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            qimg_wrist = QImage(wrist_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.lbl_wrist.setPixmap(QPixmap.fromImage(qimg_wrist).scaled(self.lbl_wrist.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            joints = self.current_demo_group["obs"]["robot0_joint_pos"][idx]
            pos = self.current_demo_group["obs"]["robot0_eef_pos"][idx]
            quat = self.current_demo_group["obs"]["robot0_eef_quat"][idx]
            actions = self.current_demo_group["actions"][idx]

            formatter = {"float_kind": lambda value: f"{value:6.3f}"}
            joints_str = np.array2string(joints, formatter=formatter)
            pos_str = np.array2string(pos, formatter=formatter)
            quat_str = np.array2string(quat, formatter=formatter)
            actions_str = np.array2string(actions, formatter=formatter)

            text = "【录制帧数据】\n"
            text += "-" * 25 + "\n"
            text += f"► 关节位置 [6]:\n {joints_str}\n\n"
            text += f"► 末端 XYZ [3]:\n {pos_str}\n\n"
            text += f"► 末端 四元数 [4]:\n {quat_str}\n\n"
            text += "-" * 25 + "\n"
            text += f"► 保存的 Action [7]:\n {actions_str}\n"
            text += "  (XYZ, RxRyRz, Gripper)"
            self.text_state.setText(text)
        except Exception as exc:
            self.text_state.setText(f"读取帧数据失败: {exc}")

    def closeEvent(self, event):
        self.pause_playback()
        if self.file_handle is not None:
            self.file_handle.close()
        event.accept()
