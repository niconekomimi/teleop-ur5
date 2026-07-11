"""Model inference controls for the main window."""

from __future__ import annotations

from typing import Protocol

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QTextEdit,
    QWidget,
)

from ..theme import set_standard_icon, set_widget_role


def _horizontal_row(*items: tuple[QWidget, int]) -> QWidget:
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    for widget, stretch in items:
        row_layout.addWidget(widget, stretch)
    return row


class InferenceSettings(Protocol):
    default_openpi_host: str
    default_openpi_port: int
    default_openpi_prompt: str
    collect_inference_action_logs: bool


class InferencePanel(QGroupBox):
    """Build the local and remote model inference controls."""

    def __init__(
        self,
        settings: InferenceSettings,
        *,
        section_style: str,
        button_height: int = 30,
        emphasis_spin_height: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("模型推理", parent)
        self.setStyleSheet(section_style)
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(3, 2)
        layout.setColumnStretch(5, 2)

        self.lbl_inference_backend = QLabel("推理后端:")
        self.inference_backend_combo = QComboBox()
        self.inference_backend_combo.addItem("real_il (本地)", "real_il")
        self.inference_backend_combo.addItem("openpi remote (远端)", "openpi_remote")

        self.lbl_openpi_host = QLabel("OpenPI Host:")
        self.inference_openpi_host_input = QLineEdit(settings.default_openpi_host)
        self.inference_openpi_host_input.setPlaceholderText("例如: 127.0.0.1")

        self.lbl_openpi_port = QLabel("端口:")
        self.inference_openpi_port_spin = QSpinBox()
        self.inference_openpi_port_spin.setRange(1, 65535)
        self.inference_openpi_port_spin.setValue(int(settings.default_openpi_port))

        backend_row = _horizontal_row(
            (self.lbl_inference_backend, 0),
            (self.inference_backend_combo, 2),
            (self.lbl_openpi_host, 0),
            (self.inference_openpi_host_input, 2),
            (self.lbl_openpi_port, 0),
            (self.inference_openpi_port_spin, 1),
        )
        layout.addWidget(backend_row, 0, 0, 1, 6)

        self.lbl_openpi_prompt = QLabel("Prompt:")
        self.inference_openpi_prompt_input = QLineEdit(settings.default_openpi_prompt)
        self.inference_openpi_prompt_input.setPlaceholderText("例如: pick up the object")
        layout.addWidget(
            _horizontal_row(
                (self.lbl_openpi_prompt, 0),
                (self.inference_openpi_prompt_input, 1),
            ),
            1,
            0,
            1,
            6,
        )

        self.lbl_inference_model_dir = QLabel("模型文件夹:")
        self.inference_model_dir_input = QLineEdit()
        self.inference_model_dir_input.setPlaceholderText("例如: models/ddim_dec_transformer")
        self.inference_model_dir_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_browse_inference_model = QPushButton("选择")
        self.btn_browse_inference_model.setMinimumHeight(button_height)
        set_widget_role(self.btn_browse_inference_model, "secondary")
        set_standard_icon(
            self.btn_browse_inference_model,
            QStyle.StandardPixmap.SP_DialogOpenButton,
        )

        self.btn_refresh_inference_options = QPushButton("刷新")
        self.btn_refresh_inference_options.setMinimumHeight(button_height)
        set_widget_role(self.btn_refresh_inference_options, "secondary")
        set_standard_icon(
            self.btn_refresh_inference_options,
            QStyle.StandardPixmap.SP_BrowserReload,
        )
        layout.addWidget(
            _horizontal_row(
                (self.lbl_inference_model_dir, 0),
                (self.inference_model_dir_input, 1),
                (self.btn_browse_inference_model, 0),
                (self.btn_refresh_inference_options, 0),
            ),
            2,
            0,
            1,
            6,
        )

        self.lbl_inference_env = QLabel("任务环境:")
        self.inference_env_combo = QComboBox()

        self.lbl_inference_task = QLabel("任务名称:")
        self.inference_task_combo = QComboBox()
        self.inference_task_combo.setMinimumContentsLength(12)
        self.inference_task_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        layout.addWidget(
            _horizontal_row(
                (self.lbl_inference_env, 0),
                (self.inference_env_combo, 1),
                (self.lbl_inference_task, 0),
                (self.inference_task_combo, 1),
            ),
            3,
            0,
            1,
            6,
        )

        self.lbl_inference_embedding = QLabel("Embeddings:")
        self.inference_embedding_input = QLineEdit()
        self.inference_embedding_input.setReadOnly(True)
        self.inference_embedding_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_browse_inference_embedding = QPushButton("手动选择")
        self.btn_browse_inference_embedding.setMinimumHeight(button_height)
        set_widget_role(self.btn_browse_inference_embedding, "secondary")
        set_standard_icon(
            self.btn_browse_inference_embedding,
            QStyle.StandardPixmap.SP_DialogOpenButton,
        )

        self.btn_auto_match_embedding = QPushButton("自动匹配")
        self.btn_auto_match_embedding.setMinimumHeight(button_height)
        set_widget_role(self.btn_auto_match_embedding, "secondary")
        set_standard_icon(
            self.btn_auto_match_embedding,
            QStyle.StandardPixmap.SP_DialogResetButton,
        )
        layout.addWidget(
            _horizontal_row(
                (self.lbl_inference_embedding, 0),
                (self.inference_embedding_input, 1),
                (self.btn_browse_inference_embedding, 0),
                (self.btn_auto_match_embedding, 0),
            ),
            4,
            0,
            1,
            6,
        )

        self.lbl_inference_global_camera = QLabel("全局相机:")
        self.inference_global_camera_combo = QComboBox()

        self.lbl_inference_wrist_camera = QLabel("手部相机:")
        self.inference_wrist_camera_combo = QComboBox()
        layout.addWidget(
            _horizontal_row(
                (self.lbl_inference_global_camera, 0),
                (self.inference_global_camera_combo, 1),
                (self.lbl_inference_wrist_camera, 0),
                (self.inference_wrist_camera_combo, 1),
            ),
            5,
            0,
            1,
            6,
        )

        self.lbl_inference_device = QLabel("运行设备:")
        self.inference_device_combo = QComboBox()
        self.inference_device_combo.addItem("auto", "auto")
        self.inference_device_combo.addItem("cuda", "cuda")
        self.inference_device_combo.addItem("cpu", "cpu")
        index = max(0, self.inference_device_combo.findData("cuda"))
        self.inference_device_combo.setCurrentIndex(index)

        self.lbl_inference_hz = QLabel("高层动作频率(Hz):")
        self.inference_hz_spin = QDoubleSpinBox()
        self.inference_hz_spin.setRange(0.2, 50.0)
        self.inference_hz_spin.setDecimals(1)
        self.inference_hz_spin.setSingleStep(0.5)
        self.inference_hz_spin.setValue(10.0)
        self.inference_hz_spin.setToolTip(
            "控制高层动作输出频率；不等同于完整重规划频率。实际重规划频率约为该值 / replan_every。"
        )
        self.inference_hz_spin.setFixedHeight(emphasis_spin_height)

        self.lbl_inference_runtime_status = QLabel("状态:")
        self.lbl_inference_status = QLabel("未启动")
        self.lbl_inference_status.setWordWrap(True)
        self.lbl_inference_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_inference_status.setMaximumWidth(260)
        set_widget_role(self.lbl_inference_status, "status-neutral")
        layout.addWidget(
            _horizontal_row(
                (self.lbl_inference_device, 0),
                (self.inference_device_combo, 1),
                (self.lbl_inference_hz, 0),
                (self.inference_hz_spin, 1),
                (self.lbl_inference_runtime_status, 0),
                (self.lbl_inference_status, 2),
            ),
            6,
            0,
            1,
            6,
        )

        self.btn_inference = QPushButton("启动推理")
        self.btn_inference.setFixedHeight(button_height)
        self.btn_inference.setCheckable(True)
        set_widget_role(self.btn_inference, "secondary")
        set_standard_icon(self.btn_inference, QStyle.StandardPixmap.SP_MediaPlay)

        self.btn_execute_inference = QPushButton("开始执行任务")
        self.btn_execute_inference.setFixedHeight(button_height)
        self.btn_execute_inference.setCheckable(True)
        self.btn_execute_inference.setEnabled(False)
        set_widget_role(self.btn_execute_inference, "primary")
        set_standard_icon(
            self.btn_execute_inference,
            QStyle.StandardPixmap.SP_MediaPlay,
        )

        self.btn_inference_estop = QPushButton("停止推理输出")
        self.btn_inference_estop.setEnabled(False)
        self.btn_inference_estop.setFixedHeight(button_height)
        set_widget_role(self.btn_inference_estop, "warning")
        set_standard_icon(
            self.btn_inference_estop,
            QStyle.StandardPixmap.SP_MediaStop,
        )

        self.lbl_inference_execute = QLabel("执行:")
        self.lbl_inference_execute_status = QLabel("未使能")
        set_widget_role(self.lbl_inference_execute_status, "status-neutral")

        action_row = _horizontal_row(
            (self.btn_inference, 1),
            (self.btn_execute_inference, 1),
            (self.btn_inference_estop, 1),
            (self.lbl_inference_execute, 0),
            (self.lbl_inference_execute_status, 1),
        )
        layout.addWidget(action_row, 7, 0, 1, 6)

        self.chk_collect_inference_logs = QCheckBox("记录执行段动作日志")
        self.chk_collect_inference_logs.setChecked(bool(settings.collect_inference_action_logs))
        self.chk_collect_inference_logs.setToolTip(
            "勾选后，仅在点击“开始执行任务”期间保存高层动作日志。"
        )
        layout.addWidget(self.chk_collect_inference_logs, 8, 0, 1, 3)

        self.inference_hint_label = QLabel(
            "说明: 直接调用 Real_IL 的 RealRobotPolicy。这里的频率表示高层动作输出频率，"
            "不等同于完整重规划频率；GUI 负责相机采集、预览发布、推理调度和动作下发。"
            "勾选上方开关后，点击开始执行任务会把执行期间的高层动作保存到 "
            "data/inference_action_logs。"
        )
        self.inference_hint_label.setWordWrap(True)
        self.inference_hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        set_widget_role(self.inference_hint_label, "hint")
        self.inference_hint_label.setVisible(False)
        layout.addWidget(self.inference_hint_label, 9, 0, 1, 6)

        self.lbl_inference_action_output = QLabel("动作输出:")
        layout.addWidget(self.lbl_inference_action_output, 10, 0)
        self.inference_action_output = QTextEdit()
        self.inference_action_output.setReadOnly(True)
        self.inference_action_output.setFixedHeight(75)
        self.inference_action_output.setObjectName("actionOutput")
        set_widget_role(self.inference_action_output, "action-output")
        layout.addWidget(self.inference_action_output, 11, 0, 1, 6)

        self._real_il_widgets = [
            self.lbl_inference_model_dir,
            self.inference_model_dir_input,
            self.btn_browse_inference_model,
            self.btn_refresh_inference_options,
            self.lbl_inference_env,
            self.inference_env_combo,
            self.lbl_inference_task,
            self.inference_task_combo,
            self.lbl_inference_embedding,
            self.inference_embedding_input,
            self.btn_browse_inference_embedding,
            self.btn_auto_match_embedding,
            self.lbl_inference_device,
            self.inference_device_combo,
        ]
        self._openpi_widgets = [
            self.lbl_openpi_host,
            self.inference_openpi_host_input,
            self.lbl_openpi_port,
            self.inference_openpi_port_spin,
            self.lbl_openpi_prompt,
            self.inference_openpi_prompt_input,
        ]
