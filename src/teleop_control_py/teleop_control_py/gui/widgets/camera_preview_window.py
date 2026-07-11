"""ROS-independent camera preview and runtime-status dialog."""

from __future__ import annotations

import cv2
from PySide6.QtCore import QSize, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import set_standard_icon, set_widget_role
from .image_viewport import ImageViewport


class _ElidedStatusLabel(QLabel):
    """Keep the complete status available while drawing a width-safe summary."""

    def __init__(self, text: str = "", parent=None):
        self._full_text = ""
        super().__init__(parent)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API compatibility
        self._full_text = str(text)
        self._refresh_display_text()

    def text(self) -> str:
        return self._full_text

    def displayed_text(self) -> str:
        return super().text()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API compatibility
        super().resizeEvent(event)
        self._refresh_display_text()

    def _refresh_display_text(self) -> None:
        available_width = max(0, self.contentsRect().width())
        visible_text = self.fontMetrics().elidedText(
            self._full_text,
            Qt.ElideRight,
            available_width,
        )
        if super().text() != visible_text:
            super().setText(visible_text)


class CameraPreviewWindow(QDialog):
    preview_record_toggle_requested = Signal(bool, str)

    _FAILURE_MARKERS = ("失败", "错误", "异常", "failed", "failure", "error")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实时预览与状态监视器")
        self.setMinimumSize(720, 500)
        self.resize(1100, 700)
        self.show_cropped_only = True
        self._preview_record_output_dir = ""

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)
        self.crop_cb = QCheckBox("仅显示中心裁切区域（与录入数据集画面一致）")
        self.crop_cb.setChecked(True)
        self.crop_cb.toggled.connect(self.on_crop_toggled)
        top_layout.addWidget(self.crop_cb)
        top_layout.addStretch(1)

        self.lbl_preview_source = _ElidedStatusLabel("预览源: 无活动图像源")
        self.lbl_preview_source.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self._make_status_label_compressible(self.lbl_preview_source)
        self._set_status_message(
            self.lbl_preview_source,
            self.lbl_preview_source.text(),
            "status-neutral",
        )
        top_layout.addWidget(self.lbl_preview_source, stretch=1)
        main_layout.addLayout(top_layout)

        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)
        self.lbl_dataset_record_status = _ElidedStatusLabel(
            "采集状态: 未录制 | 时长: 00:00 | 已录制帧数: 0 | 实时录制帧率: 0.00 Hz"
        )
        self._make_status_label_compressible(self.lbl_dataset_record_status)
        self._set_status_message(
            self.lbl_dataset_record_status,
            self.lbl_dataset_record_status.text(),
            "status-neutral",
        )
        status_layout.addWidget(self.lbl_dataset_record_status, stretch=3)

        self.lbl_preview_record_status = _ElidedStatusLabel("预览录屏: 未录制")
        self._make_status_label_compressible(self.lbl_preview_record_status)
        self._set_preview_status_message(
            self.lbl_preview_record_status.text(),
            "status-neutral",
        )
        status_layout.addWidget(self.lbl_preview_record_status, stretch=2)
        main_layout.addLayout(status_layout)

        record_controls_layout = QHBoxLayout()
        record_controls_layout.setSpacing(8)
        record_controls_layout.addWidget(QLabel("预览录屏目标:"))
        self.preview_record_target_combo = QComboBox()
        self.preview_record_target_combo.addItem("全局画面", "global")
        self.preview_record_target_combo.addItem("手部画面", "wrist")
        self.preview_record_target_combo.addItem("全局 + 手部", "both")
        self.preview_record_target_combo.setCurrentIndex(2)
        record_controls_layout.addWidget(self.preview_record_target_combo)

        record_controls_layout.addWidget(QLabel("录屏尺寸:"))
        self.preview_record_frame_mode_combo = QComboBox()
        self.preview_record_frame_mode_combo.addItem("源大小", "source")
        self.preview_record_frame_mode_combo.addItem("中心裁切正方形", "square")
        self.preview_record_frame_mode_combo.setCurrentIndex(0)
        record_controls_layout.addWidget(self.preview_record_frame_mode_combo)

        self.btn_preview_record = QPushButton("开始预览录屏")
        self.btn_preview_record.setCheckable(True)
        self.btn_preview_record.toggled.connect(self._emit_preview_record_toggle)
        set_widget_role(self.btn_preview_record, "primary")
        set_standard_icon(
            self.btn_preview_record,
            QStyle.StandardPixmap.SP_MediaPlay,
        )
        record_controls_layout.addWidget(self.btn_preview_record)
        record_controls_layout.addStretch(1)
        main_layout.addLayout(record_controls_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        camera_widget = QWidget(self)
        cameras_layout = QGridLayout(camera_widget)
        cameras_layout.setContentsMargins(0, 0, 0, 0)
        cameras_layout.setHorizontalSpacing(8)
        cameras_layout.setVerticalSpacing(5)

        global_title = QLabel("全局相机 (Agent View)")
        global_title.setAlignment(Qt.AlignCenter)
        set_widget_role(global_title, "subsection-title")
        self.global_label = ImageViewport(
            "无画面",
            preferred_size=QSize(360, 270),
            minimum_size=QSize(160, 120),
        )
        cameras_layout.addWidget(global_title, 0, 0)
        cameras_layout.addWidget(self.global_label, 1, 0)

        wrist_title = QLabel("手部相机 (Eye-in-Hand)")
        wrist_title.setAlignment(Qt.AlignCenter)
        set_widget_role(wrist_title, "subsection-title")
        self.wrist_label = ImageViewport(
            "无画面",
            preferred_size=QSize(360, 270),
            minimum_size=QSize(160, 120),
        )
        cameras_layout.addWidget(wrist_title, 0, 1)
        cameras_layout.addWidget(self.wrist_label, 1, 1)
        cameras_layout.setColumnStretch(0, 1)
        cameras_layout.setColumnStretch(1, 1)
        cameras_layout.setRowStretch(1, 1)
        content_layout.addWidget(camera_widget, stretch=3)

        robot_state_widget = QWidget(self)
        robot_state_layout = QVBoxLayout(robot_state_widget)
        robot_state_layout.setContentsMargins(0, 0, 0, 0)
        robot_state_layout.setSpacing(5)
        robot_state_title = QLabel("机器人状态")
        set_widget_role(robot_state_title, "subsection-title")
        robot_state_layout.addWidget(robot_state_title)

        self.text_robot_state = QTextEdit()
        self.text_robot_state.setObjectName("runtimeLog")
        self.text_robot_state.setReadOnly(True)
        self.text_robot_state.setAcceptRichText(False)
        self.text_robot_state.setPlainText("等待机器人状态数据...")
        self.text_robot_state.setMinimumWidth(0)
        set_widget_role(self.text_robot_state, "runtime-log")
        robot_state_layout.addWidget(self.text_robot_state, stretch=1)
        content_layout.addWidget(robot_state_widget, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

    @staticmethod
    def _make_status_label_compressible(label: QLabel) -> None:
        label.setTextFormat(Qt.PlainText)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    @staticmethod
    def _set_status_message(label: QLabel, message: str, role: str) -> None:
        label.setText(message)
        label.setToolTip(message)
        set_widget_role(label, role)

    def _set_preview_status_message(self, message: str, role: str) -> None:
        self.lbl_preview_record_status.setText(message)
        tooltip = message
        if self._preview_record_output_dir and self._preview_record_output_dir not in message:
            tooltip = f"{message}\n输出目录: {self._preview_record_output_dir}"
        self.lbl_preview_record_status.setToolTip(tooltip)
        set_widget_role(self.lbl_preview_record_status, role)

    @classmethod
    def _is_failure_status(cls, text: str) -> bool:
        normalized = str(text).casefold()
        return any(marker in normalized for marker in cls._FAILURE_MARKERS)

    def on_crop_toggled(self, checked):
        self.show_cropped_only = checked

    def _emit_preview_record_toggle(self, checked: bool) -> None:
        self.preview_record_toggle_requested.emit(
            bool(checked),
            self.selected_preview_record_target(),
        )

    def selected_preview_record_target(self) -> str:
        value = self.preview_record_target_combo.currentData()
        normalized = str(value).strip().lower() if value is not None else "both"
        return normalized if normalized in {"global", "wrist", "both"} else "both"

    def selected_preview_record_frame_mode(self) -> str:
        value = self.preview_record_frame_mode_combo.currentData()
        normalized = str(value).strip().lower() if value is not None else "source"
        return normalized if normalized in {"source", "square"} else "source"

    def process_image(self, cv_img):
        if cv_img is None or len(cv_img.shape) < 2:
            return cv_img

        height, width = cv_img.shape[:2]
        side = min(height, width)
        x0 = (width - side) // 2
        y0 = (height - side) // 2

        if self.show_cropped_only:
            return cv_img[y0:y0 + side, x0:x0 + side].copy()

        overlay = cv_img.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        masked = cv2.addWeighted(overlay, 0.5, cv_img, 0.5, 0)
        masked[y0:y0 + side, x0:x0 + side] = cv_img[y0:y0 + side, x0:x0 + side]
        cv2.rectangle(masked, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
        return masked

    def cv2_to_qpixmap(self, cv_img):
        try:
            processed = self.process_image(cv_img)
            if processed is None or processed.ndim != 3 or processed.shape[2] != 3:
                return QPixmap()
            rgb_img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_img.shape
            bytes_per_line = channels * width
            qimg = QImage(
                rgb_img.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            )
            return QPixmap.fromImage(qimg.copy())
        except Exception:
            return QPixmap()

    @Slot(object)
    def update_global_image(self, cv_img):
        pixmap = self.cv2_to_qpixmap(cv_img)
        if not pixmap.isNull():
            self.global_label.set_frame_pixmap(pixmap)
            return
        self.global_label.clear_frame("画面不可用")

    @Slot(object)
    def update_wrist_image(self, cv_img):
        pixmap = self.cv2_to_qpixmap(cv_img)
        if not pixmap.isNull():
            self.wrist_label.set_frame_pixmap(pixmap)
            return
        self.wrist_label.clear_frame("画面不可用")

    @Slot(str)
    def update_robot_state_str(self, text):
        self.text_robot_state.setPlainText(str(text))

    def set_preview_source(self, source_text: str) -> None:
        text = str(source_text).strip() or "未知"
        normalized = text.casefold()
        if any(marker in normalized for marker in ("无活动", "关闭", "不可用", "disabled", "none")):
            role = "status-neutral"
        elif any(marker in normalized for marker in ("等待", "连接中", "初始化", "pending")):
            role = "status-warning"
        else:
            role = "status-info"
        self._set_status_message(self.lbl_preview_source, f"预览源: {text}", role)

    def clear_images(self) -> None:
        self.global_label.clear_frame()
        self.wrist_label.clear_frame()

    @Slot(int, str, float)
    def update_dataset_record_stats(self, frames, time_str, realtime_fps):
        frames_str = "N/A" if frames is None or int(frames) < 0 else str(int(frames))
        fps_text = f"{float(realtime_fps):.2f} Hz" if realtime_fps is not None else "N/A"
        message = (
            f"采集状态: 录制中 | 时长: {time_str} | 已录制帧数: {frames_str} | "
            f"实时录制帧率: {fps_text}"
        )
        self._set_status_message(
            self.lbl_dataset_record_status,
            message,
            "status-info",
        )

    def reset_dataset_record_stats(self):
        self._set_status_message(
            self.lbl_dataset_record_status,
            "采集状态: 未录制 | 时长: 00:00 | 已录制帧数: 0 | 实时录制帧率: 0.00 Hz",
            "status-neutral",
        )

    @Slot(str)
    def update_preview_recording_status(self, text: str) -> None:
        message = str(text).strip() or "预览录屏: 未录制"
        normalized = message.casefold()
        if self._is_failure_status(message):
            role = "status-danger"
        elif "录制中" in normalized or "recording" in normalized:
            role = "status-info"
        elif any(marker in normalized for marker in ("已停止", "已保存", "完成", "stopped", "saved")):
            role = "status-success"
        elif any(marker in normalized for marker in ("停止中", "等待", "stopping", "pending")):
            role = "status-warning"
        else:
            role = "status-neutral"
        self._set_preview_status_message(message, role)

    def set_preview_recording_state(self, active: bool, *, output_dir: str = "") -> None:
        active = bool(active)
        self.btn_preview_record.blockSignals(True)
        self.btn_preview_record.setChecked(active)
        self.btn_preview_record.blockSignals(False)
        self.btn_preview_record.setText("停止预览录屏" if active else "开始预览录屏")
        self.preview_record_target_combo.setEnabled(not active)
        self.preview_record_frame_mode_combo.setEnabled(not active)

        self._preview_record_output_dir = str(output_dir).strip()
        self.btn_preview_record.setToolTip(self._preview_record_output_dir)
        set_widget_role(self.btn_preview_record, "secondary" if active else "primary")
        set_standard_icon(
            self.btn_preview_record,
            QStyle.StandardPixmap.SP_MediaStop if active else QStyle.StandardPixmap.SP_MediaPlay,
        )

        current_message = self.lbl_preview_record_status.text()
        if active:
            if "录制中" not in current_message and "recording" not in current_message.casefold():
                self._set_preview_status_message(
                    "预览录屏: 录制中 | 等待画面...",
                    "status-info",
                )
            else:
                self._set_preview_status_message(current_message, "status-info")
        elif self._is_failure_status(current_message):
            self._set_preview_status_message(current_message, "status-danger")
        elif any(marker in current_message.casefold() for marker in ("已停止", "stopped")):
            self._set_preview_status_message(current_message, "status-success")
        else:
            self._set_preview_status_message("预览录屏: 未录制", "status-neutral")

    @Slot(int, str, float)
    def update_record_stats(self, frames, time_str, realtime_fps):
        self.update_dataset_record_stats(frames, time_str, realtime_fps)

    def reset_record_stats(self):
        self.reset_dataset_record_stats()
