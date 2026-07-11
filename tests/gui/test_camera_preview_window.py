from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy

from teleop_control_py.gui.theme import configure_application
from teleop_control_py.gui.widgets.camera_preview_window import CameraPreviewWindow
from teleop_control_py.gui.widgets.image_viewport import ImageViewport


def _window(qtbot, qapp) -> CameraPreviewWindow:
    configure_application(qapp)
    window = CameraPreviewWindow()
    qtbot.addWidget(window)
    return window


def test_large_frames_do_not_change_layout_hints_or_prevent_shrinking(qtbot, qapp) -> None:
    window = _window(qtbot, qapp)
    assert isinstance(window.global_label, ImageViewport)
    assert isinstance(window.wrist_label, ImageViewport)

    dialog_hint_before = window.minimumSizeHint()
    viewport_hint_before = window.global_label.minimumSizeHint()
    large_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    window.resize(1500, 900)
    window.show()
    qapp.processEvents()
    window.update_global_image(large_frame)
    window.update_wrist_image(large_frame)
    qapp.processEvents()

    assert window.global_label.minimumSizeHint() == viewport_hint_before == QSize(160, 120)
    assert window.minimumSizeHint() == dialog_hint_before
    assert window.global_label.source_pixmap().size() == QSize(2160, 2160)

    window.resize(900, 600)
    qapp.processEvents()

    assert window.size() == QSize(900, 600)
    assert window.global_label.width() > 0
    assert window.wrist_label.width() > 0


def test_long_recording_paths_are_tooltips_and_do_not_expand_window(qtbot, qapp) -> None:
    window = _window(qtbot, qapp)
    long_output_dir = "C:/recordings/" + ("very-long-session-name/" * 45)
    message = f"预览录屏: 录制中 | 目录: {long_output_dir}"

    window.set_preview_recording_state(True, output_dir=long_output_dir)
    window.update_preview_recording_status(message)
    window.resize(900, 600)
    window.show()
    qapp.processEvents()

    assert window.size() == QSize(900, 600)
    assert window.lbl_preview_record_status.text() == message
    assert message in window.lbl_preview_record_status.toolTip()
    assert long_output_dir in window.btn_preview_record.toolTip()
    assert (
        window.lbl_preview_record_status.sizePolicy().horizontalPolicy()
        == QSizePolicy.Policy.Ignored
    )
    assert window.lbl_dataset_record_status.width() < window.width()
    assert window.lbl_preview_record_status.width() < window.width()


def test_failed_recording_status_survives_inactive_state_update(qtbot, qapp) -> None:
    window = _window(qtbot, qapp)
    output_dir = "C:/recordings/failed-session"
    failure_message = "预览录屏: 失败 | 无法创建视频文件"

    window.update_preview_recording_status(failure_message)
    window.set_preview_recording_state(False, output_dir=output_dir)

    assert window.lbl_preview_record_status.text() == failure_message
    assert window.lbl_preview_record_status.property("role") == "status-danger"
    assert failure_message in window.lbl_preview_record_status.toolTip()
    assert output_dir in window.lbl_preview_record_status.toolTip()
    assert not window.btn_preview_record.isChecked()
    assert window.btn_preview_record.text() == "开始预览录屏"


def test_active_recording_and_robot_log_use_semantic_roles(qtbot, qapp) -> None:
    window = _window(qtbot, qapp)

    window.set_preview_recording_state(True)
    window.update_preview_recording_status("预览录屏: 录制中 | 等待画面...")

    assert window.lbl_preview_record_status.property("role") == "status-info"
    assert window.btn_preview_record.property("role") == "secondary"
    assert not window.btn_preview_record.icon().isNull()
    assert window.text_robot_state.property("role") == "runtime-log"
    assert window.text_robot_state.objectName() == "runtimeLog"


def test_recording_status_elides_without_losing_full_text(qtbot, qapp) -> None:
    window = _window(qtbot, qapp)
    long_time = "01:24 " + ("持续录制 " * 30)
    window.update_dataset_record_stats(2520, long_time, 30.0)
    full_text = window.lbl_dataset_record_status.text()

    window.resize(900, 600)
    window.show()
    qapp.processEvents()

    visible_text = window.lbl_dataset_record_status.displayed_text()
    assert visible_text.endswith("…")
    assert visible_text != full_text
    assert window.lbl_dataset_record_status.text() == full_text
    assert window.lbl_dataset_record_status.toolTip() == full_text
    assert (
        window.lbl_dataset_record_status.fontMetrics().horizontalAdvance(visible_text)
        <= window.lbl_dataset_record_status.contentsRect().width()
    )


def test_invalid_frame_clears_stale_preview(qtbot, qapp) -> None:
    window = _window(qtbot, qapp)
    valid_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    window.update_global_image(valid_frame)
    window.update_wrist_image(valid_frame)

    assert not window.global_label.source_pixmap().isNull()
    assert not window.wrist_label.source_pixmap().isNull()

    window.update_global_image(None)
    window.update_wrist_image(np.zeros((120,), dtype=np.uint8))

    assert window.global_label.source_pixmap().isNull()
    assert window.wrist_label.source_pixmap().isNull()
    assert window.global_label.text() == "画面不可用"
    assert window.wrist_label.text() == "画面不可用"
