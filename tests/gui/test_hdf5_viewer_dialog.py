from __future__ import annotations

import os

import h5py
import numpy as np
from PySide6.QtWidgets import QMessageBox, QSizePolicy

from teleop_control_py.gui.theme import configure_application
from teleop_control_py.gui.widgets.hdf5_viewer_dialog import HDF5ViewerDialog
from teleop_control_py.gui.widgets.image_viewport import ImageViewport


def _write_demo(data_group, name: str, *, frames: int = 2, valid: bool = True) -> None:
    demo = data_group.create_group(name)
    demo.attrs["num_samples"] = frames
    demo.create_dataset("actions", data=np.zeros((frames, 7), dtype=np.float32))

    obs = demo.create_group("obs")
    agent = np.zeros((frames, 12, 16, 3), dtype=np.uint8)
    agent[..., 0] = 180
    obs.create_dataset("agentview_rgb", data=agent)
    if valid:
        wrist = np.zeros((frames, 12, 16, 3), dtype=np.uint8)
        wrist[..., 1] = 140
        obs.create_dataset("eye_in_hand_rgb", data=wrist)

    obs.create_dataset("joint_states", data=np.zeros((frames, 6), dtype=np.float32))
    obs.create_dataset("ee_pos", data=np.zeros((frames, 3), dtype=np.float32))
    obs.create_dataset("ee_ori", data=np.zeros((frames, 3), dtype=np.float32))
    obs.create_dataset("gripper_states", data=np.zeros((frames, 1), dtype=np.float32))


def _dataset(path, *, include_broken_demo: bool = False) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        _write_demo(data, "demo_0")
        if include_broken_demo:
            _write_demo(data, "demo_1", valid=False)


def _dialog(qtbot, qapp, path) -> HDF5ViewerDialog:
    configure_application(qapp)
    dialog = HDF5ViewerDialog(str(path))
    qtbot.addWidget(dialog)
    return dialog


def test_empty_state_disables_commands_and_bounds_width(qtbot, qapp, tmp_path) -> None:
    missing_path = tmp_path / ("very-long-dataset-name-" * 8 + ".hdf5")
    dialog = _dialog(qtbot, qapp, missing_path)
    dialog.show()
    qapp.processEvents()

    assert dialog.file_handle is None
    assert isinstance(dialog.lbl_agent, ImageViewport)
    assert isinstance(dialog.lbl_wrist, ImageViewport)
    assert dialog.minimumSizeHint().width() <= 1100
    assert dialog.lbl_file_path.minimumWidth() == 120
    assert (
        dialog.lbl_file_path.sizePolicy().horizontalPolicy()
        == QSizePolicy.Policy.Expanding
    )
    assert dialog.lbl_file_path.toolTip() == os.path.abspath(str(missing_path))
    assert "不存在" in dialog.text_state.toPlainText()

    unavailable_commands = (
        dialog.demo_combo,
        dialog.btn_prev_demo,
        dialog.btn_next_demo,
        dialog.btn_delete_current,
        dialog.btn_save_changes,
        dialog.btn_prev,
        dialog.btn_play,
        dialog.btn_next,
        dialog.speed_combo,
        dialog.slider,
        dialog.crop_start_spin,
        dialog.crop_end_spin,
        dialog.btn_apply_crop,
        dialog.btn_clear_crop,
        dialog.btn_auto_trim_zero_actions,
    )
    assert all(not widget.isEnabled() for widget in unavailable_commands)


def test_valid_dataset_loads_images_and_uses_semantic_commands(qtbot, qapp, tmp_path) -> None:
    path = tmp_path / "valid.hdf5"
    _dataset(path)
    dialog = _dialog(qtbot, qapp, path)
    dialog.show()
    qapp.processEvents()

    assert dialog.file_handle is not None
    assert dialog.current_demo_name == "demo_0"
    assert dialog.demo_combo.count() == 1
    assert not dialog.lbl_agent.source_pixmap().isNull()
    assert not dialog.lbl_wrist.source_pixmap().isNull()
    assert "录制帧数据" in dialog.text_state.toPlainText()
    assert "►" not in dialog.text_state.toPlainText()
    assert dialog.btn_play.isEnabled()
    assert dialog.btn_apply_crop.isEnabled()
    assert not dialog.btn_save_changes.isEnabled()

    expected_roles = (
        (dialog.btn_open_file, "secondary"),
        (dialog.btn_rebuild_schema, "secondary"),
        (dialog.btn_delete_current, "danger-secondary"),
        (dialog.btn_save_changes, "primary"),
        (dialog.btn_play, "primary"),
        (dialog.btn_apply_crop, "secondary"),
        (dialog.btn_auto_trim_zero_actions, "warning"),
    )
    for button, role in expected_roles:
        assert button.property("role") == role
        assert not button.icon().isNull()

    initial_icon_key = dialog.btn_play.icon().cacheKey()
    dialog.start_playback()
    assert dialog.btn_play.text() == "暂停"
    pause_icon_key = dialog.btn_play.icon().cacheKey()
    assert pause_icon_key != initial_icon_key
    dialog.pause_playback()
    assert dialog.btn_play.text() == "播放"
    assert dialog.btn_play.icon().cacheKey() != pause_icon_key


def test_failed_frame_read_clears_previous_images(qtbot, qapp, tmp_path) -> None:
    path = tmp_path / "broken-frame.hdf5"
    _dataset(path, include_broken_demo=True)
    dialog = _dialog(qtbot, qapp, path)

    assert not dialog.lbl_agent.source_pixmap().isNull()
    assert not dialog.lbl_wrist.source_pixmap().isNull()

    dialog.demo_combo.setCurrentIndex(1)
    qapp.processEvents()

    assert dialog.current_demo_name == "demo_1"
    assert dialog._agent_source_pixmap.isNull()
    assert dialog._wrist_source_pixmap.isNull()
    assert dialog.lbl_agent.source_pixmap().isNull()
    assert dialog.lbl_wrist.source_pixmap().isNull()
    assert dialog.lbl_agent.text() == "帧读取失败"
    assert dialog.lbl_wrist.text() == "帧读取失败"
    assert "读取帧数据失败" in dialog.text_state.toPlainText()


def test_open_failure_closes_handle_and_restores_empty_state(
    qtbot,
    qapp,
    tmp_path,
    monkeypatch,
) -> None:
    invalid_path = tmp_path / "invalid.hdf5"
    with h5py.File(invalid_path, "w") as handle:
        handle.create_group("wrong-root")

    errors = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )
    dialog = _dialog(qtbot, qapp, "")

    assert dialog.open_hdf5_file(invalid_path) is False
    assert dialog.file_handle is None
    assert dialog.current_demo_group is None
    assert dialog.current_demo_name is None
    assert dialog.demo_combo.count() == 0
    assert dialog.lbl_agent.source_pixmap().isNull()
    assert dialog.lbl_wrist.source_pixmap().isNull()
    assert not dialog.btn_play.isEnabled()
    assert not dialog.btn_apply_crop.isEnabled()
    assert not dialog.btn_delete_current.isEnabled()
    assert not dialog.btn_save_changes.isEnabled()
    assert errors and errors[0][0] == "HDF5 读取错误"

    # Opening for write succeeds only if the read handle from the failed load was closed.
    with h5py.File(invalid_path, "a") as handle:
        handle.create_group("data")


def test_close_event_releases_loaded_file(qtbot, qapp, tmp_path) -> None:
    path = tmp_path / "close.hdf5"
    _dataset(path)
    dialog = _dialog(qtbot, qapp, path)
    dialog.show()
    qapp.processEvents()
    handle = dialog.file_handle

    assert handle is not None and handle.id.valid
    dialog.close()
    qapp.processEvents()

    assert dialog.file_handle is None
    assert not handle.id.valid
