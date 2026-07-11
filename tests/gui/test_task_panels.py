from __future__ import annotations

from types import SimpleNamespace

from teleop_control_py.gui.panels.data_recording import DataRecordingPanel
from teleop_control_py.gui.panels.inference import InferencePanel


def _recording_settings() -> SimpleNamespace:
    return SimpleNamespace(
        default_hdf5_output_dir="D:/teleop/datasets",
        default_hdf5_filename="session_001.hdf5",
    )


def _inference_settings() -> SimpleNamespace:
    return SimpleNamespace(
        default_inference_backend="real_il",
        default_inference_device="cuda",
        default_inference_hz=10.0,
        default_openpi_host="10.0.0.42",
        default_openpi_port=19001,
        default_openpi_prompt="pick up the red block",
        collect_inference_action_logs=True,
    )


def test_data_recording_panel_preserves_paths_and_camera_placeholders(qtbot) -> None:
    panel = DataRecordingPanel(
        _recording_settings(),
        section_style="QGroupBox { font-weight: 700; }",
    )
    qtbot.addWidget(panel)

    assert panel.record_dir_input.text() == "D:/teleop/datasets"
    assert panel.record_dir_input.toolTip() == "D:/teleop/datasets"
    assert panel.record_name_input.text() == "session_001.hdf5"
    assert panel.record_name_input.toolTip() == "session_001.hdf5"
    assert panel.global_camera_source_combo.count() == 0
    assert panel.wrist_camera_source_combo.count() == 0
    assert panel.record_group.layout().indexOf(panel.camera_binding_hint_label) >= 0
    assert panel.camera_binding_hint_label.isHidden()


def test_data_recording_panel_preserves_action_and_log_states(qtbot) -> None:
    panel = DataRecordingPanel(
        _recording_settings(),
        section_style="",
        button_height=36,
    )
    qtbot.addWidget(panel)

    assert panel.btn_collector.isCheckable()
    assert not panel.btn_start_record.isCheckable()
    assert not panel.btn_stop_record.isCheckable()
    assert not panel.btn_discard_record.isCheckable()
    assert all(
        button.height() == 36
        for button in (
            panel.btn_collector,
            panel.btn_start_record,
            panel.btn_stop_record,
            panel.btn_discard_record,
        )
    )
    assert panel.log_output.isReadOnly()
    assert panel.preview_group.title() == "日志"
    assert panel.preview_group.findChildren(type(panel.btn_start_record)) == []
    assert panel.preview_group.layout().indexOf(panel.log_output) >= 0
    assert panel.log_output.minimumHeight() == 130
    assert panel.log_output.maximumHeight() == 130


def test_inference_panel_preserves_backend_and_openpi_defaults(qtbot) -> None:
    panel = InferencePanel(
        _inference_settings(),
        section_style="QGroupBox { font-weight: 700; }",
    )
    qtbot.addWidget(panel)

    assert [
        panel.inference_backend_combo.itemData(index)
        for index in range(panel.inference_backend_combo.count())
    ] == ["real_il", "openpi_remote"]
    assert panel.inference_backend_combo.currentData() == "real_il"
    assert panel.inference_openpi_host_input.text() == "10.0.0.42"
    assert panel.inference_openpi_prompt_input.text() == "pick up the red block"
    assert panel.inference_openpi_port_spin.minimum() == 1
    assert panel.inference_openpi_port_spin.maximum() == 65535
    assert panel.inference_openpi_port_spin.value() == 19001


def test_inference_panel_preserves_local_runtime_defaults(qtbot) -> None:
    panel = InferencePanel(
        _inference_settings(),
        section_style="",
        button_height=36,
        emphasis_spin_height=32,
    )
    qtbot.addWidget(panel)

    assert panel.inference_global_camera_combo.count() == 0
    assert panel.inference_wrist_camera_combo.count() == 0
    assert [
        panel.inference_device_combo.itemData(index)
        for index in range(panel.inference_device_combo.count())
    ] == ["auto", "cuda", "cpu"]
    assert panel.inference_device_combo.currentData() == "cuda"
    assert panel.inference_hz_spin.minimum() == 0.2
    assert panel.inference_hz_spin.maximum() == 50.0
    assert panel.inference_hz_spin.value() == 10.0
    assert panel.inference_hz_spin.height() == 32

    assert panel.btn_inference.isCheckable()
    assert panel.btn_inference.height() == 36
    assert panel.btn_execute_inference.isCheckable()
    assert not panel.btn_execute_inference.isEnabled()
    assert panel.btn_execute_inference.height() == 36
    assert not panel.btn_inference_estop.isEnabled()
    assert panel.btn_inference_estop.height() == 36
    assert panel.lbl_inference_status.text() == "\u672a\u542f\u52a8"
    assert panel.lbl_inference_execute_status.text() == "\u672a\u4f7f\u80fd"
    assert panel.chk_collect_inference_logs.isChecked()
    assert panel.inference_action_output.isReadOnly()
    assert panel.inference_action_output.minimumHeight() == 75
    assert panel.inference_action_output.maximumHeight() == 75


def test_inference_panel_exposes_backend_widget_contract(qtbot) -> None:
    panel = InferencePanel(
        _inference_settings(),
        section_style="",
    )
    qtbot.addWidget(panel)

    assert panel._real_il_widgets == [
        panel.lbl_inference_model_dir,
        panel.inference_model_dir_input,
        panel.btn_browse_inference_model,
        panel.btn_refresh_inference_options,
        panel.lbl_inference_env,
        panel.inference_env_combo,
        panel.lbl_inference_task,
        panel.inference_task_combo,
        panel.lbl_inference_embedding,
        panel.inference_embedding_input,
        panel.btn_browse_inference_embedding,
        panel.btn_auto_match_embedding,
        panel.lbl_inference_device,
        panel.inference_device_combo,
    ]
    assert panel._openpi_widgets == [
        panel.lbl_openpi_host,
        panel.inference_openpi_host_input,
        panel.lbl_openpi_port,
        panel.inference_openpi_port_spin,
        panel.lbl_openpi_prompt,
        panel.inference_openpi_prompt_input,
    ]
    assert panel.layout().indexOf(panel.inference_hint_label) >= 0
    assert panel.inference_hint_label.isHidden()
