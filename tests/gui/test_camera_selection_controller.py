from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from PySide6.QtWidgets import QComboBox

from teleop_control_py.gui.controllers import (
    CameraSelectionController,
    CameraSelectionWidgets,
)


@dataclass(frozen=True)
class FakeCamera:
    source: str
    model: str
    serial_number: str


def _settings(**overrides):
    values = {
        "default_mediapipe_camera": "d435",
        "default_mediapipe_camera_serial_number": "435-b",
        "default_mediapipe_input_topic": "/configured/color",
        "realsense_d435_serial_number": "",
        "realsense_d455_serial_number": "",
        "default_global_camera_source": "realsense",
        "default_collector_global_camera_model": "d455",
        "default_collector_global_camera_serial_number": "455-a",
        "default_wrist_camera_source": "oakd",
        "default_collector_wrist_camera_model": "oakd",
        "default_collector_wrist_camera_serial_number": "oak-1",
        "default_inference_global_camera_source": "realsense",
        "default_inference_global_camera_model": "d435",
        "default_inference_global_camera_serial_number": "435-b",
        "default_inference_wrist_camera_source": "oakd",
        "default_inference_wrist_camera_model": "oakd",
        "default_inference_wrist_camera_serial_number": "oak-1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _camera_inventory() -> list[FakeCamera]:
    return [
        FakeCamera("oakd", "oakd", "oak-1"),
        FakeCamera("realsense", "d435", "435-b"),
        FakeCamera("realsense", "d455", "455-a"),
        FakeCamera("realsense", "d435", "435-a"),
        FakeCamera("realsense", "d435", "435-a"),
    ]


def _widgets(qtbot) -> CameraSelectionWidgets:
    combos = [QComboBox() for _ in range(6)]
    for combo in combos:
        qtbot.addWidget(combo)
    combos[1].setEditable(True)
    return CameraSelectionWidgets(
        mediapipe_camera=combos[0],
        mediapipe_topic=combos[1],
        collector_global=combos[2],
        collector_wrist=combos[3],
        inference_global=combos[4],
        inference_wrist=combos[5],
    )


def _controller(settings, cameras) -> CameraSelectionController:
    return CameraSelectionController(
        settings_provider=lambda: settings,
        discover_cameras=lambda: list(cameras),
    )


def _select_serial(combo: QComboBox, serial: str) -> None:
    for index in range(combo.count()):
        option = combo.itemData(index)
        if isinstance(option, dict) and option.get("serial") == serial:
            combo.setCurrentIndex(index)
            return
    raise AssertionError(f"Camera serial not found: {serial}")


def test_discovery_deduplicates_sorts_and_labels_matching_models() -> None:
    controller = _controller(_settings(), _camera_inventory())

    options = controller.camera_option_candidates()

    assert [
        (item["source"], item["model"], item["serial"], item["label"])
        for item in options
    ] == [
        ("realsense", "d455", "455-a", "D455"),
        ("realsense", "d435", "435-a", "D435 \u76f8\u673a1"),
        ("realsense", "d435", "435-b", "D435 \u76f8\u673a2"),
        ("oakd", "oakd", "oak-1", "OAK-D"),
    ]


def test_refresh_applies_preferences_and_preserves_inference_selection(qtbot) -> None:
    widgets = _widgets(qtbot)
    controller = _controller(_settings(), _camera_inventory())

    options = controller.refresh(widgets)

    assert controller.options_loaded is True
    assert all(combo.count() == 4 for combo in (
        widgets.mediapipe_camera,
        widgets.collector_global,
        widgets.collector_wrist,
        widgets.inference_global,
        widgets.inference_wrist,
    ))
    assert widgets.mediapipe_camera.currentData()["serial"] == "435-b"
    assert widgets.collector_global.currentData()["serial"] == "455-a"
    assert widgets.collector_wrist.currentData()["serial"] == "oak-1"
    assert widgets.inference_global.currentData()["serial"] == "435-b"
    assert widgets.inference_wrist.currentData()["serial"] == "oak-1"
    assert widgets.mediapipe_topic.currentText() == "/d435/camera/color/image_raw"
    assert controller.count_by_source("realsense") == 3

    options[0]["source"] = "changed"
    assert controller.options[0]["source"] == "realsense"

    _select_serial(widgets.inference_global, "435-a")
    controller.refresh(widgets)
    assert widgets.inference_global.currentData()["serial"] == "435-a"


def test_refresh_without_cameras_drops_stale_mediapipe_serial(qtbot) -> None:
    settings = _settings(default_mediapipe_camera_serial_number="stale-serial")
    widgets = _widgets(qtbot)
    controller = _controller(settings, [])

    assert controller.selected_mediapipe_serial(widgets) == "stale-serial"

    controller.refresh(widgets)

    assert controller.options == []
    assert controller.options_loaded is True
    assert widgets.mediapipe_camera.count() == 0
    assert controller.selected_mediapipe_serial(widgets) == ""
    assert controller.selected_mediapipe_profile(widgets)["serial"] == ""
    assert controller.preference_updates(widgets)["default_mediapipe_camera_serial_number"] == ""


def test_profiles_sources_serials_and_preferences_follow_widgets(qtbot) -> None:
    widgets = _widgets(qtbot)
    controller = _controller(_settings(), _camera_inventory())
    controller.refresh(widgets)

    _select_serial(widgets.mediapipe_camera, "oak-1")
    _select_serial(widgets.collector_global, "435-a")
    _select_serial(widgets.collector_wrist, "435-b")
    _select_serial(widgets.inference_global, "455-a")
    _select_serial(widgets.inference_wrist, "oak-1")
    controller.sync_mediapipe_topic(widgets)

    profile = controller.selected_mediapipe_profile(widgets)
    assert profile["driver"] == "oakd"
    assert profile["serial"] == "oak-1"
    assert profile["depth_topic"] == "/oakd/stereo/depth"
    assert widgets.mediapipe_topic.currentText() == "/oakd/rgb/image_raw"
    assert controller.selected_collector_sources(widgets) == ("realsense", "realsense")
    assert controller.selected_collector_serials(widgets) == ("435-a", "435-b")
    assert controller.selected_inference_sources(widgets) == ("realsense", "oakd")
    assert controller.selected_inference_serials(widgets) == ("455-a", "oak-1")

    widgets.mediapipe_topic.setCurrentText("/custom/gesture/image")
    assert controller.preference_updates(widgets) == {
        "default_mediapipe_camera": "oakd",
        "default_mediapipe_camera_serial_number": "oak-1",
        "default_mediapipe_input_topic": "/custom/gesture/image",
        "default_global_camera_source": "realsense",
        "default_wrist_camera_source": "realsense",
        "default_collector_global_camera_model": "d435",
        "default_collector_wrist_camera_model": "d435",
        "default_collector_global_camera_serial_number": "435-a",
        "default_collector_wrist_camera_serial_number": "435-b",
        "default_inference_global_camera_source": "realsense",
        "default_inference_global_camera_model": "d455",
        "default_inference_global_camera_serial_number": "455-a",
        "default_inference_wrist_camera_source": "oakd",
        "default_inference_wrist_camera_model": "oakd",
        "default_inference_wrist_camera_serial_number": "oak-1",
    }


def test_runtime_status_identifies_each_camera_consumer(qtbot) -> None:
    widgets = _widgets(qtbot)
    controller = _controller(_settings(), _camera_inventory())
    controller.refresh(widgets)
    _select_serial(widgets.inference_global, "435-a")

    common = {
        "widgets": widgets,
        "teleop_running": False,
        "collector_running": False,
        "inference_running": False,
        "input_type": "joy",
    }
    assert controller.slot_runtime_status(
        slot_index=1,
        **{**common, "collector_running": True},
    ) == ("D455 \u91c7\u96c6\u5360\u7528", "#e67700")
    assert controller.slot_runtime_status(
        slot_index=2,
        **{**common, "inference_running": True},
    ) == ("D435 \u76f8\u673a1 \u63a8\u7406\u5360\u7528", "#e67700")
    assert controller.slot_runtime_status(
        slot_index=3,
        **{**common, "teleop_running": True, "input_type": "mediapipe"},
    ) == ("D435 \u76f8\u673a2 \u624b\u52bf\u5360\u7528", "#e67700")
    assert controller.slot_runtime_status(slot_index=4, **common) == (
        "OAK-D \u53ef\u7528",
        "#2b8a3e",
    )
    assert controller.slot_runtime_status(slot_index=5, **common) == (
        "\u672a\u68c0\u6d4b\u5230",
        "#6c757d",
    )


def test_legacy_combo_values_and_serial_matching_remain_supported(qtbot) -> None:
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("RealSense", "realsense")

    assert CameraSelectionController.selected_option(combo, "oakd", "oakd") == {
        "source": "realsense",
        "model": "d435",
        "serial": "",
    }

    combo.clear()
    combo.addItem("OAK-D", "oakd")
    assert CameraSelectionController.selected_option(combo, "realsense", "d455") == {
        "source": "oakd",
        "model": "oakd",
        "serial": "",
    }

    option = {"source": "realsense", "model": "d435", "serial": "435-a"}
    assert CameraSelectionController.option_matches_device(
        option,
        source="realsense",
        model="d435",
        serial="435-a",
    )
    assert not CameraSelectionController.option_matches_device(
        option,
        source="realsense",
        model="d435",
        serial="435-b",
    )
    assert CameraSelectionController.option_matches_device(
        {**option, "serial": ""},
        source="realsense",
        model="d435",
        serial="435-b",
    )
