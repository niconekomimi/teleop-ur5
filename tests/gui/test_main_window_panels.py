from __future__ import annotations

from types import SimpleNamespace

from teleop_control_py.gui.panels import StatusOverviewPanel, SystemControlPanel


def _settings():
    return SimpleNamespace(
        default_input_type="quest3",
        joy_profiles=["auto", "xbox"],
        default_joy_profile="xbox",
        ur_type="ur10e",
        mediapipe_camera_options=["d435", "d455", "oakd"],
        default_mediapipe_camera="d455",
        default_mediapipe_input_topic="/d455/camera/color/image_raw",
        default_robot_ip="192.168.1.211",
        default_gripper_type="qbsofthand",
        camera_driver_options=["realsense", "oakd"],
        default_camera_driver="oakd",
    )


def test_system_control_panel_builds_config_startup_and_home_groups(qtbot) -> None:
    panel = SystemControlPanel(
        _settings(),
        local_ip="192.168.1.10",
        section_style="QGroupBox { font-weight: 700; }",
    )
    qtbot.addWidget(panel)

    groups = [panel.layout().itemAt(index).widget() for index in range(panel.layout().count())]
    assert [group.title() for group in groups] == [
        "\u7cfb\u7edf\u914d\u7f6e",
        "\u542f\u52a8\u8282\u70b9",
        "\u56dehome\u64cd\u4f5c",
    ]
    assert panel.mode_combo.currentData() == "quest3"
    assert panel.joy_profile_combo.currentData() == "xbox"
    assert panel.ur_type_input.text() == "ur10e"
    assert panel.mediapipe_camera_combo.currentData() == "d455"
    assert panel.mediapipe_topic_combo.currentText() == "/d455/camera/color/image_raw"
    assert panel.mediapipe_topic_combo.isEditable()
    assert panel.mediapipe_topic_combo.isHidden()
    assert panel.ip_input.text() == "192.168.1.211"
    assert panel.local_ip_label.text() == "192.168.1.10"
    assert panel.ee_combo.currentData() == "qbsofthand"
    assert panel.settings_group.layout().indexOf(panel.input_hint_label) >= 0


def test_system_control_panel_preserves_command_button_states(qtbot) -> None:
    panel = SystemControlPanel(
        _settings(),
        local_ip="127.0.0.1",
        section_style="",
        button_height=34,
    )
    qtbot.addWidget(panel)

    assert panel.camera_driver_combo.currentData() == "oakd"
    assert panel.camera_driver_combo.isHidden()
    assert panel.btn_camera_driver.isHidden()
    assert not panel.btn_camera_driver.isEnabled()
    assert panel.btn_robot_driver.isCheckable()
    assert panel.btn_teleop.isCheckable()
    assert panel.btn_robot_driver.height() == 34
    assert panel.btn_teleop.height() == 34
    assert panel.btn_go_home.height() == 34
    assert panel.btn_go_home_zone.height() == 34
    assert panel.btn_set_home_current.height() == 34
    assert panel.startup_group.layout().indexOf(panel.camera_module_hint_label) >= 0
    assert panel.startup_group.layout().indexOf(panel.startup_hint_label) >= 0


def test_status_overview_panel_populates_supplied_label_maps(qtbot) -> None:
    module_labels = {}
    hardware_labels = {}
    panel = StatusOverviewPanel(
        section_style="QGroupBox { font-weight: 700; }",
        module_status_labels=module_labels,
        hardware_status_labels=hardware_labels,
    )
    qtbot.addWidget(panel)

    assert panel.title() == "\u72b6\u6001\u603b\u89c8"
    assert set(module_labels) == {
        "robot_driver",
        "teleop",
        "data_collector",
        "inference",
        "preview",
    }
    assert set(hardware_labels) == {
        "joystick",
        "camera_1",
        "camera_2",
        "camera_3",
        "robot",
        "gripper",
    }
    assert all(label.text() == "\u672a\u77e5" for label in module_labels.values())
    assert all(label.text() == "\u672a\u77e5" for label in hardware_labels.values())
