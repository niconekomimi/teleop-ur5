from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QWidget,
)

from teleop_control_py.gui.dialogs import (
    TeleopBindingCaptureDialog,
    TeleopSettingsDialog,
)
from teleop_control_py.gui.theme import APP_STYLESHEET, configure_application


class FakeRuntimeFacade:
    def is_process_running(self, _key: str) -> bool:
        return False


class FakeParent(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.gui_settings = SimpleNamespace(teleop_settings={})
        self.runtime_facade = FakeRuntimeFacade()
        self.messages: list[str] = []

    def _selected_joy_profile(self) -> str:
        return "xbox"

    def log(self, message: str) -> None:
        self.messages.append(str(message))


class FakeTeleopSettingsStorage:
    def __init__(self) -> None:
        self.teleop_updates: dict[str, object] = {}
        self.moveit_updates: dict[str, object] = {}
        self.gui_updates: dict[str, object] = {}

    def load_teleop_params(self, _current_file: str):
        return {}, {}

    def save_teleop_params_overrides(self, _current_file: str, teleop_updates, moveit_updates):
        self.teleop_updates = dict(teleop_updates)
        self.moveit_updates = dict(moveit_updates)
        return Path("teleop_params.yaml")

    def save_gui_settings_overrides(self, _current_file: str, updates):
        self.gui_updates = dict(updates)
        return Path("gui_params.yaml")

    def load_gui_settings(self, _current_file: str):
        return SimpleNamespace(teleop_settings=self.gui_updates.get("teleop_settings", {}))


def _dialog(qtbot, *, initial_mode: str = "joy"):
    parent = FakeParent()
    storage = FakeTeleopSettingsStorage()
    dialog = TeleopSettingsDialog(parent, initial_mode=initial_mode, storage=storage)
    qtbot.addWidget(parent)
    qtbot.addWidget(dialog)
    return parent, storage, dialog


def test_dialog_builds_all_input_modes(qtbot) -> None:
    _parent, _storage, dialog = _dialog(qtbot, initial_mode="quest3")

    assert dialog.tabs.count() == 3
    assert dialog.tabs.currentIndex() == 2
    assert set(dialog._field_widgets) == {"joy", "mediapipe", "quest3"}
    assert all(dialog._field_widgets[mode] for mode in dialog.MODES)


def test_dialog_stays_bounded_and_each_mode_scrolls(qtbot) -> None:
    _parent, _storage, dialog = _dialog(qtbot)
    dialog.show()
    qtbot.wait(1)

    assert dialog.width() <= 780
    assert dialog.height() <= 700

    for index in range(dialog.tabs.count()):
        dialog.tabs.setCurrentIndex(index)
        qtbot.wait(1)
        page = dialog.tabs.widget(index)
        assert isinstance(page, QScrollArea)
        assert page.verticalScrollBar().maximum() > 0


def test_dialog_has_no_horizontal_overflow_at_500_by_400(qtbot, qapp) -> None:
    configure_application(qapp)
    _parent, _storage, dialog = _dialog(qtbot)
    dialog.resize(500, 400)
    dialog.show()
    qapp.processEvents()

    assert dialog.minimumWidth() == 480
    assert dialog.minimumHeight() == 360
    assert dialog.size().width() == 500
    assert dialog.size().height() == 400

    for index in range(dialog.tabs.count()):
        dialog.tabs.setCurrentIndex(index)
        qapp.processEvents()
        page = dialog.tabs.widget(index)
        assert isinstance(page, QScrollArea)
        assert page.horizontalScrollBar().maximum() == 0
        assert page.widget().minimumSizeHint().width() <= page.viewport().width()


def test_dialog_uses_semantic_chinese_commands_and_accessible_checkboxes(qtbot) -> None:
    _parent, _storage, dialog = _dialog(qtbot)

    assert dialog.close_button.text() == "关闭"
    assert dialog.close_button.property("role") == "secondary"
    assert dialog.apply_button.text() == "应用"
    assert dialog.apply_button.property("role") == "primary"
    assert not dialog.close_button.icon().isNull()
    assert not dialog.apply_button.icon().isNull()

    buttons = dialog.findChildren(QPushButton)
    save_buttons = [button for button in buttons if button.text() == "另存方案"]
    delete_buttons = [button for button in buttons if button.text() == "删除"]
    capture_buttons = [button for button in buttons if button.text() == "录制"]
    assert len(save_buttons) == len(dialog.MODES)
    assert len(delete_buttons) == len(dialog.MODES)
    assert capture_buttons
    assert all(button.property("role") == "secondary" for button in save_buttons)
    assert all(button.property("role") == "danger-secondary" for button in delete_buttons)
    assert all(button.property("role") == "secondary" for button in capture_buttons)
    assert all(not button.icon().isNull() for button in (*save_buttons, *delete_buttons, *capture_buttons))

    expected_names = {
        str(spec["key"]): str(spec["label"])
        for mode in dialog.MODES
        for spec in dialog._all_specs(mode)
        if spec.get("type") == "bool"
    }
    for mode in dialog.MODES:
        for key, widget in dialog._field_widgets[mode].items():
            if isinstance(widget, QCheckBox):
                assert widget.text() == ""
                assert widget.accessibleName() == expected_names[key]


def test_binding_capture_dialog_uses_chinese_cancel_command(qtbot) -> None:
    parent = FakeParent()
    dialog = TeleopBindingCaptureDialog(parent, capture_kind="button", joy_profile="xbox")
    qtbot.addWidget(parent)
    qtbot.addWidget(dialog)

    button_box = dialog.findChild(QDialogButtonBox)
    cancel_button = button_box.button(QDialogButtonBox.Cancel)
    assert cancel_button.text() == "取消"
    assert cancel_button.property("role") == "secondary"
    assert not cancel_button.icon().isNull()


def test_global_form_control_and_tab_styles_are_defined(qapp) -> None:
    configure_application(qapp)

    assert qapp.styleSheet() == APP_STYLESHEET
    required_selectors = (
        "QTabWidget::pane",
        "QTabBar::tab:selected",
        "QSlider::groove:horizontal",
        "QSlider::handle:horizontal",
        'QLabel[role="image-viewport"]',
        'QLabel[role="metadata"]',
    )
    assert all(selector in APP_STYLESHEET for selector in required_selectors)

    qapp.setStyleSheet("")
    try:
        assert type(qapp.style()).__name__ == "_OperatorStyle"
    finally:
        qapp.setStyleSheet(APP_STYLESHEET)


def test_apply_settings_uses_injected_storage(qtbot, monkeypatch) -> None:
    parent, storage, dialog = _dialog(qtbot)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: QMessageBox.Ok)
    dialog._field_widgets["joy"]["max_linear_vel"].setValue(1.25)

    dialog.apply_settings()

    assert storage.teleop_updates["max_linear_vel"] == 1.25
    assert "low_pass_filter_coeff" in storage.moveit_updates
    assert "teleop_settings" in storage.gui_updates
    assert parent.messages == ["已保存遥操作设置: teleop_params.yaml"]


def test_binding_dialog_accepts_numeric_key(qtbot) -> None:
    parent = FakeParent()
    dialog = TeleopBindingCaptureDialog(parent, capture_kind="button", joy_profile="xbox")
    qtbot.addWidget(parent)
    qtbot.addWidget(dialog)
    dialog.show()

    qtbot.keyClick(dialog, Qt.Key_3)

    assert dialog.result_value == 3
    assert dialog.result() == QDialog.Accepted
