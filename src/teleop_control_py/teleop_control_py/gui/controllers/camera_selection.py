"""Camera discovery, selection, and preference mapping for the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from PySide6.QtWidgets import QComboBox


@dataclass(frozen=True)
class CameraSelectionWidgets:
    mediapipe_camera: QComboBox
    mediapipe_topic: QComboBox
    collector_global: QComboBox
    collector_wrist: QComboBox
    inference_global: QComboBox
    inference_wrist: QComboBox


class CameraSelectionController:
    def __init__(
        self,
        settings_provider: Callable[[], Any],
        discover_cameras: Callable[[], Sequence[Any]],
    ) -> None:
        self._settings_provider = settings_provider
        self._discover_cameras = discover_cameras
        self._options: list[dict[str, str]] = []
        self.options_loaded = False

    @property
    def settings(self) -> Any:
        return self._settings_provider()

    @property
    def options(self) -> list[dict[str, str]]:
        return [dict(item) for item in self._options]

    def default_mediapipe_camera_model(self) -> str:
        model = str(self.settings.default_mediapipe_camera).strip().lower()
        if model in {"", "realsense", "rs", "camera"}:
            return "d435"
        return model

    def default_mediapipe_camera_source(self) -> str:
        return "oakd" if self.default_mediapipe_camera_model() == "oakd" else "realsense"

    @staticmethod
    def camera_model_label(source: str, model: str) -> str:
        normalized_source = str(source).strip().lower()
        normalized_model = str(model).strip().lower()
        if not normalized_model or normalized_model == "camera":
            normalized_model = "oakd" if normalized_source == "oakd" else "d435"
        if normalized_model == "oakd":
            return "OAK-D"
        return normalized_model.upper()

    @classmethod
    def camera_option_data(
        cls,
        source: str,
        model: str,
        serial: str,
        *,
        label: str = "",
    ) -> dict[str, str]:
        normalized_source = str(source).strip().lower()
        normalized_model = str(model).strip().lower() or normalized_source
        normalized_serial = str(serial).strip()
        normalized_label = str(label).strip() or cls.camera_model_label(normalized_source, normalized_model)
        return {
            "source": normalized_source,
            "model": normalized_model,
            "serial": normalized_serial,
            "label": normalized_label,
        }

    def camera_option_candidates(self) -> list[dict[str, str]]:
        options = [
            self.camera_option_data(camera.source, camera.model, camera.serial_number)
            for camera in self._discover_cameras()
        ]

        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for option in options:
            key = (option["source"], option["model"], option["serial"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(option)

        def priority(item: dict[str, str]) -> tuple[int, int, int, str]:
            source = item["source"]
            model = item["model"]
            serial = item["serial"]
            source_rank = 0 if source == "realsense" else (1 if source == "oakd" else 2)
            model_rank = {"d455": 0, "d435": 1, "oakd": 2}.get(model, 9)
            serial_rank = 1 if serial else 2
            return source_rank, model_rank, serial_rank, serial

        deduped.sort(key=priority)
        model_counts: dict[str, int] = {}
        for item in deduped:
            label_key = self.camera_model_label(item["source"], item["model"])
            model_counts[label_key] = model_counts.get(label_key, 0) + 1

        model_indices: dict[str, int] = {}
        relabeled: list[dict[str, str]] = []
        for item in deduped:
            base_label = self.camera_model_label(item["source"], item["model"])
            if model_counts.get(base_label, 0) > 1:
                current_index = model_indices.get(base_label, 0) + 1
                model_indices[base_label] = current_index
                label = f"{base_label} 相机{current_index}"
            else:
                label = base_label
            relabeled.append(
                self.camera_option_data(item["source"], item["model"], item["serial"], label=label)
            )
        return relabeled

    @staticmethod
    def selected_option(
        combo: QComboBox,
        fallback_source: str,
        fallback_model: str = "",
    ) -> dict[str, str]:
        value = combo.currentData()
        if isinstance(value, dict):
            source = str(value.get("source", "")).strip().lower() or str(fallback_source).strip().lower()
            model = str(value.get("model", "")).strip().lower() or str(fallback_model).strip().lower() or source
            serial = str(value.get("serial", "")).strip()
            return {"source": source, "model": model, "serial": serial}

        normalized_value = str(value).strip().lower() if value is not None else ""
        normalized_fallback_source = str(fallback_source).strip().lower()
        normalized_fallback_model = str(fallback_model).strip().lower()
        if normalized_value == "oakd":
            normalized_source = "oakd"
        elif normalized_value in {"realsense", "rs", "d435", "d455", "camera", "l515"}:
            normalized_source = "realsense"
        else:
            normalized_source = normalized_fallback_source or "realsense"

        normalized_model = normalized_value or normalized_fallback_model
        if normalized_model in {"realsense", "rs"}:
            normalized_model = "d435"
        if not normalized_model:
            normalized_model = "oakd" if normalized_source == "oakd" else "d435"
        return {"source": normalized_source, "model": normalized_model, "serial": ""}

    @staticmethod
    def set_combo_options(
        combo: QComboBox,
        options: list[dict[str, str]],
        *,
        preferred_source: str,
        preferred_model: str = "",
        preferred_serial: str = "",
    ) -> None:
        normalized_source = str(preferred_source).strip().lower()
        normalized_model = str(preferred_model).strip().lower()
        normalized_serial = str(preferred_serial).strip()

        combo.blockSignals(True)
        combo.clear()
        for option in options:
            combo.addItem(str(option.get("label", "")), dict(option))

        selected_index = -1
        for index, option in enumerate(options):
            source = option["source"]
            model = option["model"]
            serial = option["serial"]
            if (
                normalized_source
                and normalized_serial
                and source == normalized_source
                and serial == normalized_serial
            ):
                selected_index = index
                break
            if (
                selected_index < 0
                and normalized_source
                and normalized_model
                and source == normalized_source
                and model == normalized_model
            ):
                selected_index = index
            if selected_index < 0 and normalized_source and source == normalized_source:
                selected_index = index

        if selected_index < 0 and options:
            selected_index = 0
        if selected_index >= 0:
            combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def refresh(self, widgets: CameraSelectionWidgets) -> list[dict[str, str]]:
        settings = self.settings
        options = self.camera_option_candidates()
        inference_global_selected = widgets.inference_global.currentIndex() >= 0
        inference_wrist_selected = widgets.inference_wrist.currentIndex() >= 0
        inference_global = self.selected_option(
            widgets.inference_global,
            settings.default_inference_global_camera_source,
            settings.default_inference_global_camera_model,
        )
        inference_wrist = self.selected_option(
            widgets.inference_wrist,
            settings.default_inference_wrist_camera_source,
            settings.default_inference_wrist_camera_model,
        )

        self.set_combo_options(
            widgets.mediapipe_camera,
            options,
            preferred_source=self.default_mediapipe_camera_source(),
            preferred_model=self.default_mediapipe_camera_model(),
            preferred_serial=settings.default_mediapipe_camera_serial_number,
        )
        self.set_combo_options(
            widgets.collector_global,
            options,
            preferred_source=settings.default_global_camera_source,
            preferred_model=settings.default_collector_global_camera_model,
            preferred_serial=settings.default_collector_global_camera_serial_number,
        )
        self.set_combo_options(
            widgets.collector_wrist,
            options,
            preferred_source=settings.default_wrist_camera_source,
            preferred_model=settings.default_collector_wrist_camera_model,
            preferred_serial=settings.default_collector_wrist_camera_serial_number,
        )
        self.set_combo_options(
            widgets.inference_global,
            options,
            preferred_source=inference_global["source"] or settings.default_inference_global_camera_source,
            preferred_model=inference_global["model"] or settings.default_inference_global_camera_model,
            preferred_serial=(
                inference_global["serial"]
                if inference_global_selected
                else settings.default_inference_global_camera_serial_number
            ),
        )
        self.set_combo_options(
            widgets.inference_wrist,
            options,
            preferred_source=inference_wrist["source"] or settings.default_inference_wrist_camera_source,
            preferred_model=inference_wrist["model"] or settings.default_inference_wrist_camera_model,
            preferred_serial=(
                inference_wrist["serial"]
                if inference_wrist_selected
                else settings.default_inference_wrist_camera_serial_number
            ),
        )
        self._options = [dict(item) for item in options]
        self.options_loaded = True
        self.sync_mediapipe_topic(widgets)
        return self.options

    def selected_mediapipe_camera(self, widgets: CameraSelectionWidgets) -> str:
        option = self.selected_option(
            widgets.mediapipe_camera,
            self.default_mediapipe_camera_source(),
            self.default_mediapipe_camera_model(),
        )
        return option["model"] or self.default_mediapipe_camera_model()

    def selected_mediapipe_serial(self, widgets: CameraSelectionWidgets) -> str:
        option = self.selected_option(
            widgets.mediapipe_camera,
            self.default_mediapipe_camera_source(),
            self.default_mediapipe_camera_model(),
        )
        if option["serial"]:
            return option["serial"]
        if self.options_loaded:
            return ""
        return str(self.settings.default_mediapipe_camera_serial_number).strip()

    def mediapipe_camera_profiles(self) -> dict[str, dict[str, str]]:
        settings = self.settings
        return {
            "d435": {
                "name": "d435", "driver": "realsense", "namespace": "d435", "camera_name": "camera",
                "serial": str(settings.realsense_d435_serial_number).strip(),
                "input_topic": "/d435/camera/color/image_raw",
                "depth_topic": "/d435/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/d435/camera/aligned_depth_to_color/camera_info",
            },
            "d455": {
                "name": "d455", "driver": "realsense", "namespace": "d455", "camera_name": "camera",
                "serial": str(settings.realsense_d455_serial_number).strip(),
                "input_topic": "/d455/camera/color/image_raw",
                "depth_topic": "/d455/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/d455/camera/aligned_depth_to_color/camera_info",
            },
            "oakd": {
                "name": "oakd", "driver": "oakd", "namespace": "oakd", "camera_name": "camera", "serial": "",
                "input_topic": "/oakd/rgb/image_raw", "depth_topic": "/oakd/stereo/depth",
                "camera_info_topic": "/oakd/rgb/camera_info",
            },
            "camera": {
                "name": "camera", "driver": "realsense", "namespace": "camera", "camera_name": "camera", "serial": "",
                "input_topic": "/camera/camera/color/image_raw",
                "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            },
        }

    def selected_mediapipe_profile(self, widgets: CameraSelectionWidgets) -> dict[str, str]:
        profiles = self.mediapipe_camera_profiles()
        selected = self.selected_mediapipe_camera(widgets)
        default_key = self.default_mediapipe_camera_model()
        fallback = profiles.get(default_key) or profiles.get("d435") or next(iter(profiles.values()))
        profile = dict(profiles.get(selected) or fallback)
        option = self.selected_option(
            widgets.mediapipe_camera,
            self.default_mediapipe_camera_source(),
            self.default_mediapipe_camera_model(),
        )
        if option["source"]:
            profile["driver"] = option["source"]
        serial = self.selected_mediapipe_serial(widgets)
        if serial:
            profile["serial"] = serial
        return profile

    def sync_mediapipe_topic(self, widgets: CameraSelectionWidgets) -> None:
        topic = self.selected_mediapipe_profile(widgets).get("input_topic", "").strip()
        if not topic or widgets.mediapipe_topic.currentText().strip() == topic:
            return
        widgets.mediapipe_topic.blockSignals(True)
        widgets.mediapipe_topic.setCurrentText(topic)
        widgets.mediapipe_topic.blockSignals(False)

    def selected_mediapipe_topic(self, widgets: CameraSelectionWidgets) -> str:
        profile = self.selected_mediapipe_profile(widgets)
        return (
            widgets.mediapipe_topic.currentText().strip()
            or profile.get("input_topic", "").strip()
            or str(self.settings.default_mediapipe_input_topic)
        )

    def selected_mediapipe_depth_topic(self, widgets: CameraSelectionWidgets) -> str:
        return (
            self.selected_mediapipe_profile(widgets).get("depth_topic", "").strip()
            or "/camera/camera/aligned_depth_to_color/image_raw"
        )

    def selected_mediapipe_camera_info_topic(self, widgets: CameraSelectionWidgets) -> str:
        return (
            self.selected_mediapipe_profile(widgets).get("camera_info_topic", "").strip()
            or "/camera/camera/aligned_depth_to_color/camera_info"
        )

    def selected_source(self, combo: QComboBox, fallback: str) -> str:
        return self.selected_option(combo, fallback, fallback)["source"] or fallback

    def selected_collector_sources(self, widgets: CameraSelectionWidgets) -> tuple[str, str]:
        settings = self.settings
        return (
            self.selected_source(widgets.collector_global, settings.default_global_camera_source),
            self.selected_source(widgets.collector_wrist, settings.default_wrist_camera_source),
        )

    def selected_inference_sources(self, widgets: CameraSelectionWidgets) -> tuple[str, str]:
        settings = self.settings
        return (
            self.selected_source(widgets.inference_global, settings.default_inference_global_camera_source),
            self.selected_source(widgets.inference_wrist, settings.default_inference_wrist_camera_source),
        )

    def selected_collector_serials(self, widgets: CameraSelectionWidgets) -> tuple[str, str]:
        settings = self.settings
        global_option = self.selected_option(
            widgets.collector_global,
            settings.default_global_camera_source,
            settings.default_collector_global_camera_model,
        )
        wrist_option = self.selected_option(
            widgets.collector_wrist,
            settings.default_wrist_camera_source,
            settings.default_collector_wrist_camera_model,
        )
        return global_option["serial"], wrist_option["serial"]

    def selected_inference_serials(self, widgets: CameraSelectionWidgets) -> tuple[str, str]:
        settings = self.settings
        global_option = self.selected_option(
            widgets.inference_global,
            settings.default_inference_global_camera_source,
            settings.default_inference_global_camera_model,
        )
        wrist_option = self.selected_option(
            widgets.inference_wrist,
            settings.default_inference_wrist_camera_source,
            settings.default_inference_wrist_camera_model,
        )
        return global_option["serial"], wrist_option["serial"]

    def count_by_source(self, source: str) -> int:
        normalized = str(source).strip().lower()
        return sum(1 for option in self._options if option["source"] == normalized) if normalized else 0

    @staticmethod
    def option_matches_device(
        option: dict[str, str],
        *,
        source: str,
        model: str,
        serial: str,
    ) -> bool:
        option_source = str(option.get("source", "")).strip().lower()
        option_model = str(option.get("model", "")).strip().lower()
        option_serial = str(option.get("serial", "")).strip()
        normalized_source = str(source).strip().lower()
        normalized_model = str(model).strip().lower()
        normalized_serial = str(serial).strip()
        if option_source != normalized_source or option_model != normalized_model:
            return False
        if normalized_serial and option_serial:
            return option_serial == normalized_serial
        return True

    def slot_runtime_status(
        self,
        widgets: CameraSelectionWidgets,
        slot_index: int,
        *,
        teleop_running: bool,
        collector_running: bool,
        inference_running: bool,
        input_type: str,
    ) -> tuple[str, str]:
        if slot_index < 1 or len(self._options) < slot_index:
            return "未检测到", "#6c757d"
        settings = self.settings
        device = self._options[slot_index - 1]
        label = device["label"]
        source = device["source"]
        model = device["model"]
        serial = device["serial"]
        collector_options = (
            self.selected_option(
                widgets.collector_global,
                settings.default_global_camera_source,
                settings.default_collector_global_camera_model,
            ),
            self.selected_option(
                widgets.collector_wrist,
                settings.default_wrist_camera_source,
                settings.default_collector_wrist_camera_model,
            ),
        )
        inference_options = (
            self.selected_option(
                widgets.inference_global,
                settings.default_inference_global_camera_source,
                settings.default_inference_global_camera_model,
            ),
            self.selected_option(
                widgets.inference_wrist,
                settings.default_inference_wrist_camera_source,
                settings.default_inference_wrist_camera_model,
            ),
        )
        mediapipe_option = self.selected_option(
            widgets.mediapipe_camera,
            self.default_mediapipe_camera_source(),
            self.default_mediapipe_camera_model(),
        )

        def matches(option: dict[str, str]) -> bool:
            return self.option_matches_device(
                option,
                source=source,
                model=model,
                serial=serial,
            )

        if collector_running and any(matches(option) for option in collector_options):
            return f"{label} 采集占用", "#e67700"
        if inference_running and any(matches(option) for option in inference_options):
            return f"{label} 推理占用", "#e67700"
        if teleop_running and input_type == "mediapipe" and matches(mediapipe_option):
            return f"{label} 手势占用", "#e67700"
        return f"{label} 可用", "#2b8a3e"

    def preference_updates(self, widgets: CameraSelectionWidgets) -> dict[str, object]:
        settings = self.settings
        mediapipe = self.selected_option(
            widgets.mediapipe_camera,
            self.default_mediapipe_camera_source(),
            self.default_mediapipe_camera_model(),
        )
        collector_global = self.selected_option(
            widgets.collector_global,
            settings.default_global_camera_source,
            settings.default_collector_global_camera_model,
        )
        collector_wrist = self.selected_option(
            widgets.collector_wrist,
            settings.default_wrist_camera_source,
            settings.default_collector_wrist_camera_model,
        )
        inference_global = self.selected_option(
            widgets.inference_global,
            settings.default_inference_global_camera_source,
            settings.default_inference_global_camera_model,
        )
        inference_wrist = self.selected_option(
            widgets.inference_wrist,
            settings.default_inference_wrist_camera_source,
            settings.default_inference_wrist_camera_model,
        )
        return {
            "default_mediapipe_camera": (
                mediapipe["model"] or self.default_mediapipe_camera_model()
            ),
            "default_mediapipe_camera_serial_number": mediapipe["serial"],
            "default_mediapipe_input_topic": self.selected_mediapipe_topic(widgets),
            "default_global_camera_source": (
                collector_global["source"] or settings.default_global_camera_source
            ),
            "default_wrist_camera_source": (
                collector_wrist["source"] or settings.default_wrist_camera_source
            ),
            "default_collector_global_camera_model": (
                collector_global["model"] or settings.default_collector_global_camera_model
            ),
            "default_collector_wrist_camera_model": (
                collector_wrist["model"] or settings.default_collector_wrist_camera_model
            ),
            "default_collector_global_camera_serial_number": collector_global["serial"],
            "default_collector_wrist_camera_serial_number": collector_wrist["serial"],
            "default_inference_global_camera_source": (
                inference_global["source"]
                or settings.default_inference_global_camera_source
            ),
            "default_inference_global_camera_model": (
                inference_global["model"]
                or settings.default_inference_global_camera_model
            ),
            "default_inference_global_camera_serial_number": inference_global["serial"],
            "default_inference_wrist_camera_source": (
                inference_wrist["source"]
                or settings.default_inference_wrist_camera_source
            ),
            "default_inference_wrist_camera_model": (
                inference_wrist["model"]
                or settings.default_inference_wrist_camera_model
            ),
            "default_inference_wrist_camera_serial_number": inference_wrist["serial"],
        }
