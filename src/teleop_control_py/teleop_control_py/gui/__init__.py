"""GUI package for teleop_control_py."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "main",
    "GuiAppService",
    "RosWorkerCallbacks",
    "RosWorkerConfig",
    "RobotDriverLaunchConfig",
    "TeleopLaunchConfig",
    "CollectorLaunchConfig",
    "GuiIntentController",
    "IntentResult",
    "GuiRuntimeFacade",
    "RuntimeSnapshot",
]


def __getattr__(name: str) -> Any:
    if name == "main":
        module = import_module(".app", __name__)
        return getattr(module, name)
    if name in {
        "CollectorLaunchConfig",
        "GuiAppService",
        "RobotDriverLaunchConfig",
        "RosWorkerCallbacks",
        "RosWorkerConfig",
        "TeleopLaunchConfig",
    }:
        module = import_module(".app_service", __name__)
        return getattr(module, name)
    if name in {"GuiIntentController", "IntentResult"}:
        module = import_module(".intent_controller", __name__)
        return getattr(module, name)
    if name in {"GuiRuntimeFacade", "RuntimeSnapshot"}:
        module = import_module(".runtime_facade", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
