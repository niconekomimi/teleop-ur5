"""Domain panels used by the teleoperation main window."""

from .data_recording import DataRecordingPanel
from .inference import InferencePanel
from .status_overview import StatusOverviewPanel
from .system_control import SystemControlPanel
from .workspace import WorkspaceShell

__all__ = [
    "DataRecordingPanel",
    "InferencePanel",
    "StatusOverviewPanel",
    "SystemControlPanel",
    "WorkspaceShell",
]
