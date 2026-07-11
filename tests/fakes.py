from __future__ import annotations

from dataclasses import dataclass, field

from teleop_control_py.core.models import ActionCommand


@dataclass
class RecordingArmBackend:
    commands: list[ActionCommand] = field(default_factory=list)
    zero_count: int = 0
    stopped: bool = False

    def send_delta_twist(self, command: ActionCommand) -> None:
        self.commands.append(command)

    def send_zero_twist(self) -> None:
        self.zero_count += 1

    def stop(self) -> None:
        self.stopped = True


@dataclass
class RecordingGripperBackend:
    values: list[float] = field(default_factory=list)
    stopped: bool = False

    def set_gripper(self, value: float) -> None:
        self.values.append(float(value))

    def stop(self) -> None:
        self.stopped = True

