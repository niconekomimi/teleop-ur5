import numpy as np

from teleop_control_py.core.models import ActionCommand, ControlSource
from teleop_control_py.core.mux import ActionMux

from tests.fakes import RecordingArmBackend, RecordingGripperBackend


def _command(source: ControlSource, gripper: float = 0.6) -> ActionCommand:
    return ActionCommand(
        linear_xyz=np.array([0.1, -0.2, 0.3]),
        angular_xyz=np.array([0.4, -0.5, 0.6]),
        gripper=gripper,
        source=source,
    )


def test_active_source_command_is_forwarded_to_both_backends() -> None:
    arm = RecordingArmBackend()
    gripper = RecordingGripperBackend()
    mux = ActionMux(arm, gripper, active_source=ControlSource.TELEOP)

    result = mux.publish(_command(ControlSource.TELEOP))

    assert result.accepted
    assert arm.commands[0].source == ControlSource.TELEOP
    assert gripper.values == [0.6]


def test_lower_priority_source_is_rejected() -> None:
    arm = RecordingArmBackend()
    gripper = RecordingGripperBackend()
    mux = ActionMux(arm, gripper, active_source=ControlSource.INFERENCE)

    result = mux.publish(_command(ControlSource.TELEOP))

    assert not result.accepted
    assert result.reason == "lower_priority_source"
    assert arm.commands == []
    assert gripper.values == []


def test_hold_rejects_motion_but_allows_safety_command() -> None:
    arm = RecordingArmBackend()
    gripper = RecordingGripperBackend()
    mux = ActionMux(arm, gripper, active_source=ControlSource.INFERENCE)
    mux.set_hold(True, "estopped")

    rejected = mux.publish(_command(ControlSource.INFERENCE))
    accepted = mux.publish(ActionCommand.zero(source=ControlSource.SAFETY))

    assert not rejected.accepted
    assert rejected.reason == "estopped"
    assert accepted.accepted
    assert len(arm.commands) == 1
    assert arm.commands[0].source == ControlSource.SAFETY


def test_action_command_copies_and_freezes_vectors() -> None:
    linear = np.array([1.0, 2.0, 3.0])
    command = ActionCommand(linear, [4.0, 5.0, 6.0], gripper=2.0)
    linear[0] = 99.0

    assert command.linear_xyz.tolist() == [1.0, 2.0, 3.0]
    assert command.gripper == 1.0
    assert not command.linear_xyz.flags.writeable

