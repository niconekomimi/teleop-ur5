from teleop_control_py.core.control_coordinator import ControlCoordinator
from teleop_control_py.core.models import ActionCommand, ControlSource
from teleop_control_py.core.orchestrator import SystemPhase

from tests.fakes import RecordingArmBackend, RecordingGripperBackend


def test_coordinator_routes_active_teleop_command() -> None:
    arm = RecordingArmBackend()
    gripper = RecordingGripperBackend()
    coordinator = ControlCoordinator(arm, gripper)
    coordinator.notify_teleop_started()

    result = coordinator.dispatch(
        ActionCommand([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], gripper=0.4, source=ControlSource.TELEOP)
    )

    assert result.accepted
    assert coordinator.snapshot().phase == SystemPhase.TELEOP
    assert coordinator.active_source == ControlSource.TELEOP
    assert len(arm.commands) == 1
    assert gripper.values == [0.4]


def test_coordinator_estop_blocks_motion_and_accepts_safety_zero() -> None:
    arm = RecordingArmBackend()
    gripper = RecordingGripperBackend()
    coordinator = ControlCoordinator(arm, gripper)
    coordinator.notify_inference_ready(True)
    coordinator.notify_inference_execution(True)
    coordinator.notify_estop(True)

    motion = coordinator.dispatch(
        ActionCommand([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], source=ControlSource.INFERENCE)
    )
    safety = coordinator.publish_zero()

    assert not motion.accepted
    assert motion.reason == "estopped"
    assert safety.accepted
    assert coordinator.snapshot().phase == SystemPhase.ESTOP
    assert len(arm.commands) == 1
    assert arm.commands[0].source == ControlSource.SAFETY

