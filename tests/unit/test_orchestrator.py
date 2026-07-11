from teleop_control_py.core.models import ControlSource
from teleop_control_py.core.orchestrator import SystemOrchestrator, SystemPhase


def test_inference_execution_is_blocked_while_teleop_runs() -> None:
    orchestrator = SystemOrchestrator()
    orchestrator.notify_teleop_started()
    orchestrator.notify_inference_ready(True)

    decision = orchestrator.request_enable_inference_execution()

    assert not decision.allowed
    assert decision.reason == "teleop_running"
    assert decision.state.phase == SystemPhase.TELEOP
    assert decision.state.active_source == ControlSource.TELEOP


def test_teleop_is_blocked_while_inference_executes() -> None:
    orchestrator = SystemOrchestrator()
    orchestrator.notify_inference_ready(True)
    orchestrator.notify_inference_execution(True)

    decision = orchestrator.request_start_teleop()

    assert not decision.allowed
    assert decision.reason == "inference_executing"
    assert decision.state.phase == SystemPhase.INFERENCE_EXECUTING


def test_commander_phase_temporarily_preempts_teleop() -> None:
    orchestrator = SystemOrchestrator()
    orchestrator.notify_teleop_started()

    active = orchestrator.notify_homing(True)
    resumed = orchestrator.notify_homing(False)

    assert active.phase == SystemPhase.HOMING
    assert active.active_source == ControlSource.COMMANDER
    assert resumed.phase == SystemPhase.TELEOP
    assert resumed.active_source == ControlSource.TELEOP


def test_estop_clears_motion_phases_and_can_be_released() -> None:
    orchestrator = SystemOrchestrator()
    orchestrator.notify_inference_ready(True)
    orchestrator.notify_inference_execution(True)
    orchestrator.notify_home_zone(True)

    stopped = orchestrator.notify_estop(True)
    released = orchestrator.clear_estop()

    assert stopped.phase == SystemPhase.ESTOP
    assert stopped.active_source == ControlSource.SAFETY
    assert not stopped.inference_executing
    assert not stopped.home_zone_active
    assert released.phase == SystemPhase.INFERENCE_READY

