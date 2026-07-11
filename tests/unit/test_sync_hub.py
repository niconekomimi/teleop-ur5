import numpy as np

from teleop_control_py.core.models import CameraFrameSet, RobotStateSnapshot
from teleop_control_py.core.sync_hub import SyncHub


def _frames() -> CameraFrameSet:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    return CameraFrameSet(frame, frame, 100, 101, 100, 0.001)


def _state() -> RobotStateSnapshot:
    return RobotStateSnapshot(
        joint_pos=np.zeros(6),
        eef_pos=np.zeros(3),
        eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        gripper=0.25,
    )


def test_capture_snapshot_combines_camera_state_and_action() -> None:
    hub = SyncHub(
        camera_provider=_frames,
        state_provider=lambda _frames: (_state(), None, None),
        action_provider=lambda _frames, _state: (np.arange(7, dtype=np.float32), None, None),
    )

    snapshot, reason, detail = hub.capture_snapshot()

    assert reason is None
    assert detail is None
    assert snapshot is not None
    assert snapshot.action_vector.tolist() == list(range(7))
    assert snapshot.robot_state.gripper == 0.25


def test_capture_snapshot_reports_missing_camera() -> None:
    hub = SyncHub(
        camera_provider=lambda: None,
        state_provider=lambda _frames: (_state(), None, None),
    )

    snapshot, reason, detail = hub.capture_snapshot()

    assert snapshot is None
    assert reason == "camera_empty"
    assert detail is None


def test_capture_snapshot_preserves_provider_failure_reason() -> None:
    hub = SyncHub(
        camera_provider=_frames,
        state_provider=lambda _frames: (None, "joint_stale", "age=0.5"),
    )

    snapshot, reason, detail = hub.capture_snapshot()

    assert snapshot is None
    assert reason == "joint_stale"
    assert detail == "age=0.5"

