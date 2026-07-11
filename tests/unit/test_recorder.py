from __future__ import annotations

import h5py
import numpy as np

from teleop_control_py.core.recorder import RecorderService
from teleop_control_py.data.hdf5_writer import Sample


def _sample(demo_name: str, index: int) -> Sample:
    image = np.full((224, 224, 3), index % 255, dtype=np.uint8)
    return Sample(
        demo_name=demo_name,
        agentview_rgb=image,
        eye_in_hand_rgb=image,
        robot0_joint_pos=np.full(6, index, dtype=np.float32),
        robot0_gripper_qpos=np.array([index / 100.0], dtype=np.float32),
        robot0_eef_pos=np.full(3, index, dtype=np.float32),
        robot0_eef_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        actions=np.full(7, index, dtype=np.float32),
    )


def test_close_drains_all_queued_samples(tmp_path) -> None:
    output_path = tmp_path / "recording.hdf5"
    recorder = RecorderService(
        str(output_path),
        compression=None,
        queue_maxsize=64,
        batch_size=32,
        flush_every_n=1000,
    )
    started, _message, demo_name = recorder.start_demo()
    assert started
    assert demo_name == "demo_0"

    sample_count = 24
    for index in range(sample_count):
        assert recorder.enqueue_sample(_sample(demo_name, index))

    assert recorder.close(timeout_sec=10.0)

    with h5py.File(output_path, "r") as handle:
        demo = handle["data/demo_0"]
        assert demo.attrs["num_samples"] == sample_count
        assert demo["actions"].shape == (sample_count, 7)
        assert demo["obs/agentview_rgb"].shape == (sample_count, 224, 224, 3)
        assert demo["actions"][:, 0].tolist() == list(range(sample_count))


def test_stop_then_restart_creates_separate_demos(tmp_path) -> None:
    output_path = tmp_path / "sessions.hdf5"
    recorder = RecorderService(
        str(output_path),
        compression=None,
        queue_maxsize=16,
        batch_size=4,
        flush_every_n=4,
    )

    for expected_name in ("demo_0", "demo_1"):
        started, _message, demo_name = recorder.start_demo()
        assert started
        assert demo_name == expected_name
        assert recorder.enqueue_sample(_sample(demo_name, 1))
        stopped, _message, stopped_name = recorder.stop_demo()
        assert stopped
        assert stopped_name == expected_name

    assert recorder.close()

    with h5py.File(output_path, "r") as handle:
        assert sorted(handle["data"].keys()) == ["demo_0", "demo_1"]
        assert handle["data/demo_0"].attrs["num_samples"] == 1
        assert handle["data/demo_1"].attrs["num_samples"] == 1

