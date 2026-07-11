import numpy as np
import pytest

from teleop_control_py.core.action_buffer import LatestActionBuffer


def test_action_buffer_returns_a_fresh_copy() -> None:
    buffer = LatestActionBuffer()
    action = np.arange(7, dtype=np.float32)

    accepted, reason = buffer.update(action, received_monotonic=10.0)
    action[0] = 99.0
    result = buffer.read(max_age_sec=0.25, now_monotonic=10.1)

    assert accepted
    assert reason == ""
    assert result.reason == ""
    assert result.age_sec == pytest.approx(0.1)
    assert result.action is not None
    assert result.action.tolist() == list(range(7))


def test_action_buffer_rejects_non_finite_values() -> None:
    buffer = LatestActionBuffer()
    buffer.update([0.1] * 7)

    accepted, reason = buffer.update([0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0])

    assert not accepted
    assert reason == "action_not_finite"
    assert buffer.read(max_age_sec=1.0).reason == "action_empty"


def test_action_buffer_rejects_short_actions() -> None:
    buffer = LatestActionBuffer()

    accepted, reason = buffer.update([0.0] * 6)

    assert not accepted
    assert reason == "action_too_short"


def test_action_buffer_expires_old_action() -> None:
    buffer = LatestActionBuffer()
    buffer.update([0.1] * 7, received_monotonic=20.0)

    result = buffer.read(max_age_sec=0.25, now_monotonic=20.251)

    assert result.action is None
    assert result.reason == "action_stale"
    assert result.age_sec is not None
    assert result.age_sec > 0.25


def test_action_buffer_clear_removes_previous_command() -> None:
    buffer = LatestActionBuffer()
    buffer.update([0.1] * 7)

    buffer.clear()

    assert buffer.read(max_age_sec=1.0).reason == "action_empty"
