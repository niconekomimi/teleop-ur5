"""Thread-safe validation and freshness tracking for control actions."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class ActionBufferRead:
    action: Optional[np.ndarray]
    reason: str = ""
    age_sec: Optional[float] = None


class LatestActionBuffer:
    def __init__(self, size: int = 7, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._size = max(1, int(size))
        self._clock = clock
        self._lock = threading.Lock()
        self._action: Optional[np.ndarray] = None
        self._received_monotonic: Optional[float] = None

    def _reject(self, reason: str) -> tuple[bool, str]:
        self.clear()
        return False, reason

    def update(self, values, *, received_monotonic: Optional[float] = None) -> tuple[bool, str]:
        try:
            action = np.asarray(values, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return self._reject("invalid_action")
        if action.size < self._size:
            return self._reject("action_too_short")
        action = action[: self._size]
        if not np.all(np.isfinite(action)):
            return self._reject("action_not_finite")

        stamp = self._clock() if received_monotonic is None else float(received_monotonic)
        if not np.isfinite(stamp):
            return self._reject("invalid_timestamp")

        with self._lock:
            self._action = action.copy()
            self._received_monotonic = stamp
        return True, ""

    def read(
        self,
        *,
        max_age_sec: float,
        now_monotonic: Optional[float] = None,
    ) -> ActionBufferRead:
        with self._lock:
            if self._action is None or self._received_monotonic is None:
                return ActionBufferRead(None, reason="action_empty")
            action = self._action.copy()
            received_monotonic = self._received_monotonic

        now = self._clock() if now_monotonic is None else float(now_monotonic)
        age_sec = max(0.0, now - received_monotonic)
        if age_sec > max(0.0, float(max_age_sec)):
            return ActionBufferRead(None, reason="action_stale", age_sec=age_sec)
        return ActionBufferRead(action, age_sec=age_sec)

    def clear(self) -> None:
        with self._lock:
            self._action = None
            self._received_monotonic = None
