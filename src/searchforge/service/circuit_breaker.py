"""A small thread-safe circuit breaker for remote backends (CLOSED → OPEN → HALF_OPEN)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum

from searchforge.errors import CircuitOpenError


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Opens after ``failure_threshold`` consecutive failures; after ``reset_timeout_s``
    one trial call is allowed (HALF_OPEN). Success closes it, failure re-opens it."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self.reset_timeout_s:
            self._state = BreakerState.HALF_OPEN
            self._trial_in_flight = False

    def _before(self) -> None:
        with self._lock:
            self._maybe_half_open()
            if self._state is BreakerState.OPEN:
                raise CircuitOpenError(f"circuit '{self.name}' is open")
            if self._state is BreakerState.HALF_OPEN:
                if self._trial_in_flight:
                    raise CircuitOpenError(f"circuit '{self.name}' is half-open (trial in flight)")
                self._trial_in_flight = True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = BreakerState.CLOSED
            self._trial_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is BreakerState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = self._clock()
            self._trial_in_flight = False

    @contextmanager
    def guard(self) -> Iterator[None]:
        self._before()
        try:
            yield
        except Exception:
            self.record_failure()
            raise
        self.record_success()
