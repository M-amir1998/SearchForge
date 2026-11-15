"""Approximate peak-RSS measurement by background sampling (psutil)."""

from __future__ import annotations

import threading
from types import TracebackType

import psutil


class PeakRSSMonitor:
    """Context manager sampling this process's RSS every ``interval_s`` seconds.

    Sampling can miss very short spikes; reported values are approximate (documented
    as such in reports).
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss = 0
        self.peak_rss = 0

    def _sample(self) -> None:
        rss = int(self._proc.memory_info().rss)
        self.peak_rss = max(self.peak_rss, rss)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> PeakRSSMonitor:
        self.start_rss = int(self._proc.memory_info().rss)
        self.peak_rss = self.start_rss
        self._thread = threading.Thread(target=self._run, name="peak-rss", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()

    @property
    def peak_mb(self) -> float:
        return round(self.peak_rss / (1024 * 1024), 1)

    @property
    def delta_mb(self) -> float:
        return round((self.peak_rss - self.start_rss) / (1024 * 1024), 1)
