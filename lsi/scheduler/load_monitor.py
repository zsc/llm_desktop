from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LoadSample:
    cpu_percent: float
    load1: float | None
    mem_available_gb: float
    cores: int


def default_sampler() -> LoadSample:
    import psutil

    cpu = float(psutil.cpu_percent(interval=None))
    try:
        load1 = float(os.getloadavg()[0])
    except Exception:
        load1 = None
    mem_avail = float(psutil.virtual_memory().available) / (1024**3)
    cores = int(os.cpu_count() or 1)
    return LoadSample(cpu_percent=cpu, load1=load1, mem_available_gb=mem_avail, cores=cores)


class LoadMonitor:
    def __init__(
        self,
        *,
        cpu_pause_percent: float,
        mem_available_min_gb: float,
        pause_after_s: float,
        resume_after_s: float,
        sampler: Callable[[], LoadSample] = default_sampler,
        sample_interval_s: float = 1.0,
    ):
        self.cpu_pause_percent = float(cpu_pause_percent)
        self.mem_available_min_gb = float(mem_available_min_gb)
        self.pause_after_s = float(pause_after_s)
        self.resume_after_s = float(resume_after_s)
        self.sampler = sampler
        self.sample_interval_s = float(sample_interval_s)

        self._paused = False
        self._pause_reason: str | None = None
        self._paused_since: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause_event: threading.Event | None = None

        self._high_since: float | None = None
        self._low_since: float | None = None

    def attach_pause_event(self, ev: threading.Event) -> None:
        self._pause_event = ev

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lsi-load-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def pause_reason(self) -> str | None:
        return self._pause_reason

    @property
    def paused_for_s(self) -> float | None:
        if not self._paused or self._paused_since is None:
            return None
        return max(0.0, time.time() - self._paused_since)

    def _is_high(self, s: LoadSample) -> tuple[bool, str | None]:
        if s.mem_available_gb < self.mem_available_min_gb:
            return True, f"mem_available_gb<{self.mem_available_min_gb:.2f}"
        if s.cpu_percent > self.cpu_pause_percent:
            return True, f"cpu>{self.cpu_pause_percent:.0f}%"
        if s.load1 is not None:
            if s.load1 > s.cores * 0.8:
                return True, f"load1>{s.cores * 0.8:.2f}"
        return False, None

    def _set_paused(self, paused: bool, reason: str | None) -> None:
        self._paused = paused
        self._pause_reason = reason
        self._paused_since = time.time() if paused else None
        if self._pause_event is not None:
            if paused:
                self._pause_event.set()
            else:
                self._pause_event.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            try:
                sample = self.sampler()
                high, reason = self._is_high(sample)
            except Exception:
                high, reason = False, None

            if high:
                self._low_since = None
                if self._high_since is None:
                    self._high_since = now
                if not self._paused and (now - self._high_since) >= self.pause_after_s:
                    self._set_paused(True, reason or "high_load")
            else:
                self._high_since = None
                if self._paused:
                    if self._low_since is None:
                        self._low_since = now
                    if (now - self._low_since) >= self.resume_after_s:
                        self._set_paused(False, None)
                else:
                    self._low_since = None

            time.sleep(self.sample_interval_s)

