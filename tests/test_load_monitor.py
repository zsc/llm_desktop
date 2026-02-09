import itertools
import threading
import time

import unittest

from lsi.scheduler.load_monitor import LoadMonitor, LoadSample


class TestLoadMonitor(unittest.TestCase):
    def test_load_monitor_debounce_pause_and_resume(self) -> None:
        # Enough samples to satisfy debounce timing.
        seq = itertools.chain([True] * 10, [False] * 60)

        def sampler():
            high = next(seq)
            if high:
                return LoadSample(cpu_percent=99.0, load1=99.0, mem_available_gb=0.1, cores=4)
            return LoadSample(cpu_percent=1.0, load1=0.0, mem_available_gb=100.0, cores=4)

        pause_event = threading.Event()
        m = LoadMonitor(
            cpu_pause_percent=70,
            mem_available_min_gb=2,
            pause_after_s=0.2,
            resume_after_s=0.3,
            sampler=sampler,
            sample_interval_s=0.05,
        )
        m.attach_pause_event(pause_event)
        m.start()
        try:
            deadline = time.time() + 2.0
            while time.time() < deadline and not pause_event.is_set():
                time.sleep(0.05)
            self.assertTrue(pause_event.is_set())

            deadline = time.time() + 2.0
            while time.time() < deadline and pause_event.is_set():
                time.sleep(0.05)
            self.assertFalse(pause_event.is_set())
        finally:
            m.stop()
