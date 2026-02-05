from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import signal
import time
from typing import Callable


LOGGER = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    interval_minutes: int = 60


class Scheduler:
    def __init__(self, config: ScheduleConfig, job: Callable[[], None]) -> None:
        self.config = config
        self.job = job
        self._stop = False

    def _handle_stop(self, *_: object) -> None:
        self._stop = True
        LOGGER.info("Scheduler stopping...")

    def run_forever(self) -> None:
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        LOGGER.info("Scheduler started with %s minutes interval", self.config.interval_minutes)
        while not self._stop:
            start = time.time()
            try:
                LOGGER.info("Job started at %s", datetime.now().isoformat(timespec="seconds"))
                self.job()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Job failed: %s", exc)

            elapsed = time.time() - start
            sleep_seconds = max(0, self.config.interval_minutes * 60 - elapsed)
            if sleep_seconds:
                time.sleep(sleep_seconds)
