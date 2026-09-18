"""Phase-local estimates that exclude downloads and already completed work."""
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    done: float = 0
    total: float = 0
    device: str = ""


class TimeEstimator:
    def __init__(self) -> None:
        self.phase = ""
        self.started = 0.0
        self.baseline = 0.0
        self.done = 0.0
        self.total = 0.0
        self.updated = 0.0
        self.seconds_per_unit: float | None = None

    def update(self, event: ProgressEvent, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if event.phase != self.phase or event.done < self.done:
            self.phase, self.started, self.baseline = event.phase, now, event.done
            self.seconds_per_unit = None
        self.done, self.total, self.updated = event.done, event.total, now
        advanced = self.done - self.baseline
        elapsed = now - self.started
        if advanced > 0 and elapsed >= 2:
            self.seconds_per_unit = elapsed / advanced

    def remaining(self, now: float | None = None) -> float | None:
        now = time.monotonic() if now is None else now
        if self.total > 0 and self.done >= self.total:
            return 0
        if self.phase not in {"transcribing", "diarizing", "speaker_download"} or self.seconds_per_unit is None:
            return None
        expected = (self.total - self.done) * self.seconds_per_unit
        remaining = expected - max(0, now - self.updated)
        # If inference took longer than the prediction, don't show a false zero.
        return remaining if remaining >= 1 else None

