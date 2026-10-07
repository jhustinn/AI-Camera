from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_SAMPLES = 3


@dataclass
class EnrollmentSample:
    vector: np.ndarray
    timestamp: float
    similarity: float | None = None


@dataclass
class EnrollmentSession:
    name: str
    employee_no: str | None = None
    dept: str | None = None
    target_count: int = 10
    min_interval_sec: float = 0.8
    samples: list[EnrollmentSample] = field(default_factory=list)
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    last_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def count(self) -> int:
        return len(self.samples)

    def ready(self, min_samples: int = MIN_SAMPLES) -> bool:
        return self.count >= min(min_samples, self.target_count)

    def progress(self) -> dict[str, Any]:
        elapsed = max(0.0, time.time() - self.started_at)
        return {
            "name": self.name,
            "employee_no": self.employee_no,
            "dept": self.dept,
            "status": self.status,
            "samples": self.count,
            "target": self.target_count,
            "ready": self.ready(),
            "elapsed_sec": round(elapsed, 1),
            "last_error": self.last_error,
        }

    def add_sample(self, vector: np.ndarray, similarity: float | None = None) -> bool:
        with self.lock:
            if self.status != "running":
                return False
            if self.samples:
                elapsed = time.time() - self.samples[-1].timestamp
                if elapsed < self.min_interval_sec:
                    return False
            self.samples.append(
                EnrollmentSample(
                    vector=np.asarray(vector, dtype=np.float32),
                    timestamp=time.time(),
                    similarity=similarity,
                )
            )
            return True

    def average_embedding(self) -> np.ndarray:
        if not self.samples:
            raise ValueError("belum ada sampel wajah")
        matrix = np.stack([s.vector for s in self.samples])
        mean = matrix.mean(axis=0)
        return mean / max(float(np.linalg.norm(mean)), 1e-8)

    def finish(self) -> np.ndarray:
        with self.lock:
            if self.status != "running":
                raise ValueError(f"sesi enrollment sudah {self.status}")
            embedding = self.average_embedding()
            self.status = "committed"
            return embedding

    def cancel(self) -> None:
        with self.lock:
            self.status = "cancelled"
            self.samples.clear()