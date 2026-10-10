"""Thread-safe single-slot frame buffer with identity metadata.

The buffer never queues frames: a slow consumer always sees the newest one.
Each frame carries an id and a capture timestamp so answers can be tied to
the exact image that produced them, and so stale frames can be rejected.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    frame_id: int
    timestamp: float
    source: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.timestamp)

    def copy(self) -> "Frame":
        return Frame(
            image=self.image.copy(),
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            source=self.source,
            width=self.width,
            height=self.height,
            fps=self.fps,
            meta=dict(self.meta),
        )

    @property
    def tag(self) -> str:
        return f"frame_{self.frame_id:04d}"


class FrameBuffer:
    def __init__(self, max_age_s: float = 1.5):
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._max_age_s = max_age_s
        self._dropped = 0
        self._written = 0

    def put(self, image: np.ndarray, **kwargs: Any) -> Frame:
        with self._lock:
            self._written += 1
            frame = Frame(
                image=image,
                frame_id=self._written,
                timestamp=time.time(),
                width=int(image.shape[1]) if image.ndim >= 2 else 0,
                height=int(image.shape[0]) if image.ndim >= 2 else 0,
                **kwargs,
            )
            self._frame = frame
            return frame

    def get(self, copy: bool = True) -> Optional[Frame]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        return frame.copy() if copy else frame

    def latest_fresh(self, max_age_s: Optional[float] = None) -> Optional[Frame]:
        """Return a frame only if it is newer than the freshness threshold."""
        limit = self._max_age_s if max_age_s is None else max_age_s
        frame = self.get(copy=True)
        if frame is None:
            return None
        if frame.age_s > limit:
            with self._lock:
                self._dropped += 1
            return None
        return frame

    def clear(self) -> None:
        with self._lock:
            self._frame = None

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"written": self._written, "dropped_stale": self._dropped}

    @property
    def empty(self) -> bool:
        with self._lock:
            return self._frame is None
