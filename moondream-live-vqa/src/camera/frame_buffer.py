"""Thread-safe *latest frame* buffer.

The camera runs at ~30 FPS while a VLM question can take a second or more.
Buffering every frame would waste memory and hand the model stale images, so
only the newest frame is kept: :class:`LatestFrameBuffer` stores exactly one
frame and every reader receives an independent copy.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FrameSnapshot:
    """An immutable, self-contained capture of one camera frame.

    ``frame`` is a private copy: mutating it never affects the camera or other
    consumers.  ``frame_id``/``timestamp`` bind an inference result back to the
    exact image that produced it.
    """

    frame: np.ndarray
    frame_id: int
    timestamp: float
    width: int
    height: int
    source_fps: float = 0.0
    camera_index: int = -1
    mode: str = "live"

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.timestamp)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.frame.shape)

    def describe(self) -> str:
        return (
            f"frame #{self.frame_id} @ {self.timestamp:.3f} "
            f"({self.width}x{self.height})"
        )


class LatestFrameBuffer:
    """Single-slot, lock-protected frame storage."""

    __slots__ = ("_lock", "_snapshot", "_frame_id", "_updates", "_rejected")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: Optional[FrameSnapshot] = None
        self._frame_id = 0
        self._updates = 0
        self._rejected = 0

    # -- write --------------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        *,
        source_fps: float = 0.0,
        camera_index: int = -1,
        mode: str = "live",
        timestamp: Optional[float] = None,
    ) -> FrameSnapshot:
        """Store ``frame`` as the newest image and return its snapshot.

        Raises ``ValueError`` for empty/invalid frames, which are recorded as
        rejected so callers can surface corruption instead of crashing.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            with self._lock:
                self._rejected += 1
            raise ValueError(
                f"refusing to buffer an invalid frame (expected HxWx3 array, "
                f"got {type(frame).__name__} shape={getattr(frame, 'shape', None)})"
            )
        if frame.size == 0 or frame.shape[0] < 1 or frame.shape[1] < 1:
            with self._lock:
                self._rejected += 1
            raise ValueError("refusing to buffer an empty frame")

        stored = np.ascontiguousarray(frame).copy()
        ts = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            self._frame_id += 1
            snapshot = FrameSnapshot(
                frame=stored,
                frame_id=self._frame_id,
                timestamp=ts,
                width=int(stored.shape[1]),
                height=int(stored.shape[0]),
                source_fps=float(source_fps),
                camera_index=int(camera_index),
                mode=mode,
            )
            self._snapshot = snapshot
            self._updates += 1
        return snapshot

    # -- read ---------------------------------------------------------------
    def get_latest(self, copy: bool = True) -> Optional[FrameSnapshot]:
        """Return the newest snapshot.

        With ``copy=True`` (default) the returned ``frame`` array is a fresh
        copy, so callers may freely mutate it.
        """
        with self._lock:
            snap = self._snapshot
        if snap is None:
            return None
        if not copy:
            return snap
        return FrameSnapshot(
            frame=snap.frame.copy(),
            frame_id=snap.frame_id,
            timestamp=snap.timestamp,
            width=snap.width,
            height=snap.height,
            source_fps=snap.source_fps,
            camera_index=snap.camera_index,
            mode=snap.mode,
        )

    def get_array(self, copy: bool = True) -> Optional[np.ndarray]:
        snap = self.get_latest(copy=copy)
        return None if snap is None else snap.frame

    def clear(self) -> None:
        with self._lock:
            self._snapshot = None

    # -- stats --------------------------------------------------------------
    @property
    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    @property
    def update_count(self) -> int:
        with self._lock:
            return self._updates

    @property
    def rejected_count(self) -> int:
        with self._lock:
            return self._rejected

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._snapshot is None

    def stats(self) -> dict:
        with self._lock:
            snap = self._snapshot
            return {
                "frame_id": self._frame_id,
                "updates": self._updates,
                "rejected": self._rejected,
                "empty": snap is None,
                "age_seconds": (time.time() - snap.timestamp) if snap else None,
            }
