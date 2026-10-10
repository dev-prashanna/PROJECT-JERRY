"""Tests for the thread-safe frame buffer and the camera worker.

Hardware-dependent cases are skipped automatically when no camera is present
(``requires_camera``), so the suite also passes on headless machines.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from config.settings import CameraConfig
from src.camera.camera import (
    CameraSource,
    CameraStatus,
    discover_cameras,
    probe_camera,
)
from src.camera.frame_buffer import FrameSnapshot, LatestFrameBuffer
from tests.conftest import requires_camera


def make_frame(value: int = 0, size: tuple[int, int] = (16, 16)) -> np.ndarray:
    return np.full((size[0], size[1], 3), value, dtype=np.uint8)


def wait_until(predicate, timeout: float = 6.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


class TestFrameBuffer:
    def test_starts_empty(self):
        buffer = LatestFrameBuffer()
        assert buffer.is_empty
        assert buffer.get_latest() is None
        assert buffer.get_array() is None

    def test_update_stores_snapshot(self):
        buffer = LatestFrameBuffer()
        snap = buffer.update(make_frame(7), source_fps=30.0, camera_index=2)
        assert isinstance(snap, FrameSnapshot)
        assert snap.frame_id == 1
        assert snap.width == 16 and snap.height == 16
        assert snap.source_fps == 30.0
        assert snap.camera_index == 2

    def test_only_latest_frame_is_kept(self):
        buffer = LatestFrameBuffer()
        for value in range(1, 60):
            buffer.update(make_frame(value))
        assert buffer.update_count == 59
        assert buffer.get_latest().frame_id == 59
        # A single slot means memory stays constant regardless of frame count.
        assert buffer.frame_id == 59

    def test_frame_ids_increment(self):
        buffer = LatestFrameBuffer()
        ids = [buffer.update(make_frame(i)).frame_id for i in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_get_latest_returns_a_copy(self):
        buffer = LatestFrameBuffer()
        buffer.update(make_frame(1))
        first = buffer.get_latest()
        first.frame[0, 0] = 250
        second = buffer.get_latest()
        assert second.frame[0, 0, 0] == 1

    def test_reader_copy_does_not_affect_buffer(self):
        buffer = LatestFrameBuffer()
        buffer.update(make_frame(3))
        snap = buffer.get_latest(copy=True)
        snap.frame[:] = 255
        assert buffer.get_latest().frame[0, 0, 0] == 3

    def test_invalid_frames_are_rejected(self):
        buffer = LatestFrameBuffer()
        for bad in (None, np.zeros((4, 4), dtype=np.uint8), []):
            with pytest.raises(ValueError):
                buffer.update(bad)
        assert buffer.rejected_count == 3
        assert buffer.is_empty

    def test_clear(self):
        buffer = LatestFrameBuffer()
        buffer.update(make_frame())
        buffer.clear()
        assert buffer.get_latest() is None

    def test_stats(self):
        buffer = LatestFrameBuffer()
        buffer.update(make_frame())
        stats = buffer.stats()
        assert stats["updates"] == 1
        assert stats["empty"] is False
        assert stats["age_seconds"] >= 0

    def test_concurrent_writers_and_readers(self):
        buffer = LatestFrameBuffer()
        errors: list[BaseException] = []
        stop = threading.Event()

        def writer(offset: int) -> None:
            try:
                for i in range(200):
                    buffer.update(make_frame((i + offset) % 255))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            try:
                while not stop.is_set():
                    snap = buffer.get_latest()
                    if snap is not None:
                        assert snap.frame.shape == (16, 16, 3)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        readers = [threading.Thread(target=reader) for _ in range(2)]
        writers = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for thread in readers + writers:
            thread.start()
        for thread in writers:
            thread.join()
        stop.set()
        for thread in readers:
            thread.join()
        assert not errors
        assert buffer.update_count == 800


class TestDiscovery:
    def test_probe_rejects_negative_index(self):
        assert probe_camera(-1) is None

    def test_discover_returns_list(self):
        found = discover_cameras(max_index=4)
        assert isinstance(found, list)
        for info in found:
            assert info.index >= 0
            assert info.name

    @requires_camera
    def test_probe_finds_index_zero(self):
        info = probe_camera(0)
        assert info is not None
        assert info.index == 0


@requires_camera
class TestLiveCamera:
    def test_start_streams_and_stops_cleanly(self):
        camera = CameraSource(CameraConfig(device_index=0, auto_reconnect=False))
        try:
            camera.start()
            deadline = time.time() + 6
            while camera.get_latest_frame() is None and time.time() < deadline:
                time.sleep(0.05)
            snap = camera.get_latest_frame()
            assert snap is not None, "no frame within 6s"
            assert camera.status is CameraStatus.STREAMING
            assert camera.actual_fps > 0
            assert camera.resolution[0] > 0

            # snapshots are independent copies
            owned = camera.get_latest_frame(copy=True)
            owned.frame[:] = 0
            fresh = camera.get_latest_frame()
            assert fresh is not None and fresh.frame.any()
        finally:
            camera.stop()
        assert camera.status is CameraStatus.STOPPED
        assert camera.buffer.is_empty
        assert not camera.is_running

    def test_snapshot_raises_without_frames(self):
        camera = CameraSource(CameraConfig(device_index=0))
        with pytest.raises(RuntimeError):
            camera.snapshot()

    def test_bad_index_reports_error(self):
        events = []
        camera = CameraSource(
            CameraConfig(device_index=99, auto_reconnect=False),
            on_event=events.append,
        )
        camera.start()
        deadline = time.time() + 6
        while camera.status not in {
            CameraStatus.ERROR,
            CameraStatus.DISCONNECTED,
        } and time.time() < deadline:
            time.sleep(0.1)
        observed = camera.status
        camera.stop()
        assert observed in {CameraStatus.ERROR, CameraStatus.DISCONNECTED}
        assert any(e.status is CameraStatus.CONNECTING for e in events)
        assert camera.status is CameraStatus.STOPPED

    def test_start_is_idempotent_while_running(self):
        camera = CameraSource(CameraConfig(device_index=0, auto_reconnect=False))
        try:
            assert camera.start() is True
            time.sleep(0.3)
            assert camera.start() is False  # already running
        finally:
            camera.stop()

    def test_read_failures_report_disconnection(self):
        """Simulates a camera that stops delivering frames (unplugged device)."""
        events = []
        camera = CameraSource(
            CameraConfig(
                device_index=0,
                auto_reconnect=False,
                max_read_failures=3,
            ),
            on_event=events.append,
        )
        camera.start()
        assert wait_until(lambda: camera.get_latest_frame() is not None)
        assert camera.status is CameraStatus.STREAMING

        camera._capture = lambda: (False, None)  # type: ignore[method-assign]
        assert wait_until(
            lambda: camera.status is CameraStatus.DISCONNECTED, timeout=8
        ), f"status stayed at {camera.status}"
        assert any(e.status is CameraStatus.DISCONNECTED for e in events)
        camera.stop()

    def test_auto_reconnect_recovers_after_read_failures(self):
        events = []
        camera = CameraSource(
            CameraConfig(
                device_index=0,
                auto_reconnect=True,
                max_read_failures=3,
                reconnect_interval=0.1,
            ),
            on_event=events.append,
        )
        camera.start()
        assert wait_until(lambda: camera.get_latest_frame() is not None)
        original_capture = camera._capture

        camera._capture = lambda: (False, None)  # type: ignore[method-assign]
        assert wait_until(
            lambda: any(e.status is CameraStatus.DISCONNECTED for e in events),
            timeout=8,
        )

        camera._capture = original_capture  # type: ignore[method-assign]
        assert wait_until(
            lambda: camera.status is CameraStatus.STREAMING
            and camera.get_latest_frame() is not None,
            timeout=8,
        ), f"camera did not recover (status={camera.status})"
        camera.stop()

    def test_context_manager_releases_device(self):
        with CameraSource(CameraConfig(device_index=0)) as camera:
            assert wait_until(lambda: camera.get_latest_frame() is not None)
        assert camera.status is CameraStatus.STOPPED
