"""Camera tests: frame buffer concurrency, freshness, snapshot semantics."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.perception.frame_buffer import Frame, FrameBuffer
from tests.conftest import FakeCamera, _test_image


class TestFrameBuffer:
    def test_put_and_get(self):
        buf = FrameBuffer()
        assert buf.get() is None
        frame = buf.put(_test_image(), source="cam")
        got = buf.get()
        assert got.frame_id == frame.frame_id
        assert got.width == 64 and got.height == 48
        assert got.tag == "frame_0001"

    def test_only_latest_frame_is_kept(self):
        buf = FrameBuffer()
        ids = [buf.put(_test_image(i)).frame_id for i in range(1, 6)]
        assert ids == [1, 2, 3, 4, 5]
        assert buf.get().frame_id == 5
        assert buf.stats["written"] == 5

    def test_get_returns_a_copy(self):
        buf = FrameBuffer()
        buf.put(_test_image())
        a = buf.get()
        a.image[0, 0, 0] = 255
        b = buf.get()
        assert b.image[0, 0, 0] != 255

    def test_stale_frames_are_rejected(self):
        buf = FrameBuffer(max_age_s=0.05)
        buf.put(_test_image())
        time.sleep(0.1)
        assert buf.latest_fresh() is None
        assert buf.get() is not None          # raw access still works
        assert buf.stats["dropped_stale"] == 1

    def test_explicit_age_override(self):
        buf = FrameBuffer(max_age_s=0.01)
        buf.put(_test_image())
        time.sleep(0.05)
        assert buf.latest_fresh(max_age_s=10.0) is not None

    def test_concurrent_writers_never_corrupt(self):
        buf = FrameBuffer()
        errors: list[Exception] = []
        stop = threading.Event()

        def writer(n: int):
            try:
                while not stop.is_set():
                    buf.put(_test_image(n))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def reader():
            try:
                while not stop.is_set():
                    frame = buf.get()
                    if frame is not None:
                        assert frame.image.shape == (48, 64, 3)
                        assert frame.frame_id >= 1
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(0.4)
        stop.set()
        for t in threads:
            t.join(timeout=2)
        assert not errors
        assert buf.stats["written"] > 0

    def test_clear(self):
        buf = FrameBuffer()
        buf.put(_test_image())
        buf.clear()
        assert buf.empty and buf.get() is None

    def test_frame_age_and_copy(self):
        buf = FrameBuffer()
        f = buf.put(_test_image())
        assert f.age_s >= 0
        c = f.copy()
        assert c.frame_id == f.frame_id and c.image is not f.image

    def test_snapshot_image_is_immutable(self):
        buf = FrameBuffer()
        buf.put(_test_image())
        first = buf.get(copy=True)
        buf.put(_test_image(10))
        assert first.frame_id == 1


class TestFakeCameraContract:
    """The fake must behave like the real CameraManager for the agent."""

    def test_snapshot_returns_frame(self):
        cam = FakeCamera()
        cam.start()
        frame = cam.snapshot()
        assert isinstance(frame, Frame)
        assert frame.image.shape == (48, 64, 3)
        cam.close()
        assert cam.closed

    def test_snapshot_failure_raises_camera_error(self):
        from src.utils.exceptions import CameraError

        cam = FakeCamera(fail=True)
        with pytest.raises(CameraError):
            cam.snapshot()

    def test_info_keys_match_real_camera(self):
        from src.perception.camera_manager import CameraManager

        expected = set(CameraManager().info().keys())
        assert expected == set(FakeCamera().info().keys())


def test_real_camera_blank_frame_detection():
    """Unit-level check of the blank-frame heuristic (no hardware needed)."""
    from src.perception.camera_manager import CameraManager

    cm = CameraManager()
    cm._check_blank(np.zeros((48, 64, 3), dtype=np.uint8))
    for _ in range(20):
        cm._check_blank(np.zeros((48, 64, 3), dtype=np.uint8))
    assert cm.info()["blank_warning"] is not None

    for _ in range(3):
        cm._check_blank(_test_image())
    assert cm.info()["blank_warning"] is None
