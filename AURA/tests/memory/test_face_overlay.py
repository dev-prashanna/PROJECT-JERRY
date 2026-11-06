"""Face overlay worker: live boxes + names for the camera preview."""

from __future__ import annotations

import time

import numpy as np

from src.memory.face_memory import FaceMatchThresholds, FaceMemoryService
from src.memory.store import MemoryStore
from src.perception.face_overlay_service import (
    FaceBox,
    FaceOverlayService,
    draw_face_boxes,
)
from tests.conftest import FakeCamera
from tests.memory.conftest import FakeFaceBackend, face_image


def _face_setup(tmp_path):
    store = MemoryStore(tmp_path / "faces.db")
    service = FaceMemoryService(store, FakeFaceBackend(), FaceMatchThresholds())
    return store, service


def _wait_for_overlay(overlay: FaceOverlayService, timeout_s: float = 3.0):
    deadline = time.monotonic() + timeout_s
    latest = None
    while time.monotonic() < deadline:
        latest = overlay.latest()
        if latest is not None and latest.faces:
            return latest
        time.sleep(0.05)
    return latest


def test_overlay_detects_an_unenrolled_face(tmp_path):
    store, face_service = _face_setup(tmp_path)
    overlay = FaceOverlayService(FakeCamera(face_image()), face_service, max_fps=60)
    overlay.start()
    try:
        latest = _wait_for_overlay(overlay)
        assert latest is not None and latest.faces
        box = latest.faces[0]
        assert box.label == "unknown"
        assert not box.matched
        assert latest.drawable()
        assert overlay.status()["frames"] >= 1
    finally:
        overlay.stop()
    store.close()


def test_overlay_labels_an_enrolled_face(tmp_path):
    store, face_service = _face_setup(tmp_path)
    face_service.enroll_identity(
        "Prashanna", face_image(), transcript="remember my face as Prashanna"
    )
    overlay = FaceOverlayService(FakeCamera(face_image()), face_service, max_fps=60)
    overlay.start()
    try:
        latest = _wait_for_overlay(overlay)
        assert latest is not None and latest.faces
        box = latest.faces[0]
        assert box.matched
        assert box.label == "Prashanna"
    finally:
        overlay.stop()
    store.close()


def test_overlay_service_tolerates_a_missing_camera_image(tmp_path):
    store, face_service = _face_setup(tmp_path)

    class _BrokenCamera:
        class buffer:
            @staticmethod
            def get(copy=True):
                raise RuntimeError("camera exploded")

    overlay = FaceOverlayService(_BrokenCamera(), face_service, max_fps=60)
    overlay.start()
    try:
        time.sleep(0.2)
        assert overlay.latest() is None
        assert overlay.last_error
    finally:
        overlay.stop()
    store.close()


def test_draw_face_boxes_stamps_without_mutating_the_source():
    image = face_image()
    boxes = [
        FaceBox(bbox_xyxy=(53, 30, 106, 60), label="Prashanna", matched=True, confidence=0.9),
        FaceBox(bbox_xyxy=(0, 0, 10, 10), label="unknown", matched=False, confidence=0.5),
    ]
    out = draw_face_boxes(image, boxes)
    assert out.shape == image.shape
    assert not np.shares_memory(out, image)
    assert not np.array_equal(out, image)
    # degenerate boxes must be ignored rather than raising
    draw_face_boxes(image, [FaceBox(bbox_xyxy=(90, 90, 10, 10), label="x", matched=False, confidence=0.1)])
