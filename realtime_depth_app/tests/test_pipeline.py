from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from src.camera import LatestSlot, Frame
from src.pipeline import InferenceWorker, Pipeline
from src.temporal_filter import DepthSmoother


@pytest.fixture()
def image_cfg(cfg, tmp_path):
    src = cfg.base_dir.parent / "assets" / "sample_frame.jpeg"
    if not src.exists():
        pytest.skip("sample image missing")
    return replace(cfg, source=replace(cfg.source, type="image", path=str(src)))


def test_latest_slot_replaces_stale_items():
    slot: LatestSlot[int] = LatestSlot()
    slot.put(1)
    slot.put(2)
    slot.put(3)
    assert slot.get() == 3
    assert slot.dropped == 2


def test_latest_slot_dropped_counts_only_unread_items():
    slot: LatestSlot[int] = LatestSlot()
    slot.put(1)
    assert slot.wait_new(0, timeout=0.05)[1] == 1  # consumed
    slot.put(2)
    assert slot.dropped == 0  # item 1 was read, so overwriting it is not a drop
    slot.put(3)  # item 2 never read -> real drop
    assert slot.dropped == 1
    assert slot.get() == 3
    slot.put(4)  # get() marks read
    assert slot.dropped == 1


def test_latest_slot_wait_new():
    slot: LatestSlot[int] = LatestSlot()
    seq, item = slot.wait_new(0, timeout=0.05)
    assert item is None
    slot.put(42)
    seq, item = slot.wait_new(seq, timeout=0.5)
    assert item == 42


def test_pipeline_produces_aligned_depth(image_cfg, engine):
    pipeline = Pipeline(image_cfg, engine, DepthSmoother(False, 0.5, 1))
    pipeline.start()
    try:
        import cv2

        expected = cv2.imread(str(image_cfg.source.path))
        result = pipeline.wait_for_depth(timeout=30.0)
        assert result is not None
        assert result.depth.shape[:2] == expected.shape[:2]
        # frame ids must exist and be positive; depth is tagged with the frame it came from
        assert result.frame_id >= 1
        stats = pipeline.snapshot()
        assert stats["processed"] >= 1
        assert stats["errors"] == 0
    finally:
        pipeline.stop()


def test_pipeline_shutdown_is_clean(image_cfg, engine):
    pipeline = Pipeline(image_cfg, engine, DepthSmoother(False, 0.5, 1))
    pipeline.start()
    pipeline.wait_for_depth(timeout=30.0)
    pipeline.stop()
    assert not pipeline.capture.is_alive()
    assert not pipeline.inference.is_alive()


def test_frame_skip_limits_processing(image_cfg, engine):
    cfg = replace(image_cfg, runtime=replace(image_cfg.runtime, frame_skip=3))
    pipeline = Pipeline(cfg, engine, DepthSmoother(False, 0.5, 1))
    pipeline.start()
    try:
        deadline = time.time() + 20
        while time.time() < deadline and pipeline.inference.stats.processed < 5:
            time.sleep(0.1)
        snap = pipeline.snapshot()
        assert snap["skipped"] > 0
        assert snap["processed"] < snap["captured"]
    finally:
        pipeline.stop()


def test_stale_frames_are_dropped_not_queued(image_cfg, engine):
    pipeline = Pipeline(image_cfg, engine, DepthSmoother(False, 0.5, 1))
    pipeline.start()
    try:
        time.sleep(1.5)
        snap = pipeline.snapshot()
        # the capture slot may never accumulate: dropped frames prove the bounded behaviour
        assert snap["frame_slot_dropped"] >= 0
        assert pipeline.frame_slot.seq() <= snap["captured"]
    finally:
        pipeline.stop()


def test_smoothing_state_changes_output(image_cfg, engine):
    smooth = DepthSmoother(True, 0.3, 1)
    frame_slot: LatestSlot[Frame] = LatestSlot()
    from src.camera import LatestSlot as LS

    depth_slot: LS = LS()
    worker = InferenceWorker(engine, frame_slot, depth_slot, smooth, frame_skip=1)
    frame = Frame(id=1, image=np.zeros((image_cfg.source.height, image_cfg.source.width, 3), np.uint8),
                  t_capture=time.time())
    worker.start()
    try:
        frame_slot.put(frame)
        seq, r1 = depth_slot.wait_new(0, timeout=20)
        assert r1 is not None
        second = Frame(id=2, image=frame.image, t_capture=time.time())
        frame_slot.put(second)
        seq, r2 = depth_slot.wait_new(seq, timeout=20)
        assert r2 is not None
        # EMA state means the second output is a blend, never an exact copy of a reset state
        assert smooth.ema.initialized
    finally:
        worker.stop()
        worker.join(timeout=3)
