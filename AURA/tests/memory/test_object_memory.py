"""Object isolation, embedding determinism and engine-level object memory."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.models import EnrollObjectRequest, RecognizeObjectsRequest
from src.memory.object_isolation import (
    BoundingBoxFallbackSegmenter,
    CenterPriorGrounder,
    GroundedRegion,
    crop_with_margin,
    create_grounder,
    create_segmenter,
)
from src.memory.visual_embeddings import SiftColorEmbedder, create_embedder
from tests.memory.conftest import FrameStub, make_test_image


def test_center_prior_returns_single_region():
    img = make_test_image()
    regions = CenterPriorGrounder().locate(img, "motor controller")
    assert len(regions) == 1
    x1, y1, x2, y2 = regions[0].bbox_xyxy
    assert 0 <= x1 < x2 <= img.shape[1]
    assert 0 <= y1 < y2 <= img.shape[0]


def test_segmenter_crops_bbox():
    img = make_test_image()
    crop = BoundingBoxFallbackSegmenter().segment(img, (10, 20, 60, 80))
    assert crop.shape[0] == 60 and crop.shape[1] == 50


def test_crop_with_margin_clamped():
    img = make_test_image()
    crop = crop_with_margin(img, (0, 0, 160, 120), margin=0.2)
    assert crop.shape[0] <= 120 and crop.shape[1] <= 160


def test_grounder_and_segmenter_factories():
    assert isinstance(create_grounder("center-prior"), CenterPriorGrounder)
    assert isinstance(create_segmenter("bbox-crop"), BoundingBoxFallbackSegmenter)


def test_embedder_deterministic_and_distinct():
    emb = SiftColorEmbedder()
    a = make_test_image(seed=1, color=(0, 180, 220))
    b = make_test_image(seed=1, color=(0, 180, 220))
    c = make_test_image(seed=2, color=(200, 40, 40))
    va, vb, vc = emb.encode(a), emb.encode(b), emb.encode(c)
    assert va.shape == vb.shape
    assert np.allclose(va, vb, atol=1e-5)
    d_same = float(np.linalg.norm(va - vb))
    d_diff = float(np.linalg.norm(va - vc))
    assert d_same < d_diff


def test_create_embedder_returns_sift_color():
    e = create_embedder("sift-color-v1")
    assert isinstance(e, SiftColorEmbedder)


def test_engine_object_enroll_recognize_and_novelty(engine):
    img = make_test_image(seed=3, color=(0, 180, 220))
    res = engine.enroll_object(
        EnrollObjectRequest(
            object_name="custom motor controller",
            frame=FrameStub(img),
            text_query="circuit board",
        )
    )
    assert res.ok
    assert res.payload.get("entity_id")

    rec = engine.recognize_objects(
        RecognizeObjectsRequest(frame=FrameStub(img), text_query="device")
    )
    assert rec.ok
    assert "motor controller" in rec.message

    # Structurally unrelated image (half dark / half bright) falls below the
    # novelty threshold and must not match the enrolled object.
    h, w = 120, 160
    novel_img = np.full((h, w, 3), 10, dtype=np.uint8)
    novel_img[:, w // 2 :] = 250
    rec2 = engine.recognize_objects(
        RecognizeObjectsRequest(frame=FrameStub(novel_img), text_query="device")
    )
    assert rec2.ok
    strong = [
        m for m in rec2.payload.get("matches", []) if m.get("status") == "matched"
    ]
    assert not strong
    assert "do not remember" in rec2.message or "not sure" in rec2.message


def test_object_memory_disabled(tmp_path):
    from config.settings import Settings, reset_settings_cache
    from src.memory.engine import MultimodalMemoryEngine

    reset_settings_cache()
    try:
        s = Settings().with_overrides(
            **{
                "memory.enabled": True,
                "memory.db_path": str(tmp_path / "mem.db"),
                "memory.asset_dir": str(tmp_path / "assets"),
                "memory.object_memory_enabled": False,
                "memory.background_indexing": False,
            }
        )
        eng = MultimodalMemoryEngine(s.memory)
        eng.initialize()
        try:
            res = eng.enroll_object(
                EnrollObjectRequest(object_name="x", frame=FrameStub(make_test_image()))
            )
            assert not res.ok
        finally:
            eng.shutdown()
    finally:
        reset_settings_cache()
