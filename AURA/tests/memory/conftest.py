"""Shared fixtures for MMS tests — hardware- and network-free."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import Settings, reset_settings_cache  # noqa: E402
from src.memory.engine import MultimodalMemoryEngine  # noqa: E402
from src.memory.face_memory import (  # noqa: E402
    DetectedFace,
    FaceMatchThresholds,
    FaceMemoryService,
)
from src.memory.store import MemoryStore  # noqa: E402


def make_test_image(seed: int = 0, color=(0, 180, 220), size=(120, 160)) -> np.ndarray:
    h, w = size
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[30:90, 40:120] = color
    noise = rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class FakeFaceBackend:
    """Deterministic face backend driven by pixel markers.

    A blue rectangle at a known location "is a face".  The embedding is the
    mean colour of that rectangle plus a seed-driven offset, 128-D unit-ish.
    """

    name = "fake-face"
    dimension = 128

    def __init__(self):
        self.fail_detect = False

    def available(self) -> bool:
        return True

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        if self.fail_detect:
            return []
        h, w = image_bgr.shape[:2]
        # Blue-ish patch in the upper-centre counts as a face
        patch = image_bgr[h // 4 : h // 2, w // 3 : 2 * w // 3]
        if patch.size == 0:
            return []
        mean_bgr = patch.reshape(-1, 3).mean(axis=0)
        if mean_bgr[0] > mean_bgr[2] + 30:  # blue-dominant
            return [
                DetectedFace(
                    bbox_xyxy=(w // 3, h // 4, 2 * w // 3, h // 2),
                    confidence=0.95,
                )
            ]
        return []

    def embed(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in face.bbox_xyxy]
        patch = image_bgr[y1:y2, x1:x2]
        # 8x8 spatial grid of channel means (64*3 = 192 -> trim to 128)
        gh = max(1, patch.shape[0] // 8)
        gw = max(1, patch.shape[1] // 8)
        grid = np.zeros((8, 8, 3), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                cell = patch[i * gh : (i + 1) * gh, j * gw : (j + 1) * gw]
                if cell.size:
                    grid[i, j] = cell.reshape(-1, 3).mean(axis=0)
        vec = grid.reshape(-1)[:128].astype(np.float32) / 255.0
        if vec.shape[0] < 128:
            vec = np.pad(vec, (0, 128 - vec.shape[0]))
        norm = float(np.linalg.norm(vec))
        return vec / (norm + 1e-12)


def face_image(seed: int = 0) -> np.ndarray:
    img = np.full((120, 160, 3), 90, dtype=np.uint8)
    # blue-dominant "face" patch with a seed-dependent stripe pattern
    rng = np.random.default_rng(seed)
    patch = np.array([220, 80, 60], dtype=np.int16)  # BGR: blue heavy
    noise = rng.integers(-15, 15, (30, 53, 3), dtype=np.int16)
    region = np.clip(patch + noise, 0, 255).astype(np.int16)
    phase = (seed * 3) % 7
    stripe_rows = np.where((np.arange(30) + phase) % 7 < 3)[0]
    for r in stripe_rows:
        region[r] = np.clip(region[r] - 90, 0, 255)
    img[30:60, 53:106] = region.astype(np.uint8)
    return img


@pytest.fixture
def memory_settings(tmp_path) -> Settings:
    reset_settings_cache()
    s = Settings().with_overrides(
        **{
            "memory.enabled": True,
            "memory.db_path": str(tmp_path / "mem.db"),
            "memory.asset_dir": str(tmp_path / "assets"),
            "memory.face_recognition_enabled": False,
            "memory.object_memory_enabled": True,
            "memory.object_grounder": "center-prior",
            "memory.background_indexing": False,
            "memory.auto_learn_facts": False,
            "memory.clip_allow_download": False,
        }
    )
    yield s
    reset_settings_cache()


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    s = MemoryStore(tmp_path / "store.db")
    yield s
    s.close()


@pytest.fixture
def engine(memory_settings) -> MultimodalMemoryEngine:
    eng = MultimodalMemoryEngine(memory_settings.memory)
    eng.initialize()
    yield eng
    eng.shutdown()


@pytest.fixture
def face_service(store) -> FaceMemoryService:
    backend = FakeFaceBackend()
    return FaceMemoryService(
        store,
        backend,
        FaceMatchThresholds(match_distance=0.55, ambiguity_margin=0.05),
        require_confirmation=True,
    )


class FrameStub:
    def __init__(self, image: np.ndarray, frame_id: str = "frame_0001"):
        self.image = image
        self.tag = frame_id
        self.frame_id = frame_id
