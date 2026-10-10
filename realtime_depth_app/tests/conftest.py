from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.config import load_config  # noqa: E402

SAMPLE_IMAGE = APP_ROOT.parent / "assets" / "sample_frame.jpeg"


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sample_image():
    import cv2

    img = cv2.imread(str(SAMPLE_IMAGE))
    if img is None:
        pytest.skip(f"sample image missing: {SAMPLE_IMAGE}")
    return img


@pytest.fixture(scope="session")
def engine():
    from src.depth_engine import DepthEngine

    e = DepthEngine(load_config())
    e.warmup(1)
    return e


@pytest.fixture()
def depth_map():
    rng = np.random.default_rng(0)
    return rng.random((120, 160)).astype(np.float32) * 5.0
