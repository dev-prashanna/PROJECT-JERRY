"""Shared pytest fixtures and hardware/model availability markers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKSPACE_ROOT = ROOT.parent
SAMPLE_IMAGE = WORKSPACE_ROOT / "assets" / "sample_frame.jpeg"


def model_available() -> bool:
    """True when the local Moondream checkpoint can be located."""
    try:
        from config.settings import resolve_model_path

        resolve_model_path()
    except Exception:  # noqa: BLE001
        return False
    return True


def camera_available() -> bool:
    """True when camera index 0 yields a frame right now."""
    try:
        from src.camera.camera import probe_camera

        return probe_camera(0) is not None
    except Exception:  # noqa: BLE001
        return False


requires_model = pytest.mark.skipif(
    not model_available(), reason="local Moondream checkpoint not found"
)
requires_camera = pytest.mark.skipif(
    not camera_available(), reason="no camera available (index 0)"
)
requires_image = pytest.mark.skipif(
    not SAMPLE_IMAGE.is_file(), reason=f"sample image missing: {SAMPLE_IMAGE}"
)


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    if not SAMPLE_IMAGE.is_file():
        pytest.skip(f"sample image missing: {SAMPLE_IMAGE}")
    return SAMPLE_IMAGE


@pytest.fixture(scope="session")
def sample_image(sample_image_path: Path):
    from PIL import Image

    return Image.open(sample_image_path).convert("RGB")


@pytest.fixture
def bgr_frame():
    """Small deterministic BGR frame (non-uniform, so channel order matters)."""
    import numpy as np

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 10  # blue channel
    frame[:, :, 1] = 200  # green channel
    frame[:, :, 2] = 30  # red channel
    frame[0:10, 0:10] = (255, 0, 0)  # pure blue patch in BGR
    return frame
