"""Image preprocessing: OpenCV BGR frames -> model-ready PIL RGB images.

The pipeline is deliberately conservative.  Moondream's vision encoder crops
large images internally, and aggressive downscaling destroys exactly the
detail VQA depends on (small objects, sign text, subtle colour differences).
Resizing is therefore *opt-in* through ``InferenceConfig.encode_max_side``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "InvalidFrameError",
    "validate_frame",
    "bgr_to_rgb",
    "resize_max_side",
    "frame_to_pil",
    "describe_frame",
]


class InvalidFrameError(ValueError):
    """Raised when a frame is missing, empty, or structurally unusable."""


def validate_frame(frame: object) -> np.ndarray:
    """Validate an OpenCV frame and return it unchanged.

    Checks performed: not ``None``, ``ndarray``, 3-D, 3 channels, non-zero
    area, and a convertible dtype.
    """
    if frame is None:
        raise InvalidFrameError("frame is None (camera produced no image)")
    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError(f"frame must be a numpy array, got {type(frame)!r}")
    if frame.ndim != 3:
        raise InvalidFrameError(f"frame must have 3 dimensions (H, W, 3), got {frame.ndim}")
    if frame.shape[2] != 3:
        raise InvalidFrameError(f"frame must have 3 colour channels, got {frame.shape[2]}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise InvalidFrameError(f"frame is degenerate: shape {frame.shape}")
    if frame.size == 0:
        raise InvalidFrameError("frame is empty")
    if frame.dtype != np.uint8:
        arr = frame.astype(np.float32)
        if arr.max() <= 1.0 and arr.min() >= 0.0:  # normalised float image
            arr = arr * 255.0
        frame = np.clip(arr, 0, 255).astype(np.uint8)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB without copying when possible."""
    frame = validate_frame(frame)
    return frame[:, :, ::-1]


def resize_max_side(
    image: Image.Image,
    max_side: Optional[int],
    *,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """Downscale so the longest side is at most ``max_side``.

    Upscaling never happens; ``max_side=None`` or ``<=0`` returns the input.
    """
    if not max_side or max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size: Tuple[int, int] = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, resample=resample)


def frame_to_pil(
    frame: np.ndarray,
    *,
    max_side: Optional[int] = None,
    copy: bool = True,
) -> Image.Image:
    """Convert a BGR ``uint8`` frame into an RGB :class:`PIL.Image.Image`.

    Parameters
    ----------
    frame:
        OpenCV-style BGR frame.
    max_side:
        Optional longest-side cap (see :func:`resize_max_side`).
    copy:
        Copy the pixel buffer first.  ``True`` guarantees the PIL image owns
        its data even if the source buffer is recycled by OpenCV.
    """
    rgb = bgr_to_rgb(frame)
    if copy:
        rgb = np.ascontiguousarray(rgb)
    image = Image.fromarray(rgb, mode="RGB")
    if max_side:
        image = resize_max_side(image, max_side)
    return image


def describe_frame(frame: Optional[np.ndarray]) -> str:
    """Short diagnostics string for logs and error messages."""
    if frame is None:
        return "None"
    if not isinstance(frame, np.ndarray):
        return type(frame).__name__
    return f"shape={frame.shape} dtype={frame.dtype} bytes={frame.nbytes}"
