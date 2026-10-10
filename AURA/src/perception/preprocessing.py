"""Image preparation and *untrusted-output* sanitation.

Two jobs:

1. Turn camera frames into the PIL images Moondream expects, optionally
   downscaling first (cheap, bounded).
2. Treat everything a perception model emits as **untrusted data**: strip
   control characters, cap the length, and flag text that looks like an
   attempt to issue instructions (prompt injection seen on a poster/screen).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from src.utils.logger import get_logger

log = get_logger("preprocess")

MAX_EVIDENCE_CHARS = 1200

_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:\w+\s+){0,4}(instructions?|prompts?|rules?)\b", re.I),
    re.compile(r"\bnew instructions?\b", re.I),
    re.compile(r"\bact as\b", re.I),
    re.compile(r"\byou are (now|actually)\b", re.I),
    re.compile(r"\b(system|developer) prompt\b", re.I),
    re.compile(r"\bdisregard (the|your|all)\b", re.I),
    re.compile(r"\bcall (this )?tool\b", re.I),
    re.compile(r"\bexecute\b.*\b(command|script)\b", re.I),
    re.compile(r"<\|.*?\|>"),
    re.compile(r"\b(rm -rf|sudo|bash -c|python -c)\b", re.I),
]


@dataclass
class SanitizedEvidence:
    text: str
    truncated: bool = False
    injection_suspected: bool = False
    removed_chars: int = 0


def to_pil(image: Union[np.ndarray, "Any"]) -> "Any":
    """ndarray (BGR or RGB) / path / PIL -> PIL RGB image."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str,)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        # OpenCV delivers BGR; accept both via a keyword.
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr[..., ::-1] if _looks_bgr(arr) else arr).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image).__name__}")


def _looks_bgr(arr: np.ndarray) -> bool:
    """Heuristic: webcam frames from OpenCV are BGR.

    We assume BGR for uint8 HxWx3 arrays coming from the camera path; the
    caller can force RGB with ``to_pil_rgb``.
    """
    return arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3


def frame_to_pil(frame_image: np.ndarray) -> "Any":
    """Camera frame (BGR, OpenCV order) -> PIL RGB."""
    from PIL import Image

    rgb = np.ascontiguousarray(frame_image[..., ::-1])
    return Image.fromarray(rgb).convert("RGB")


def maybe_resize(image: "Any", max_side: int = 0) -> "Any":
    from PIL import Image

    if not max_side or max_side <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def sanitize_perception_text(text: str, max_chars: int = MAX_EVIDENCE_CHARS) -> SanitizedEvidence:
    """Normalise untrusted VLM output before it reaches the LLM.

    Flags (but does not silently delete) anything that looks like an attempt
    to smuggle instructions through the camera feed.
    """
    raw = text or ""
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", raw)
    cleaned = cleaned.replace("\r", " ").strip()
    injection = any(p.search(cleaned) for p in _INJECTION_PATTERNS)
    truncated = False
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
        truncated = True
    if injection:
        log.warning(
            "perception output contains instruction-like text; treating as data only "
            "(%d chars)",
            len(cleaned),
        )
    return SanitizedEvidence(
        text=cleaned,
        truncated=truncated,
        injection_suspected=injection,
        removed_chars=len(raw) - len(cleaned),
    )


def describe_image_briefly(image: "Any") -> str:
    """Cheap, dependency-free image description used for diagnostics."""
    w, h = image.size
    return f"{w}x{h} RGB"
