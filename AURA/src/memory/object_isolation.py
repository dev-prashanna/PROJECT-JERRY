"""Object isolation: grounding, segmentation and ROI extraction.

Modular backends:

* Grounding – Moondream text-grounding (best effort), or a centre-prior
  fallback that returns the full frame when nothing better is available.
* Segmentation – bounding-box crop fallback (labelled as such).  SAM-class
  mask models are optional and not loaded unless already installed.

A bounding-box crop is an explicitly labelled fallback, never silently
presented as a segmentation mask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from src.memory.exceptions import ObjectNotLocated, ObjectSegmentationUnavailable
from src.utils.logger import get_logger

log = get_logger("memory.isolation")


@dataclass
class GroundedRegion:
    bbox_xyxy: tuple[int, int, int, int]
    label: str
    score: Optional[float] = None
    source: str = "fallback"
    mask: Optional[np.ndarray] = None


@runtime_checkable
class ObjectGrounder(Protocol):
    name: str

    def locate(self, image_bgr: np.ndarray, text_query: str) -> list[GroundedRegion]: ...


@runtime_checkable
class Segmenter(Protocol):
    name: str

    def segment(self, image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray: ...


_BBOX_RE = re.compile(
    r"\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?"
)


class MoondreamBoxGrounder:
    """Best-effort text grounding through the existing VLM.

    Moondream is asked for a bounding box; the reply is parsed for four
    numbers.  Coordinates are interpreted as ``[x1, y1, x2, y2]`` in pixels
    when they exceed 1.5, otherwise as normalised ``[0,1]`` fractions.
    """

    name = "moondream-box"

    def __init__(self, vlm_engine=None):
        self._vlm = vlm_engine

    def _engine(self):
        if self._vlm is None:
            from src.perception.vlm.registry import create_vlm

            self._vlm = create_vlm()
        return self._vlm

    def locate(self, image_bgr: np.ndarray, text_query: str) -> list[GroundedRegion]:
        query = (text_query or "the main object").strip()
        prompt = (
            f"Locate the {query} in this image. "
            "Reply with only a bounding box as [x1, y1, x2, y2] in pixels."
        )
        try:
            evidence = self._engine().ask(image_bgr, prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning("moondream grounding failed: %s", exc)
            return []
        if not evidence.usable:
            return []
        m = _BBOX_RE.search(evidence.raw_answer)
        if not m:
            return []
        h, w = image_bgr.shape[:2]
        vals = [float(v) for v in m.groups()]
        if max(vals) <= 1.5:
            x1, y1, x2, y2 = vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h
        else:
            x1, y1, x2, y2 = vals
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(x1 + 1, min(x2, w)))
        y2 = int(max(y1 + 1, min(y2, h)))
        if (x2 - x1) * (y2 - y1) < 100:
            return []
        return [
            GroundedRegion(
                bbox_xyxy=(x1, y1, x2, y2),
                label=query,
                score=None,
                source="moondream",
            )
        ]


class CenterPriorGrounder:
    """Fallback grounder: largest high-variance region near the centre."""

    name = "center-prior"

    def locate(self, image_bgr: np.ndarray, text_query: str) -> list[GroundedRegion]:
        import cv2

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        cx, cy = w / 2.0, h / 2.0
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < (w * h) * 0.01 or area > (w * h) * 0.9:
                continue
            dist = abs((x + bw / 2) - cx) / w + abs((y + bh / 2) - cy) / h
            score = area / (w * h) - 0.25 * dist
            if score > best_score:
                best_score = score
                best = (x, y, x + bw, y + bh)
        if best is None:
            best = (0, 0, w, h)
            source = "full-frame"
        else:
            source = "center-prior"
        return [
            GroundedRegion(
                bbox_xyxy=tuple(int(v) for v in best),
                label=(text_query or "object").strip(),
                score=float(best_score) if source == "center-prior" else None,
                source=source,
            )
        ]


class BoundingBoxFallbackSegmenter:
    """Explicit fallback: crop the bounding box (no learned mask)."""

    name = "bbox-crop"

    def segment(self, image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        h, w = image_bgr.shape[:2]
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(x1 + 1, min(x2, w)))
        y2 = int(max(y1 + 1, min(y2, h)))
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            raise ObjectSegmentationUnavailable("bounding-box crop is empty")
        return np.ascontiguousarray(crop)


def crop_with_margin(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    margin: float = 0.08,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image_bgr.shape[:2]
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ObjectNotLocated("cropped region is empty")
    return np.ascontiguousarray(crop)


def create_grounder(name: str = "auto", vlm_engine=None) -> ObjectGrounder:
    name = (name or "auto").strip().lower()
    if name in {"moondream", "moondream-box"}:
        return MoondreamBoxGrounder(vlm_engine)
    if name in {"center", "center-prior", "fallback"}:
        return CenterPriorGrounder()
    # auto: try moondream, callers may fall back on empty results
    return MoondreamBoxGrounder(vlm_engine)


def create_segmenter(name: str = "auto") -> Segmenter:
    name = (name or "auto").strip().lower()
    if name in {"bbox", "bbox-crop", "auto", "fallback"}:
        return BoundingBoxFallbackSegmenter()
    log.warning("segmentation backend %r unavailable; using bbox-crop", name)
    return BoundingBoxFallbackSegmenter()
