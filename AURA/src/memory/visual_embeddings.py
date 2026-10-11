"""Visual embedding backends for object memory.

* ``clip-vit-b32`` – CLIP image encoder via transformers (used only when the
  weights are already cached locally or download is explicitly allowed).
* ``sift-color-v1`` – deterministic OpenCV SIFT + HSV-histogram descriptor.
  Always available; fixed 512-D; used as the offline baseline.

All embeddings are L2-normalized float32 vectors; cosine similarity is the
matching metric.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from src.memory.exceptions import InvalidEmbedding
from src.memory.vector_index import l2_normalize
from src.utils.logger import get_logger

log = get_logger("memory.visual")


@runtime_checkable
class VisualEmbedder(Protocol):
    model_id: str
    embedding_dimension: int

    def encode(self, image_bgr: np.ndarray) -> np.ndarray: ...


class SiftColorEmbedder:
    """Content-cropped spatial colour signature + hue histogram (512-D).

    The crop is first reduced to its high-variance content region so a small
    object on a large uniform background does not wash out the descriptor.
    """

    model_id = "sift-color-v1"
    embedding_dimension = 512

    def __init__(self, grid: int = 12, hist_bins: int = 80):
        self._grid = int(grid)
        self._hist_bins = int(hist_bins)

    def _content_crop(self, img: np.ndarray) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # High-variance content: edge map + contours
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area > best_area and area >= (h * w) * 0.005:
                best_area = area
                best = (x, y, x + bw, y + bh)
        if best is None:
            return img
        x1, y1, x2, y2 = best
        pad_x = max(1, (x2 - x1) // 8)
        pad_y = max(1, (y2 - y1) // 8)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        crop = img[y1:y2, x1:x2]
        return crop if crop.size else img

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            raise InvalidEmbedding("empty image crop")
        img = image_bgr
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        content = self._content_crop(img)

        g = self._grid
        small = cv2.resize(content, (g, g), interpolation=cv2.INTER_AREA)
        spatial = small.reshape(-1).astype(np.float32) / 255.0  # g*g*3

        hsv = cv2.cvtColor(content, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [self._hist_bins], [0, 180])
        hist = hist.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(hist))
        if norm > 1e-12:
            hist = hist / norm

        pad = self.embedding_dimension - spatial.size - hist.size
        if pad > 0:
            vec = np.concatenate([spatial, hist, np.zeros(pad, dtype=np.float32)])
        else:
            vec = np.concatenate([spatial, hist])[: self.embedding_dimension]
        return l2_normalize(vec)


# Backwards-compatible alias (older config values said "hog-color")
HogColorEmbedder = SiftColorEmbedder


class ClipViTEmbedder:
    """OpenAI CLIP ViT-B/32 through HuggingFace transformers (lazy load)."""

    model_id = "clip-vit-b32"
    embedding_dimension = 512

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        log.info("CLIP embedder loaded on %s", self.device)

    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("CLIP unavailable: %s", exc)
            return False

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        from PIL import Image

        if image_bgr is None or image_bgr.size == 0:
            raise InvalidEmbedding("empty image crop")
        self._load()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        vec = feats[0].detach().cpu().numpy().astype(np.float32)
        return l2_normalize(vec)


def clip_weights_cached(model_name: str = "openai/clip-vit-base-patch32") -> bool:
    try:
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        return model is not None
    except Exception:  # noqa: BLE001
        return False


def create_embedder(
    name: str = "auto", *, allow_download: bool = False, device: str = "cpu"
) -> VisualEmbedder:
    """Resolve the object embedder; never downloads large weights silently."""
    name = (name or "auto").strip().lower()
    if name in {"clip-vit-b32", "clip", "clip-vit-base-patch32"}:
        if clip_weights_cached() or allow_download:
            try:
                return ClipViTEmbedder(device=device)
            except Exception as exc:  # noqa: BLE001
                log.warning("CLIP init failed, falling back to sift-color: %s", exc)
        else:
            log.info("CLIP weights not cached and download disabled; using sift-color-v1")
    elif name not in {"sift-color-v1", "sift", "hog-color-v1", "hog", "auto", "fallback"}:
        log.warning("unknown object embedder %r; using sift-color-v1", name)
    return SiftColorEmbedder()
