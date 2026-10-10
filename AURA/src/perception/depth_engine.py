"""Depth Anything V2 inference for AURA.

Compact AURA-native runner (mirrors the sibling ``realtime_depth_app`` but
imported as ``src.perception.depth`` so the two ``src`` packages never
collide).  Loads the HF-cached checkpoint once, runs BGR -> float32 depth,
and colourizes for the GUI preview.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from config.settings import DepthSettings, get_settings
from src.utils.exceptions import InferenceError, ModelNotFoundError
from src.utils.logger import get_logger

log = get_logger("depth.engine")

#: GUI colour maps (BGR out).  Inverse-depth convention: warm = near.
COLORMAPS: dict[str, int] = {
    "TURBO": cv2.COLORMAP_TURBO,
    "JET": cv2.COLORMAP_JET,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "CIVIDIS": cv2.COLORMAP_CIVIDIS,
    "BONE": cv2.COLORMAP_BONE,
    "PARULA": cv2.COLORMAP_PARULA,
}


@dataclass
class DepthFrame:
    """One aligned (colourized, frame-tagged) depth prediction."""

    frame_id: int
    timestamp: float
    depth: np.ndarray  # float32 (H, W), affine-invariant inverse depth
    viz: np.ndarray  # uint8 BGR (H, W, 3) colour map for display
    inference_ms: float
    convention: str = "inverse"  # higher = closer
    unit: str = "relative"

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.timestamp)


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise InferenceError("DEPTH_DEVICE=cuda requested but CUDA unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def colorize_depth(depth: np.ndarray, colormap: str = "TURBO") -> np.ndarray:
    """float32 relative depth -> uint8 BGR colour map (2-98 percentile range)."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((np.nan_to_num(depth, nan=lo, posinf=hi, neginf=lo) - lo) / max(1e-9, hi - lo) * 255.0, 0, 255).astype(np.uint8)
    cmap = COLORMAPS.get(colormap.upper(), cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(norm, cmap)


class DepthEngine:
    """Loads the checkpoint once and answers ``infer(bgr_frame)``."""

    def __init__(self, settings: Optional[DepthSettings] = None):
        self.settings = settings or get_settings().depth
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._warm_done = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def load(self, force: bool = False) -> None:
        with self._load_lock:
            if self._model is not None and not force:
                return
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self._device = _resolve_device(self.settings.device)
            try:
                self._processor = AutoImageProcessor.from_pretrained(
                    self.settings.model_id, local_files_only=True
                )
                self._model = AutoModelForDepthEstimation.from_pretrained(
                    self.settings.model_id, local_files_only=True
                )
            except Exception as exc:  # noqa: BLE001 - transformers raises many types
                raise ModelNotFoundError(
                    f"depth checkpoint '{self.settings.model_id}' not loadable: {exc}",
                    hint="Ensure the HF cache holds the model, or set DEPTH_MODEL.",
                ) from exc
            self._model.to(self._device)
            if self.settings.precision in {"fp16", "bf16"} and self._device == "cuda":
                self._model = self._model.half() if self.settings.precision == "fp16" else self._model.to(torch.bfloat16)
            self._model.eval()
            self._warm()
            log.info(
                "depth ready — %s · %s · %s",
                self.settings.model_id.split("/")[-1],
                self._device,
                self.settings.precision,
            )

    def _warm(self) -> None:
        if self._warm_done or self.settings.warmup_iters <= 0:
            self._warm_done = True
            return
        dummy = np.zeros((self.settings.input_size, self.settings.input_size, 3), dtype=np.uint8)
        for _ in range(max(1, self.settings.warmup_iters)):
            try:
                self.infer(dummy)
            except Exception:  # noqa: BLE001
                break
        self._warm_done = True

    def infer(self, frame_bgr: np.ndarray, frame_id: int = 0, timestamp: Optional[float] = None) -> DepthFrame:
        if self._model is None:
            self.load()
        import torch
        import torch.nn.functional as F

        ts = time.time() if timestamp is None else timestamp
        with self._infer_lock:
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = self._processor(images=rgb, return_tensors="pt")
            pixel = inputs["pixel_values"].to(self._device)
            if self.settings.precision == "fp16" and self._device == "cuda":
                pixel = pixel.half()
            elif self.settings.precision == "bf16" and self._device == "cuda":
                pixel = pixel.to(torch.bfloat16)
            use_amp = self.settings.precision.startswith("autocast") and self._device == "cuda"
            with torch.inference_mode():
                if use_amp:
                    dtype = torch.bfloat16 if self.settings.precision.endswith("bf16") else torch.float16
                    with torch.autocast("cuda", dtype=dtype):
                        out = self._model(pixel)
                else:
                    out = self._model(pixel)
            t1 = time.perf_counter()
            predicted = out.predicted_depth.float()
            h, w = frame_bgr.shape[:2]
            depth = F.interpolate(
                predicted.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze(1)
            depth_np = depth.squeeze(0).detach().to("cpu").numpy().astype(np.float32, copy=False)
            inference_ms = (time.perf_counter() - t1) * 1000.0
        viz = colorize_depth(depth_np, self.settings.colormap)
        return DepthFrame(
            frame_id=frame_id,
            timestamp=ts,
            depth=depth_np,
            viz=viz,
            inference_ms=inference_ms,
            convention="inverse",
            unit="relative",
        )

    def unload(self) -> None:
        with self._load_lock:
            self._model = None
            self._processor = None
            self._warm_done = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
