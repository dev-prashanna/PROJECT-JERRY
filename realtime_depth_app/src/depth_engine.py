"""Depth Anything V2 model loading and inference.

The checked checkpoint is a *relative* Depth Anything V2 model: its output is
affine-invariant inverse depth (see paper section 7.2), i.e. higher value = closer,
with an arbitrary per-image scale and shift. It is NOT a metric distance in meters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .config import AppConfig, ConfigError


@dataclass
class DepthMeta:
    model_id: str
    estimation_type: str          # "relative" | "metric"
    convention: str               # "inverse" (higher=closer) | "forward" (higher=farther)
    unit: str                     # "relative" | "m"
    precision: str
    device: str
    input_size: int


@dataclass
class DepthResult:
    frame_id: int
    t_capture: float
    t_start: float
    t_end: float
    depth: np.ndarray             # float32 (H, W) aligned with the source frame
    meta: DepthMeta
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def age_ms(self) -> float:
        return (time.time() - self.t_capture) * 1000.0


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("model.device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision in {"fp16", "autocast_fp16"}:
        return torch.float16
    if precision in {"bf16", "autocast_bf16"}:
        return torch.bfloat16
    return None


class DepthEngine:
    """Loads the checkpoint once and runs depth inference on RGB frames."""

    def __init__(self, cfg: AppConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or resolve_device(cfg.model.device)
        self._use_autocast = cfg.model.precision.startswith("autocast")
        self._dtype = _autocast_dtype(cfg.model.precision)
        if self._dtype is not None and not self._use_autocast:
            # direct half/bfloat16 weights: only reliable on CUDA
            if self.device.type != "cuda":
                raise ConfigError(f"model.precision={cfg.model.precision} requires CUDA; use fp32 on CPU")

        source = cfg.model.local_dir or cfg.model.repo_id
        if cfg.model.local_dir is not None and not Path(cfg.model.local_dir).exists():
            raise ConfigError(f"Checkpoint directory does not exist: {cfg.model.local_dir}")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                source,
                local_files_only=cfg.model.local_files_only,
                size={"height": cfg.model.input_size, "width": cfg.model.input_size},
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                source,
                local_files_only=cfg.model.local_files_only,
            )
        except Exception as exc:  # transformers raises many exception types
            hint = (
                "Checkpoint could not be loaded. If it has never been downloaded, either allow a "
                "one-time download (model.local_files_only=false) or set model.local_dir to a "
                f"local copy of {source}."
            )
            raise ConfigError(f"Failed to load depth checkpoint '{source}': {exc}\n{hint}") from exc

        self.model.to(self.device)
        if self._dtype is not None and not self._use_autocast:
            self.model = self.model.half()
        self.model.eval()

        self.estimation_type = self._resolve_estimation_type()
        self.meta = DepthMeta(
            model_id=source,
            estimation_type=self.estimation_type,
            convention="forward" if self.estimation_type == "metric" else "inverse",
            unit="m" if self.estimation_type == "metric" else "relative",
            precision=cfg.model.precision,
            device=str(self.device),
            input_size=cfg.model.input_size,
        )
        self._warm(cfg.model.warmup_iters)

    def _resolve_estimation_type(self) -> str:
        requested = self.cfg.depth.estimation_type
        if requested != "auto":
            return requested
        return getattr(self.model.config, "depth_estimation_type", "relative")

    @property
    def input_size(self) -> int:
        return self.cfg.model.input_size

    @torch.inference_mode()
    def _forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._use_autocast and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self._dtype or torch.float16):
                out = self.model(pixel_values=pixel_values)
        else:
            out = self.model(pixel_values=pixel_values)
        return out.predicted_depth

    def _to_device(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        if self._dtype is not None and not self._use_autocast:
            pixel_values = pixel_values.half()
        return pixel_values

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        return self._to_device(inputs["pixel_values"])

    def _sync(self) -> None:
        if self.cfg.model.sync_cuda_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def infer(
        self,
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        t_capture: float | None = None,
    ) -> DepthResult:
        t_capture = time.time() if t_capture is None else t_capture
        self._sync()
        t0 = time.time()
        pixel_values = self.preprocess(frame_bgr)
        self._sync()
        t1 = time.time()
        predicted = self._forward(pixel_values)
        self._sync()
        t2 = time.time()
        h, w = frame_bgr.shape[:2]
        depth = F.interpolate(
            predicted.float().unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        depth_np = depth.squeeze(0).detach().to("cpu").numpy().astype(np.float32, copy=False)
        t3 = time.time()
        return DepthResult(
            frame_id=frame_id,
            t_capture=t_capture,
            t_start=t0,
            t_end=t3,
            depth=depth_np,
            meta=self.meta,
            inference_ms=(t2 - t1) * 1000.0,
            preprocess_ms=(t1 - t0) * 1000.0,
            postprocess_ms=(t3 - t2) * 1000.0,
        )

    def warmup(self, iters: int | None = None) -> None:
        self._warm(self.cfg.model.warmup_iters if iters is None else iters)

    def _warm(self, iters: int) -> None:
        if iters <= 0:
            return
        dummy = np.zeros((self.cfg.source.height, self.cfg.source.width, 3), dtype=np.uint8)
        for _ in range(iters):
            self.infer(dummy)

    def gpu_memory_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
