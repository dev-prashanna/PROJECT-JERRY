"""Speech-to-text via faster-whisper (CTranslate2).

* Loads the model once and reuses it for every utterance.
* Prefers the local cache; downloads only when ``STT_ALLOW_DOWNLOAD=1``.
* Rejects empty/whitespace transcripts so silence never reaches the agent.
* Reports measured latency, detected language and confidence statistics.
* Uses the built-in Silero VAD filter (when enabled) as a second line of
  defence against noise and music.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.settings import STTSettings, get_settings
from src.utils.exceptions import InferenceError, ModelNotFoundError
from src.utils.logger import get_logger

log = get_logger("whisper")


@dataclass
class TranscriptionResult:
    text: str
    language: str = ""
    language_probability: float = 0.0
    latency_s: float = 0.0
    duration_s: float = 0.0
    segments: list[dict] = field(default_factory=list)
    model: str = ""
    device: str = ""
    is_empty: bool = True
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.is_empty = not bool(self.text.strip())

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None


class WhisperEngine:
    def __init__(self, settings: Optional[STTSettings] = None):
        self.settings = settings or get_settings().stt
        self._model: Any = None
        self._info: dict[str, Any] = {}
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._transcriptions = 0

    # -- introspection ---------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def device(self) -> str:
        return str(self._info.get("device", ""))

    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model": self.settings.model,
            "device": self.device,
            "compute_type": self.settings.compute_type,
            "transcriptions": self._transcriptions,
            "last_error": self._last_error,
        }

    # -- loading ---------------------------------------------------------
    def _resolve_device(self) -> str:
        pref = (self.settings.device or "auto").lower()
        if pref in {"cuda", "gpu"}:
            return "cuda"
        if pref == "cpu":
            return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self) -> dict[str, Any]:
        with self._load_lock:
            if self._model is not None:
                return self._info
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise InferenceError(
                    "faster-whisper is not installed",
                    hint="pip install --user --break-system-packages faster-whisper",
                ) from exc

            device = self._resolve_device()
            compute = self.settings.compute_type
            if device == "cpu" and compute in {"float16", "int8_float16"}:
                compute = "int8"

            download_root = Path(self.settings.download_root)
            download_root.mkdir(parents=True, exist_ok=True)
            local = download_root / self.settings.model.replace("/", "--")
            if not local.exists() and not self.settings.allow_download:
                raise ModelNotFoundError(
                    f"whisper weights not found at {local}",
                    hint="Set STT_ALLOW_DOWNLOAD=1 or place the model there.",
                )

            started = time.perf_counter()
            if device == "cuda":
                from src.utils.device_manager import ensure_ct2_cuda_libs

                if not ensure_ct2_cuda_libs():
                    log.warning(
                        "CUDA 12 libraries for CTranslate2 not found; "
                        "falling back to CPU inference"
                    )
                    device, compute = "cpu", "int8"
            log.info(
                "loading whisper '%s' (device=%s compute=%s cache=%s)",
                self.settings.model, device, compute, download_root,
            )
            try:
                model = WhisperModel(
                    self.settings.model,
                    device=device,
                    compute_type=compute,
                    download_root=str(download_root),
                )
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                raise InferenceError(f"failed to load whisper: {exc}", cause=exc) from exc

            self._model = model
            self._info = {
                "model": self.settings.model,
                "device": device,
                "compute_type": compute,
                "download_root": str(download_root),
                "load_seconds": round(time.perf_counter() - started, 2),
                "local_path": str(local),
            }
            log.info("whisper ready in %.1fs", self._info["load_seconds"])
            return self._info

    # -- transcription ---------------------------------------------------
    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: Optional[int] = None,
        *,
        language: Optional[str] = None,
        metrics=None,
    ) -> TranscriptionResult:
        """Transcribe mono float32 audio. Empty input → empty result."""
        sr = int(sample_rate or self.settings.sample_rate)
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="", error="no audio")

        # Accept higher-rate audio and downsample for the model.
        if sr != 16000:
            audio = _resample(audio, sr, 16000)
        peak = float(np.max(np.abs(audio)))
        if peak > 0 and peak < 0.005:
            return TranscriptionResult(text="", error="audio too quiet")
        if peak > 1.0:
            audio = audio / peak

        self.load()
        started = time.perf_counter()
        language = language if language is not None else (self.settings.language or None)
        if language in ("", "auto", "none"):
            language = None

        try:
            with self._infer_lock, _stage(metrics, "whisper_transcribe"):
                segments_iter, info = self._model.transcribe(
                    audio,
                    language=language,
                    beam_size=1,
                    vad_filter=self.settings.vad_enabled,
                    vad_parameters={
                        "threshold": self.settings.vad_threshold,
                        "min_silence_duration_ms": 300,
                    },
                    condition_on_previous_text=False,
                    temperature=0.0,
                    hotwords=None,
                )
                segments = []
                parts = []
                for seg in segments_iter:
                    text = (seg.text or "").strip()
                    if text:
                        parts.append(text)
                    segments.append(
                        {
                            "start": round(seg.start, 2),
                            "end": round(seg.end, 2),
                            "text": text,
                            "no_speech_prob": round(getattr(seg, "no_speech_prob", 0.0), 3),
                            "avg_logprob": round(getattr(seg, "avg_logprob", 0.0), 3),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            log.exception("whisper transcription failed")
            return TranscriptionResult(text="", error=str(exc))

        latency = time.perf_counter() - started
        text = " ".join(parts).strip()
        self._transcriptions += 1
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            latency_s=latency,
            duration_s=audio.size / 16000.0,
            segments=segments,
            model=self.settings.model,
            device=self.device,
            is_empty=not bool(text),
        )

    def warmup(self) -> float:
        """Run one tiny transcription so the first real request is not cold."""
        started = time.perf_counter()
        try:
            self.transcribe(np.zeros(1600, dtype=np.float32), 16000)
        except Exception as exc:  # noqa: BLE001
            log.warning("whisper warmup failed: %s", exc)
        return time.perf_counter() - started

    def unload(self) -> None:
        with self._load_lock:
            self._model = None
            self._info = {}
            try:
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass
            log.info("whisper unloaded")


class _stage:
    def __init__(self, metrics, name: str):
        self.metrics = metrics
        self.name = name

    def __enter__(self):
        if self.metrics:
            self.metrics.start(self.name)
        return self

    def __exit__(self, *exc):
        if self.metrics:
            self.metrics.stop(self.name)


def _resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    from math import gcd

    if src == dst or samples.size == 0:
        return samples
    g = gcd(int(src), int(dst))
    try:
        from scipy.signal import resample_poly

        return resample_poly(samples, dst // g, src // g).astype(np.float32)
    except Exception:  # pragma: no cover - scipy is a hard dependency
        duration = samples.size / src
        n = int(duration * dst)
        x_old = np.linspace(0.0, duration, samples.size, endpoint=False)
        x_new = np.linspace(0.0, duration, n, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)


_ENGINES: dict[str, WhisperEngine] = {}
_LOCK = threading.Lock()


def get_stt_engine(settings: Optional[STTSettings] = None) -> WhisperEngine:
    settings = settings or STTSettings()
    key = f"{settings.model}:{settings.device}:{settings.compute_type}:{settings.download_root}"
    with _LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = WhisperEngine(settings)
            _ENGINES[key] = engine
        return engine


def reset_engine() -> None:
    with _LOCK:
        for e in _ENGINES.values():
            e.unload()
        _ENGINES.clear()
