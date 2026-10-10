"""Moondream 2 vision-language engine (local weights, loaded once).

Responsibilities
----------------
* Locate the already-installed checkpoint — never download it.
* Load the model a single time (thread-safe, idempotent).
* Detect the API surface of the installed build (``query``/``encode_image``
  for current Moondream builds, ``answer_question`` for older ones).
* Cache the encoded image of the current frame so follow-up questions about
  the same snapshot skip the vision encoder.
* Produce a structured :class:`VisualEvidence` record with *measured*
  latency — no invented confidence scores.
* Serialise GPU access with an internal lock (one shared KV cache).

Failure modes handled explicitly: missing weights, CUDA OOM, unsupported
API, empty answers, and timeouts enforced by the caller.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from src.orchestrator.schemas import VisualEvidence, new_id
from src.perception.frame_buffer import Frame
from src.perception.preprocessing import (
    frame_to_pil,
    maybe_resize,
    sanitize_perception_text,
    to_pil,
)
from src.utils.exceptions import GPUOutOfMemoryError, InferenceError, ModelNotFoundError
from src.utils.logger import get_logger

log = get_logger("moondream")

DEFAULT_GROUNDING_SUFFIX = (
    " Answer only from what is visible in the image. If you cannot tell, say so."
)


@dataclass
class EngineInfo:
    model_path: str
    device: str
    dtype: str
    api: str
    model_class: str
    load_seconds: float
    parameter_count: int = 0
    gpu_memory_mb: Optional[float] = None

    def describe(self) -> str:
        params = f"{self.parameter_count / 1e9:.2f}B params" if self.parameter_count else "unknown size"
        return f"{self.model_class} · {params} · {self.dtype} · {self.device}"


class MoondreamEngine:
    def __init__(self, settings=None):
        from config.settings import VLMSettings, get_settings

        self.settings = settings or get_settings().vlm
        self._model: Any = None
        self._info: Optional[EngineInfo] = None
        self._api = ""
        self._device = "cpu"
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cached_key: Optional[tuple] = None
        self._cached_encoded: Any = None
        self._last_error: Optional[str] = None
        self._questions_answered = 0

    # -- introspection ---------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def questions_answered(self) -> int:
        return self._questions_answered

    def info(self) -> Optional[EngineInfo]:
        return self._info

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "device": self._device,
            "api": self._api,
            "model_path": self.settings.model_path,
            "questions_answered": self._questions_answered,
            "last_error": self._last_error,
            "describe": self._info.describe() if self._info else "not loaded",
        }

    # -- loading ---------------------------------------------------------
    def load(self, force: bool = False) -> EngineInfo:
        with self._load_lock:
            if self._info is not None and not force:
                return self._info
            return self._load_locked()

    def _load_locked(self) -> EngineInfo:
        started = time.perf_counter()
        path = Path(self.settings.model_path)
        if not path.is_dir():
            raise ModelNotFoundError(
                f"Moondream weights not found at {path}",
                hint="Set VLM_MODEL_PATH to the folder containing model.safetensors.",
            )
        if not (path / "model.safetensors").exists() and not (path / "pytorch_model.bin").exists():
            raise ModelNotFoundError(
                f"{path} does not contain model.safetensors",
                hint="The checkpoint looks incomplete.",
            )

        # Offline by default: the weights and tokenizer must come from disk.
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.utils.device_manager import resolve_torch_device

        self._device = resolve_torch_device(self.settings.device)
        dtype = self._resolve_dtype(torch)
        log.info("loading Moondream from %s (device=%s dtype=%s)", path, self._device, dtype)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(path), trust_remote_code=True, torch_dtype=dtype
            )
            tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
            model = model.to(self._device)
            if hasattr(model, "eval"):
                model.eval()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise GPUOutOfMemoryError("CUDA out of memory while loading Moondream", cause=exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"failed to load Moondream: {exc}", cause=exc) from exc

        self._model = model
        self._tokenizer = tokenizer
        self._api = self._detect_api(model)

        params = sum(p.numel() for p in model.parameters())
        gpu_mb = None
        if self._device == "cuda" and torch.cuda.is_available():
            gpu_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

        info = EngineInfo(
            model_path=str(path),
            device=self._device,
            dtype=str(dtype).replace("torch.", ""),
            api=self._api,
            model_class=type(model).__name__,
            load_seconds=time.perf_counter() - started,
            parameter_count=params,
            gpu_memory_mb=gpu_mb,
        )
        self._info = info
        log.info("Moondream ready in %.1fs — %s", info.load_seconds, info.describe())
        return info

    def _resolve_dtype(self, torch):
        pref = (self.settings.dtype or "auto").lower()
        if self._device == "cpu":
            return torch.float32 if pref in {"auto", "", "fp32"} else getattr(torch, pref, torch.float32)
        if pref in {"auto", ""}:
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return mapping.get(pref, torch.bfloat16)

    @staticmethod
    def _detect_api(model: Any) -> str:
        has_encode = hasattr(model, "encode_image")
        if has_encode and hasattr(model, "query"):
            return "hf_query"
        if has_encode and hasattr(model, "answer_question"):
            return "legacy"
        if hasattr(model, "query"):
            return "query_no_encode"
        raise InferenceError(
            f"unsupported Moondream API on {type(model).__name__}",
            hint="Expected encode_image()/query() or encode_image()/answer_question().",
        )

    # -- image encoding --------------------------------------------------
    def encode_image(self, image: Any, cache_key: Optional[tuple] = None) -> tuple[Any, float, bool]:
        """Return (encoding, seconds, reused)."""
        if cache_key is not None:
            with self._cache_lock:
                if self._cached_key == cache_key and self._cached_encoded is not None:
                    return self._cached_encoded, 0.0, True
        t0 = time.perf_counter()
        with self._infer_lock:
            if not self.is_loaded:
                self.load()
            if self._api in {"hf_query", "legacy"}:
                encoded = self._model.encode_image(image)
            else:
                encoded = image
        elapsed = time.perf_counter() - t0
        if cache_key is not None:
            with self._cache_lock:
                self._cached_key = cache_key
                self._cached_encoded = encoded
        return encoded, elapsed, False

    def invalidate_cache(self) -> None:
        with self._cache_lock:
            self._cached_key = None
            self._cached_encoded = None
        if self._device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass

    # -- inference -------------------------------------------------------
    def ask(
        self,
        image: Union[Frame, Any],
        question: str,
        *,
        frame_id: Optional[str] = None,
        ground: bool = True,
        max_new_tokens: Optional[int] = None,
    ) -> VisualEvidence:
        """Answer a visual question and return a structured evidence record."""
        question = (question or "").strip()
        frame_ts = None
        image_age = None
        src_frame_id = frame_id or new_id("frame")

        if isinstance(image, Frame):
            frame_ts = image.timestamp
            image_age = image.age_s
            src_frame_id = frame_id or image.tag
            pil = frame_to_pil(image.image)
        else:
            pil = to_pil(image)

        pil = maybe_resize(pil, 0)
        prompt = question + (DEFAULT_GROUNDING_SUFFIX if ground else "")

        if not question:
            return VisualEvidence(
                frame_id=src_frame_id,
                question=question,
                raw_answer="",
                status="failed",
                error="empty visual question",
                frame_timestamp=frame_ts,
                image_age_s=image_age,
            )

        # Load outside the timed region: model loading is a one-off cost and
        # must not be reported as per-request inference latency.
        self.load()
        started = time.perf_counter()
        try:
            cache_key = (src_frame_id, id(pil) if not isinstance(image, Frame) else src_frame_id)
            # Prefer caching on the frame id when we have a real frame.
            if isinstance(image, Frame):
                cache_key = ("frame", image.frame_id, pil.size)
            encoded, encode_s, reused = self.encode_image(pil, cache_key=cache_key)
            gen_s, raw = self._generate(encoded, prompt, max_new_tokens)
        except GPUOutOfMemoryError as exc:
            self._last_error = str(exc)
            self.invalidate_cache()
            return VisualEvidence(
                frame_id=src_frame_id,
                question=question,
                raw_answer="",
                status="failed",
                error=f"GPU out of memory: {exc}",
                frame_timestamp=frame_ts,
                image_age_s=image_age,
                inference_time_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            log.exception("Moondream inference failed")
            return VisualEvidence(
                frame_id=src_frame_id,
                question=question,
                raw_answer="",
                status="failed",
                error=str(exc),
                frame_timestamp=frame_ts,
                image_age_s=image_age,
                inference_time_ms=(time.perf_counter() - started) * 1000,
            )

        total_ms = (time.perf_counter() - started) * 1000
        self._questions_answered += 1

        san = sanitize_perception_text(str(raw or ""))
        notes: list[str] = []
        if san.truncated:
            notes.append("answer truncated to fit the evidence budget")
        if san.injection_suspected:
            notes.append("perception output contained instruction-like text; ignored as data")
        if not san.text.strip():
            notes.append("model returned an empty answer")

        return VisualEvidence(
            source="moondream2",
            frame_id=src_frame_id,
            question=question,
            raw_answer=san.text,
            status="completed" if san.text.strip() else "failed",
            uncertainty_notes=notes,
            inference_time_ms=round(total_ms, 1),
            frame_timestamp=frame_ts,
            image_age_s=None if image_age is None else round(image_age, 3),
            error=None if san.text.strip() else "empty answer",
        )

    def _generate(self, encoded: Any, prompt: str, max_new_tokens: Optional[int]) -> tuple[float, str]:
        import torch

        tokens = max_new_tokens or self.settings.max_new_tokens
        temperature = self.settings.temperature
        t0 = time.perf_counter()
        sampling: dict[str, Any] = {"max_tokens": tokens}
        if temperature is not None:
            sampling["temperature"] = float(temperature)

        with self._infer_lock:
            if not self.is_loaded:
                self.load()
            try:
                if self._api in {"hf_query", "query_no_encode"}:
                    out = self._model.query(encoded, prompt, settings=sampling)
                    answer = out.get("answer", "") if isinstance(out, dict) else str(out)
                elif self._api == "legacy":
                    raw = self._model.answer_question(
                        encoded, prompt, self._tokenizer,
                        max_new_tokens=tokens,
                    )
                    answer = raw.get("answer", str(raw)) if isinstance(raw, dict) else str(raw)
                else:  # pragma: no cover - defensive
                    out = self._model.query(encoded, prompt, settings=sampling)
                    answer = out.get("answer", "") if isinstance(out, dict) else str(out)
                if self._device == "cuda":
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise GPUOutOfMemoryError("CUDA out of memory during Moondream generation", cause=exc) from exc
        return time.perf_counter() - t0, str(answer)

    def caption(self, image: Union[Frame, Any], length: str = "normal") -> VisualEvidence:
        """Scene caption convenience wrapper (uses model.caption when present)."""
        if isinstance(image, Frame):
            pil = frame_to_pil(image.image)
            frame_id, ts, age = image.tag, image.timestamp, image.age_s
        else:
            pil = to_pil(image)
            frame_id, ts, age = new_id("frame"), None, None

        if self.is_loaded and hasattr(self._model, "caption"):
            started = time.perf_counter()
            try:
                with self._infer_lock:
                    out = self._model.caption(pil, length=length)
                text = out.get("caption", "") if isinstance(out, dict) else str(out)
                ms = (time.perf_counter() - started) * 1000
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                return VisualEvidence(
                    frame_id=frame_id, question="Describe the scene.",
                    raw_answer="", status="failed", error=str(exc),
                    frame_timestamp=ts, image_age_s=age,
                )
            san = sanitize_perception_text(str(text))
            return VisualEvidence(
                source="moondream2",
                frame_id=frame_id,
                question="Describe the scene.",
                raw_answer=san.text,
                status="completed" if san.text else "failed",
                inference_time_ms=round(ms, 1),
                frame_timestamp=ts,
                image_age_s=None if age is None else round(age, 3),
            )
        return self.ask(image, "Describe everything you can see in this image.")

    # -- memory ----------------------------------------------------------
    def unload(self) -> None:
        with self._load_lock:
            if self._model is None:
                return
            self._model = None
            self._tokenizer = None
            self._info = None
            self.invalidate_cache()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass
            log.info("Moondream unloaded")


_ENGINES: dict[str, MoondreamEngine] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(settings=None) -> MoondreamEngine:
    """Process-wide singleton so the model is loaded at most once."""
    from config.settings import VLMSettings

    settings = settings or VLMSettings()
    key = str(settings.model_path) + settings.device + settings.dtype
    with _ENGINES_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = MoondreamEngine(settings)
            _ENGINES[key] = engine
        return engine


def reset_engine() -> None:
    with _ENGINES_LOCK:
        for engine in _ENGINES.values():
            engine.unload()
        _ENGINES.clear()
