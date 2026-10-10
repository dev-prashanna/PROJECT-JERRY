"""Moondream 2 inference engine.

Responsibilities
----------------
* locate and load the **already installed** checkpoint exactly once
* choose CUDA (bf16) when available, fall back to CPU (fp32)
* detect which API the installed Moondream build exposes
  (``query``/``encode_image`` for the current HF build, ``answer_question``
  for older ones) and normalise both into one application-level result dict
* cache the encoded image of the active snapshot so repeated questions about
  a frozen frame skip the vision encoder
* measure per-stage latency with correct CUDA synchronisation

All public methods are safe to call from a worker thread; concurrent GPU
calls are serialised with an internal lock because the model owns one shared
KV cache.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image

from config.settings import InferenceConfig, ModelConfig, ModelNotFoundError, resolve_model_path
from src.camera.frame_buffer import FrameSnapshot
from src.utils.logging_utils import get_logger
from src.utils.performance import StageTimer, describe_device, gpu_memory_mb
from src.vlm.preprocessing import (
    InvalidFrameError,
    frame_to_pil,
    resize_max_side,
)

log = get_logger(__name__)

__all__ = [
    "EngineInfo",
    "ModelLoadError",
    "ModelApiError",
    "MoondreamEngine",
    "successful_result",
    "failed_result",
]


class ModelLoadError(RuntimeError):
    """The checkpoint exists but could not be loaded."""


class ModelApiError(RuntimeError):
    """The loaded object does not expose a supported inference API."""


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def successful_result(**base: Any) -> Dict[str, Any]:
    """Canonical success payload (fields may be overridden by ``base``)."""
    payload: Dict[str, Any] = {
        "answer": "",
        "status": "success",
        "error": None,
        "inference_time_seconds": 0.0,
        "preprocess_time_seconds": 0.0,
        "encode_time_seconds": 0.0,
        "generate_time_seconds": 0.0,
        "encoding_reused": False,
        "device": "",
        "frame_id": None,
        "frame_timestamp": None,
        "question": "",
        "prompt_used": "",
        "grounded": False,
        "model_path": "",
        "api": "",
        "gpu_memory_mb": {},
        "timestamp": time.time(),
    }
    payload.update(base)
    return payload


def failed_result(error: str, **base: Any) -> Dict[str, Any]:
    payload = successful_result(**base)
    payload["status"] = "error"
    payload["error"] = error
    payload["answer"] = ""
    return payload


# ---------------------------------------------------------------------------
# Engine info
# ---------------------------------------------------------------------------

@dataclass
class EngineInfo:
    model_path: str
    device: str
    dtype: str
    api: str
    model_class: str
    torch_version: str
    transformers_version: str
    load_seconds: float
    parameter_count: int
    gpu_name: str = ""
    gpu_memory_total_gb: float = 0.0

    def describe(self) -> str:
        params = self.parameter_count / 1e9 if self.parameter_count else 0
        return (
            f"{self.model_class} · {params:.2f}B params · {self.dtype} · "
            f"{self.device}"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MoondreamEngine:
    """Loads Moondream once and answers image-grounded questions."""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()

        self._model: Any = None
        self._tokenizer: Any = None
        self._info: Optional[EngineInfo] = None
        self._api: str = ""
        self._device: str = "cpu"
        self._dtype: torch.dtype = torch.float32

        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cached_key: Optional[str] = None
        self._cached_encoded: Any = None
        self._loading = False

    # -- state --------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def device(self) -> str:
        return self._device

    @property
    def api(self) -> str:
        return self._api

    def info(self) -> Optional[EngineInfo]:
        return self._info

    def describe(self) -> str:
        if self._info is None:
            return "model not loaded"
        return self._info.describe()

    # -- loading ------------------------------------------------------------
    def load(self, force: bool = False) -> EngineInfo:
        """Load the checkpoint (idempotent, thread-safe)."""
        with self._load_lock:
            if self._info is not None and not force:
                return self._info
            self._loading = True
            try:
                info = self._load_locked()
            finally:
                self._loading = False
            return info

    def _load_locked(self) -> EngineInfo:
        started = time.perf_counter()
        cfg = self.model_config

        model_path = resolve_model_path(cfg.model_path)
        device = self._resolve_device(cfg.device)
        dtype = self._resolve_dtype(cfg.dtype, device)
        log.info("Loading Moondream from %s (device=%s, dtype=%s)", model_path, device, dtype)

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ModelLoadError(
                "transformers is not installed; the local Moondream checkpoint "
                "cannot be loaded."
            ) from exc

        kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(cfg.trust_remote_code),
            "local_files_only": bool(cfg.local_files_only),
        }

        def _from_pretrained(target_dtype: torch.dtype) -> Any:
            try:
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        str(model_path), dtype=target_dtype, **kwargs
                    )
                except TypeError:
                    # transformers < 4.56 spells the argument ``torch_dtype``.
                    return AutoModelForCausalLM.from_pretrained(
                        str(model_path), torch_dtype=target_dtype, **kwargs
                    )
            except OSError as exc:
                raise ModelLoadError(
                    f"model files missing or unreadable: {exc}"
                ) from exc

        try:
            model = _from_pretrained(dtype)
        except (torch.OutOfMemoryError, MemoryError) as exc:
            # A CPU box without ~8 GB free cannot hold fp32 weights; bf16 fits
            # in half the RAM and is supported by PyTorch on CPU.
            if device == "cpu" and dtype is not torch.bfloat16:
                log.warning(
                    "Out of memory loading weights as %s on CPU — retrying "
                    "with bfloat16 (%s)",
                    dtype,
                    exc,
                )
                dtype = torch.bfloat16
                model = _from_pretrained(dtype)
            else:
                raise ModelLoadError(
                    f"out of memory while loading the model on {device}: {exc}"
                ) from exc

        try:
            model = model.to(device)
        except (torch.OutOfMemoryError, MemoryError) as exc:
            raise ModelLoadError(
                f"out of memory while moving the model to {device}: {exc}. "
                "Close other GPU applications or run with --device cpu."
            ) from exc
        except Exception as exc:  # pragma: no cover - device placement failure
            raise ModelLoadError(f"could not move model to {device}: {exc}") from exc

        if hasattr(model, "eval"):
            model.eval()

        self._model = model
        self._device = device
        self._dtype = dtype
        self._api = self._detect_api(model)
        if self._api == "legacy":
            self._tokenizer = self._load_tokenizer(model_path)

        info = EngineInfo(
            model_path=str(model_path),
            device=device,
            dtype=str(dtype).replace("torch.", ""),
            api=self._api,
            model_class=type(model).__name__,
            torch_version=torch.__version__,
            transformers_version=self._transformers_version(),
            load_seconds=time.perf_counter() - started,
            parameter_count=self._count_parameters(model),
            gpu_name=torch.cuda.get_device_name(0) if device.startswith("cuda") else "",
            gpu_memory_total_gb=(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if device.startswith("cuda")
                else 0.0
            ),
        )
        self._info = info
        log.info("Model ready in %.2fs — %s", info.load_seconds, info.describe())
        return info

    def unload(self) -> None:
        """Release model weights and cached encodings (used by tests/shutdown)."""
        with self._load_lock:
            self.clear_image_cache()
            self._model = None
            self._tokenizer = None
            self._info = None
            self._api = ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        log.info("Model unloaded")

    # -- configuration helpers ---------------------------------------------
    @staticmethod
    def _resolve_device(pref: str) -> str:
        pref = (pref or "auto").lower()
        if pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if pref.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable — falling back to CPU")
            return "cpu"
        if pref == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            log.warning("MPS requested but unavailable — falling back to CPU")
            return "cpu"
        return pref

    @staticmethod
    def _resolve_dtype(pref: str, device: str) -> torch.dtype:
        pref = (pref or "auto").lower()
        if pref == "bf16":
            return torch.bfloat16
        if pref == "fp16":
            return torch.float16
        if pref == "fp32":
            return torch.float32
        # auto
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.startswith("cuda"):
            return torch.float16
        return torch.float32

    @staticmethod
    def _transformers_version() -> str:
        try:
            import transformers

            return getattr(transformers, "__version__", "unknown")
        except Exception:  # pragma: no cover
            return "unknown"

    @staticmethod
    def _count_parameters(model: Any) -> int:
        try:
            return int(sum(p.numel() for p in model.parameters()))
        except Exception:  # pragma: no cover
            return 0

    def _load_tokenizer(self, model_path: Path) -> Any:
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=bool(self.model_config.trust_remote_code),
                local_files_only=bool(self.model_config.local_files_only),
            )
        except Exception as exc:
            log.warning("Tokenizer could not be loaded (%s); legacy API may fail", exc)
            return None

    @staticmethod
    def _detect_api(model: Any) -> str:
        """Pick the newest API the installed build actually supports."""
        has_encode = hasattr(model, "encode_image")
        has_query = hasattr(model, "query")
        if has_query and has_encode:
            return "hf_query"
        if has_encode and hasattr(model, "answer_question"):
            return "legacy"
        raise ModelApiError(
            f"Unsupported Moondream interface on {type(model).__name__}: "
            f"encode_image={has_encode}, query={has_query}, "
            f"answer_question={hasattr(model, 'answer_question')}. "
            "Expected a Moondream build exposing encode_image()/query()."
        )

    # -- image encoding -----------------------------------------------------
    def encode_image(
        self,
        image: Image.Image,
        *,
        cache_key: Optional[str] = None,
    ) -> tuple[Any, float, bool]:
        """Encode ``image`` (reusing the cache when ``cache_key`` matches).

        Returns ``(encoded, encode_seconds, reused)``.
        """
        self._require_loaded()
        if cache_key is not None:
            with self._cache_lock:
                if self._cached_key == cache_key and self._cached_encoded is not None:
                    return self._cached_encoded, 0.0, True

        timer = StageTimer(self._device)
        with timer.stage("encode"), torch.inference_mode():
            encoded = self._model.encode_image(image)
        encode_s = timer.durations.get("encode", 0.0)

        if cache_key is not None:
            with self._cache_lock:
                self._cached_key = cache_key
                self._cached_encoded = encoded
        return encoded, encode_s, False

    def clear_image_cache(self) -> None:
        with self._cache_lock:
            self._cached_key = None
            self._cached_encoded = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def cached_frame_key(self) -> Optional[str]:
        with self._cache_lock:
            return self._cached_key

    # -- inference ----------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        frame: Optional[Any] = None,
        image: Optional[Image.Image] = None,
        frame_id: Optional[int] = None,
        frame_timestamp: Optional[float] = None,
        cache_key: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Answer ``question`` about one image.

        Exactly one of ``frame`` (BGR ndarray) or ``image`` (PIL) must be
        given.  Never raises for model-side failures: an error payload with
        ``status == "error"`` is returned instead.
        """
        question = (question or "").strip()
        base = {
            "question": question,
            "frame_id": frame_id,
            "frame_timestamp": frame_timestamp,
        }

        if not question:
            return failed_result("question is empty", **base)
        if len(question) > 4000:
            return failed_result("question is too long (max 4000 characters)", **base)
        if frame is None and image is None:
            return failed_result("no image supplied for the question", **base)
        if not self.is_loaded:
            return failed_result("model is not loaded yet", **base)

        grounded = bool(self.inference_config.grounding_prompt)
        prompt_used = self._build_prompt(question, grounded)
        base["prompt_used"] = prompt_used
        base["grounded"] = grounded

        # Serialise GPU work: the model owns one shared KV cache.
        with self._inference_lock:
            timer = StageTimer(self._device)
            try:
                with timer.stage("preprocess"):
                    pil_image = self._to_pil(image, frame)
                encoded, encode_s, reused = self.encode_image(
                    pil_image,
                    cache_key=cache_key,
                )
                with timer.stage("generate"), torch.inference_mode():
                    answer = self._generate(encoded, prompt_used, settings)
                total = timer.snapshot()
                result = successful_result(
                    answer=answer,
                    inference_time_seconds=total["total"],
                    preprocess_time_seconds=total.get("preprocess", 0.0),
                    encode_time_seconds=encode_s,
                    generate_time_seconds=total.get("generate", 0.0),
                    encoding_reused=reused,
                    device=self._device,
                    model_path=str(self._info.model_path) if self._info else "",
                    api=self._api,
                    gpu_memory_mb=gpu_memory_mb(self._device),
                    **base,
                )
                log.info(
                    "Q: %.2fs (pre=%.0fms enc=%.0fms gen=%.0fms%s) A: %s",
                    result["inference_time_seconds"],
                    result["preprocess_time_seconds"] * 1000,
                    result["encode_time_seconds"] * 1000,
                    result["generate_time_seconds"] * 1000,
                    ", reused" if reused else "",
                    (answer[:120] + "…") if len(answer) > 120 else answer,
                )
                return result
            except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
                log.error("CUDA out of memory: %s", exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return failed_result(
                    f"CUDA out of memory: {exc}. Try a smaller resolution or "
                    "restart the application.",
                    device=self._device,
                    **base,
                )
            except InvalidFrameError as exc:
                return failed_result(f"invalid image: {exc}", device=self._device, **base)
            except Exception as exc:  # noqa: BLE001 - boundary: never crash the app
                log.exception("Inference failed")
                return failed_result(
                    f"{type(exc).__name__}: {exc}", device=self._device, **base
                )

    def _to_pil(
        self,
        image: Optional[Image.Image],
        frame: Optional[Any],
    ) -> Image.Image:
        if image is not None:
            if not isinstance(image, Image.Image):
                raise InvalidFrameError(f"expected PIL image, got {type(image)!r}")
            converted = image.convert("RGB")
            return resize_max_side(
                converted, self.inference_config.encode_max_side
            )
        return frame_to_pil(frame, max_side=self.inference_config.encode_max_side)

    def _build_prompt(self, question: str, grounded: bool) -> str:
        if not grounded:
            return question
        suffix = (self.inference_config.grounding_text or "").strip()
        if not suffix:
            return question
        return f"{question}\n\n{suffix}"

    def _generate(
        self,
        encoded: Any,
        prompt: str,
        settings: Optional[Dict[str, Any]],
    ) -> str:
        cfg = self.model_config
        sampling = dict(cfg.sampling_settings)
        if settings:
            sampling.update({k: v for k, v in settings.items() if v is not None})

        if self._api == "hf_query":
            out = self._model.query(image=encoded, question=prompt, settings=sampling)
            return self._normalise_answer(out)

        # Legacy API -------------------------------------------------------
        answer = self._model.answer_question(encoded, prompt, self._tokenizer)
        return self._normalise_answer(answer)

    @staticmethod
    def _normalise_answer(out: Any) -> str:
        if isinstance(out, dict):
            for key in ("answer", "caption", "text"):
                if key in out:
                    return str(out[key]).strip()
            return str(out).strip()
        if isinstance(out, (list, tuple)):
            return " ".join(str(x) for x in out).strip()
        if torch.is_tensor(out):
            return str(out.detach().cpu().tolist()).strip()
        return str(out).strip()

    # -- convenience --------------------------------------------------------
    def caption(self, frame: Any, length: str = "short") -> Dict[str, Any]:
        """Optional caption helper (same result schema as :meth:`ask`)."""
        self._require_loaded()
        if not hasattr(self._model, "caption"):
            return failed_result("captioning is not supported by this build")
        try:
            pil_image = self._to_pil(None, frame)
            timer = StageTimer(self._device)
            with timer.stage("generate"), torch.inference_mode():
                out = self._model.caption(pil_image, length=length)
            text = out.get("caption", "") if isinstance(out, dict) else str(out)
            return successful_result(
                answer=str(text).strip(),
                inference_time_seconds=timer.snapshot()["total"],
                generate_time_seconds=timer.durations.get("generate", 0.0),
                device=self._device,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("caption failed")
            return failed_result(f"{type(exc).__name__}: {exc}")

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelLoadError(
                "model is not loaded — call load() first "
                f"(searched path: {self.model_config.model_path or 'auto'})"
            )

    # -- debugging ----------------------------------------------------------
    def snapshot_cache(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "device": self._device,
            "api": self._api,
            "cached_frame_key": self.cached_frame_key,
            "gpu_memory_mb": gpu_memory_mb(self._device),
        }
