"""Cloud vision backend (any OpenAI-compatible VLM endpoint).

Sends a base64 JPEG of the camera frame to a chat-completions endpoint that
accepts ``image_url`` content parts (GPT-4o, Groq vision models, Qwen-VL via
DashScope-compat, LLaVA-on-vLLM, …).  Configuration comes from
``VLM_PROVIDER`` / ``VLM_BASE_URL`` / ``VLM_API_KEY`` / ``VLM_MODEL`` (see
``config/provider_presets.py``); when those are unset it reuses the main
LLM credentials, which is usually what you want for OpenAI/Groq.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any, Optional

from config.settings import VLMSettings, get_settings
from src.intelligence.llm.base import classify_provider_error
from src.orchestrator.schemas import VisualEvidence
from src.perception.preprocessing import frame_to_pil
from src.utils.exceptions import AuthenticationError, CloudServiceError, InferenceError
from src.utils.logger import get_logger

log = get_logger("vlm.cloud")


def _frame_to_jpeg_b64(image: Any, quality: int = 85) -> str:
    """Camera frame (BGR/ndarray, PIL image, or Frame) → base64 JPEG."""
    frame = getattr(image, "image", image)  # unwrap Frame
    if not hasattr(frame, "convert"):
        frame = frame_to_pil(frame)
    buf = io.BytesIO()
    frame.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class CloudVLM:
    """OpenAI-compatible vision chat client."""

    def __init__(self, settings: Optional[VLMSettings] = None, provider_name: str = "openai_compat"):
        self.settings = settings or get_settings().vlm
        self.provider = provider_name
        self._client = None
        self._lock = threading.Lock()
        self._loaded = False
        self._questions = 0
        self._last_error: Optional[str] = None

    # -- credentials ------------------------------------------------------
    @property
    def _api_key(self) -> str:
        if self.settings.api_key:
            return self.settings.api_key
        llm = get_settings().llm
        return llm.api_key or ""

    @property
    def _base_url(self) -> str:
        if self.settings.base_url:
            return self.settings.base_url
        llm = get_settings().llm
        if self.provider in {"groq", "openai"} or llm.provider == self.provider:
            return llm.base_url
        return ""

    @property
    def _model(self) -> str:
        return self.settings.model or get_settings().llm.model

    # -- interface --------------------------------------------------------
    def load(self, force: bool = False) -> None:
        with self._lock:
            if self._loaded and not force:
                return
            if not self._api_key:
                raise AuthenticationError(
                    f"No API key for vision provider '{self.provider}'.",
                    hint="Set VLM_API_KEY, LLM_API_KEY or the provider key env var.",
                )
            if not self._base_url:
                raise CloudServiceError(
                    f"Vision provider '{self.provider}' needs a base URL",
                    hint="Set VLM_BASE_URL (or LLM_BASE_URL) to an OpenAI-compatible /v1 endpoint.",
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise CloudServiceError(
                    "the 'openai' package is required for cloud vision",
                    hint="pip install --user --break-system-packages openai",
                ) from exc
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=max(get_settings().llm.timeout_s, 30.0),
                max_retries=0,
            )
            self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ask(
        self,
        image: Any,
        question: str,
        *,
        frame_id: Optional[str] = None,
        ground: bool = True,
        max_new_tokens: Optional[int] = None,
    ) -> VisualEvidence:
        if not self._loaded:
            self.load()
        question = (question or "").strip() or "Describe everything visible in front of the camera."
        frame = getattr(image, "image", image)
        src_frame_id = frame_id or getattr(image, "tag", None)
        frame_ts = getattr(image, "timestamp", None)
        image_age = getattr(image, "age_s", None)

        t0 = time.perf_counter()
        try:
            b64 = _frame_to_jpeg_b64(frame)
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=max_new_tokens or get_settings().vlm.max_new_tokens,
                temperature=get_settings().vlm.temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            self._questions += 1
            self._last_error = None
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            raise classify_provider_error(exc, self.provider) from exc

        total_ms = (time.perf_counter() - t0) * 1000
        return VisualEvidence(
            source=f"{self.provider}:{self._model}",
            frame_id=src_frame_id or "",
            question=question,
            raw_answer=text,
            status="completed" if text else "failed",
            inference_time_ms=round(total_ms, 1),
            frame_timestamp=frame_ts,
            image_age_s=None if image_age is None else round(image_age, 3),
            error=None if text else "empty answer",
        )

    def caption(self, image: Any, length: str = "normal") -> VisualEvidence:
        question = {
            "brief": "List the main objects you can see.",
            "normal": "Describe everything you can see in front of the camera.",
            "detailed": "Describe the scene in detail: objects, positions, colours, relationships.",
        }.get(length, "Describe everything you can see in front of the camera.")
        return self.ask(image, question)

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded,
            "provider": self.provider,
            "model": self._model,
            "base_url": self._base_url,
            "questions_answered": self._questions,
            "last_error": self._last_error,
            "describe": f"cloud VLM · {self._model} · {self.provider}",
        }

    def unload(self) -> None:
        with self._lock:
            self._client = None
            self._loaded = False
