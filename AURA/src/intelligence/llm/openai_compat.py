"""OpenAI-compatible chat client.

Talks the ``POST {base}/chat/completions`` dialect used by OpenAI, Groq,
Moonshot/Kimi, DeepSeek, MiMo, Together, Mistral, vLLM, Ollama, LM Studio
and most local servers.  One class covers every one of them — only
``base_url`` / ``api_key`` / ``model`` change (see
``config/provider_presets.py``).

The official ``openai`` SDK is used when available (it is a soft
dependency); otherwise we fall back to plain ``httpx`` so the module works
anywhere.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional, Sequence

from config.settings import LLMSettings, get_settings
from src.intelligence.llm.base import (
    BaseLLMClient,
    LLMResult,
    build_model_chain,
    classify_provider_error,
)
from src.utils.exceptions import AuthenticationError, CloudServiceError
from src.utils.logger import get_logger
from src.utils.metrics import RequestMetrics

log = get_logger("llm.openai_compat")


class OpenAICompatClient(BaseLLMClient):
    """Chat client for any OpenAI-compatible endpoint."""

    provider = "openai_compat"

    def __init__(self, settings: Optional[LLMSettings] = None, provider_name: str = ""):
        self.settings = settings or get_settings().llm
        self.provider = provider_name or self.settings.provider or "openai_compat"
        self._client = None
        self._lock = threading.Lock()
        self._model_chain = build_model_chain(
            self.settings.model, self.settings.fallback_model
        )

    # -- construction ----------------------------------------------------
    @property
    def model_chain(self) -> list[str]:
        return list(self._model_chain)

    @property
    def offline(self) -> bool:
        return not self.settings.available

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    if not self.settings.api_key:
                        raise AuthenticationError(
                            f"No API key for provider '{self.provider}'.",
                            hint=f"Set LLM_API_KEY or the provider's key env var in AURA/.env.",
                        )
                    try:
                        from openai import OpenAI
                    except ImportError as exc:  # pragma: no cover - soft dep
                        raise CloudServiceError(
                            "the 'openai' package is required for this provider",
                            hint="pip install --user --break-system-packages openai",
                        ) from exc
                    if not self.settings.base_url:
                        raise CloudServiceError(
                            f"provider '{self.provider}' needs a base URL",
                            hint="Set LLM_BASE_URL to the provider's /v1 endpoint.",
                        )
                    self._client = OpenAI(
                        api_key=self.settings.api_key,
                        base_url=self.settings.base_url,
                        timeout=self.settings.timeout_s,
                        max_retries=0,  # retries are handled here, explicitly
                    )
        return self._client

    # -- core call -------------------------------------------------------
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Any = None,
        response_format: Optional[dict[str, Any]] = None,
        metrics: Optional[RequestMetrics] = None,
        budget_s: Optional[float] = None,
    ) -> LLMResult:
        if self.offline:
            raise CloudServiceError(
                f"No API key configured for provider '{self.provider}'.",
                hint="Set LLM_API_KEY (or the provider key env var) in AURA/.env.",
            )

        chain = [model] if model else self._model_chain
        chain = [m for m in chain if m]
        budget = budget_s or self.settings.timeout_s * max(1, self.settings.max_retries + 1)
        deadline = time.perf_counter() + budget
        last_exc: Optional[BaseException] = None

        for attempt_index in range(self.settings.max_retries + 1):
            for idx, model_id in enumerate(chain):
                if time.perf_counter() > deadline:
                    break
                kwargs: dict[str, Any] = {
                    "model": model_id,
                    "messages": list(messages),
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                # Most OpenAI-compatible servers accept `max_tokens`;
                # newer OpenAI models prefer `max_completion_tokens`.
                kwargs["max_tokens"] = max_tokens or self.settings.max_completion_tokens
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice is not None:
                        kwargs["tool_choice"] = tool_choice
                if response_format is not None:
                    kwargs["response_format"] = response_format

                t0 = time.perf_counter()
                if metrics:
                    metrics.start(f"{self.provider}_call")
                try:
                    resp = self._get_client().chat.completions.create(**kwargs)
                except Exception as exc:  # noqa: BLE001 - normalised below
                    if metrics:
                        metrics.stop(f"{self.provider}_call")
                    last_exc = exc
                    status = getattr(exc, "status_code", None)
                    if status in (400, 404) and idx < len(chain) - 1:
                        log.warning(
                            "model %s rejected (%s); trying %s", model_id, status, chain[idx + 1]
                        )
                        continue
                    if status == 429 and attempt_index < self.settings.max_retries:
                        delay = min(2.0 * (2**attempt_index), 8.0)
                        log.warning("rate limited; backing off %.1fs", delay)
                        time.sleep(delay)
                        break
                    if status in (400, 401, 403, 404) or status is None:
                        raise classify_provider_error(exc, self.provider)
                    if attempt_index >= self.settings.max_retries:
                        raise classify_provider_error(exc, self.provider)
                    time.sleep(min(2.0 * (2**attempt_index), 8.0))
                    break
                finally:
                    if metrics:
                        metrics.stop(f"{self.provider}_call")

                latency = time.perf_counter() - t0
                choice = resp.choices[0]
                msg = choice.message
                usage = getattr(resp, "usage", None)
                reasoning = 0
                if usage is not None:
                    details = getattr(usage, "completion_tokens_details", None)
                    reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
                result = LLMResult(
                    text=msg.content or "",
                    model=getattr(resp, "model", "") or model_id,
                    latency_s=latency,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    reasoning_tokens=reasoning,
                    tool_calls=list(getattr(msg, "tool_calls", None) or []) or None,
                    finish_reason=getattr(choice, "finish_reason", "") or "",
                    from_fallback=(model_id != self._model_chain[0] if self._model_chain else False),
                    provider=self.provider,
                )
                if result.from_fallback:
                    log.info("served by fallback model %s", model_id)
                return result
            else:
                if attempt_index >= self.settings.max_retries:
                    break
                time.sleep(min(2.0 * (2**attempt_index), 8.0))
                continue

        raise classify_provider_error(
            last_exc or CloudServiceError(f"{self.provider} request failed."), self.provider
        )

    def ping(self) -> bool:
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=8, budget_s=10.0)
            return True
        except Exception:  # noqa: BLE001
            return False
