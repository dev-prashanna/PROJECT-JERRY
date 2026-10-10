"""Anthropic (Claude) chat client via the Messages API.

Uses plain ``httpx`` (no anthropic SDK required).  The transport is a
swappable callable so tests can inject a fake without touching the network:

    client._transport = lambda payload: {"content": [{"text": "hi"}], ...}
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Sequence

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

log = get_logger("llm.anthropic")

ANTHROPIC_VERSION = "2023-06-01"


class _HTTPError(Exception):
    """Minimal error object carrying an HTTP status for classification."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, settings: Optional[LLMSettings] = None):
        self.settings = settings or get_settings().llm
        self.provider = self.settings.provider or "anthropic"
        self._lock = threading.Lock()
        self._model_chain = build_model_chain(
            self.settings.model, self.settings.fallback_model
        )
        #: ``payload -> response-dict``; default posts to the Messages API.
        self._transport: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None

    # -- interface -------------------------------------------------------
    @property
    def model_chain(self) -> list[str]:
        return list(self._model_chain)

    @property
    def offline(self) -> bool:
        return not self.settings.available

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(payload)
        if not self.settings.api_key:
            raise AuthenticationError(
                "No API key for Anthropic.",
                hint="Set ANTHROPIC_API_KEY or LLM_API_KEY in AURA/.env.",
            )
        import httpx

        url = (self.settings.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        try:
            resp = httpx.post(
                url,
                headers={
                    "x-api-key": self.settings.api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json=payload,
                timeout=self.settings.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise CloudServiceError(f"network problem talking to Anthropic: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text[:200])
            except Exception:  # noqa: BLE001
                detail = resp.text[:200]
            raise _HTTPError(resp.status_code, detail)
        return resp.json()

    def _call(self, model_id: str, messages: Sequence[dict[str, Any]], kwargs: dict[str, Any]) -> LLMResult:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in {"user", "assistant"}
        ]
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens") or self.settings.max_completion_tokens,
        }
        if system_parts:
            payload["system"] = "\n\n".join(str(p) for p in system_parts)
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]

        t0 = time.perf_counter()
        data = self._post(payload)
        latency = time.perf_counter() - t0
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            model=data.get("model") or model_id,
            latency_s=latency,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            finish_reason=data.get("stop_reason") or "",
            from_fallback=(model_id != self._model_chain[0] if self._model_chain else False),
            provider=self.provider,
        )

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
                "No Anthropic API key configured.",
                hint="Set ANTHROPIC_API_KEY in AURA/.env.",
            )
        chain = [model] if model else self._model_chain
        chain = [m for m in chain if m]
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        budget = budget_s or self.settings.timeout_s * max(1, self.settings.max_retries + 1)
        deadline = time.perf_counter() + budget
        last_exc: Optional[BaseException] = None

        for attempt_index in range(self.settings.max_retries + 1):
            for idx, model_id in enumerate(chain):
                if time.perf_counter() > deadline:
                    break
                if metrics:
                    metrics.start(f"{self.provider}_call")
                try:
                    return self._call(model_id, messages, kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    status = getattr(exc, "status_code", None)
                    if status in (400, 404) and idx < len(chain) - 1:
                        log.warning("model %s rejected (%s); trying %s", model_id, status, chain[idx + 1])
                        continue
                    if status == 429 and attempt_index < self.settings.max_retries:
                        time.sleep(min(2.0 * (2**attempt_index), 8.0))
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
            else:
                if attempt_index >= self.settings.max_retries:
                    break
                time.sleep(min(2.0 * (2**attempt_index), 8.0))
                continue

        raise classify_provider_error(
            last_exc or CloudServiceError("Anthropic request failed."), self.provider
        )

    def ping(self) -> bool:
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=8, budget_s=10.0)
            return True
        except Exception:  # noqa: BLE001
            return False
