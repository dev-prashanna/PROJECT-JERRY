"""GroqCloud client with bounded retries, model fallback and strict budgets.

Design notes
------------
* The configuration value ``GROQ_BASE_URL`` follows the OpenAI-compatible
  convention ``https://api.groq.com/openai/v1``.  The official ``groq``
  SDK appends that path itself, so :func:`sdk_base_url` strips it.
* Retries use bounded exponential backoff and **never** retry 4xx errors
  other than 429.
* Every call receives a wall-clock budget so a hung network request can not
  block the pipeline forever.
* Only text is transmitted.  Images and audio never leave the machine.

This is the *groq* provider of AURA's pluggable chat layer; other providers
(OpenAI, Claude, Kimi, DeepSeek, MiMo, any OpenAI-compatible server) live in
:mod:`src.intelligence.llm` and are selected with ``LLM_PROVIDER``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Sequence

from config.settings import LLMSettings, get_settings
from src.intelligence.llm.base import (  # noqa: F401 - re-exported for compat
    BaseLLMClient,
    LLMResult,
    build_model_chain,
    parse_json_reply as _parse_json,
)
from src.utils.exceptions import (
    AURAError,
    AuthenticationError,
    CloudServiceError,
    InvalidModelError,
    RateLimitError,
    classify_groq_error,
)
from src.utils.logger import get_logger
from src.utils.metrics import RequestMetrics, StageTimer

log = get_logger("groq")


def sdk_base_url(base_url: str) -> str:
    """Convert an OpenAI-compatible base URL to what the Groq SDK expects."""
    url = (base_url or "").rstrip("/")
    for suffix in ("/openai/v1",):
        if url.endswith(suffix):
            return url[: -len(suffix)] or "https://api.groq.com"
    return url


class GroqClient(BaseLLMClient):
    """Thin, thread-safe wrapper around the Groq SDK."""

    provider = "groq"

    def __init__(self, settings: Optional[LLMSettings] = None):
        self.settings = settings or get_settings().llm
        self._client = None
        self._async_client = None
        self._lock = threading.Lock()
        self._model_chain = self._build_chain()

    # -- construction ----------------------------------------------------
    def _build_chain(self) -> list[str]:
        return build_model_chain(
            self.settings.model,
            self.settings.fallback_model,
            default="openai/gpt-oss-120b",
        )

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
                    from groq import Groq

                    if not self.settings.api_key:
                        raise AuthenticationError("GROQ_API_KEY is not configured.")
                    self._client = Groq(
                        api_key=self.settings.api_key,
                        base_url=sdk_base_url(self.settings.base_url),
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
        """Run one chat completion with retries and model fallback."""
        if self.offline:
            raise CloudServiceError(
                "No Groq API key configured.",
                hint="Set GROQ_API_KEY (or LLM_API_KEY) in AURA/.env (offline mode remains available).",
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
                kwargs["max_completion_tokens"] = max_tokens or self.settings.max_completion_tokens
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice is not None:
                        kwargs["tool_choice"] = tool_choice
                if response_format is not None:
                    kwargs["response_format"] = response_format

                t0 = time.perf_counter()
                if metrics:
                    metrics.start("groq_call")
                try:
                    resp = self._get_client().chat.completions.create(**kwargs)
                except Exception as exc:  # noqa: BLE001 - normalised below
                    if metrics:
                        metrics.stop("groq_call")
                    last_exc = exc
                    status = getattr(exc, "status_code", None)
                    if status in (400, 404) and idx < len(chain) - 1:
                        log.warning("model %s rejected (%s); trying %s", model_id, status, chain[idx + 1])
                        continue
                    if status == 429 and attempt_index < self.settings.max_retries:
                        delay = min(2.0 * (2**attempt_index), 8.0)
                        log.warning("rate limited; backing off %.1fs", delay)
                        time.sleep(delay)
                        break  # restart attempts loop
                    if status in (400, 401, 403, 404) or status is None:
                        raise classify_groq_error(exc)
                    if attempt_index >= self.settings.max_retries:
                        raise classify_groq_error(exc)
                    time.sleep(min(2.0 * (2**attempt_index), 8.0))
                    break
                finally:
                    if metrics:
                        metrics.stop("groq_call")

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
                    model=resp.model or model_id,
                    latency_s=latency,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    reasoning_tokens=reasoning,
                    tool_calls=list(getattr(msg, "tool_calls", None) or []) or None,
                    finish_reason=getattr(choice, "finish_reason", "") or "",
                    from_fallback=(model_id != self._model_chain[0]),
                    provider=self.provider,
                )
                if result.from_fallback:
                    log.info("served by fallback model %s", model_id)
                return result
            else:
                # exhausted chain without success and without break
                if attempt_index >= self.settings.max_retries:
                    break
                time.sleep(min(2.0 * (2**attempt_index), 8.0))
                continue

        raise classify_groq_error(last_exc or CloudServiceError("Groq request failed."))

    def chat_json(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> tuple[Any, LLMResult]:
        """Chat and parse the reply as JSON."""
        result = self.chat(messages, **kwargs)
        import json as _json

        try:
            return result.json(), result
        except (ValueError, _json.JSONDecodeError) as exc:
            raise CloudServiceError(
                f"LLM returned unparseable JSON: {exc}",
                hint="The raw reply is available in the logs.",
            ) from exc

    def ping(self) -> bool:
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=8, budget_s=10.0)
            return True
        except AURAError:
            return False


class AsyncGroqClient:
    """AsyncIO facade used by network-facing stages of the pipeline."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        self.settings = settings or get_settings().llm
        self._client = None
        self._lock = threading.Lock()

    def _get(self):
        if self._client is None:
            from groq import AsyncGroq

            if not self.settings.api_key:
                raise AuthenticationError("GROQ_API_KEY is not configured.")
            self._client = AsyncGroq(
                api_key=self.settings.api_key,
                base_url=sdk_base_url(self.settings.base_url),
                timeout=self.settings.timeout_s,
                max_retries=0,
            )
        return self._client

    async def chat(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> LLMResult:
        import asyncio

        client = self._get()
        models = [kwargs.pop("model", None) or m for m in self.model_chain]
        max_retries = self.settings.max_retries
        last_exc: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            for idx, model_id in enumerate(models):
                payload = dict(kwargs)
                payload.setdefault("max_completion_tokens", self.settings.max_completion_tokens)
                t0 = time.perf_counter()
                try:
                    resp = await client.chat.completions.create(
                        model=model_id, messages=list(messages), **payload
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    status = getattr(exc, "status_code", None)
                    if status == 429 and attempt < max_retries:
                        await asyncio.sleep(min(2.0 * 2**attempt, 8.0))
                        break
                    if idx < len(models) - 1 and status in (400, 404):
                        continue
                    if attempt >= max_retries:
                        raise classify_groq_error(exc)
                    await asyncio.sleep(min(2.0 * 2**attempt, 8.0))
                    break
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                return LLMResult(
                    text=choice.message.content or "",
                    model=resp.model or model_id,
                    latency_s=time.perf_counter() - t0,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    tool_calls=list(getattr(choice.message, "tool_calls", None) or []) or None,
                    finish_reason=choice.finish_reason or "",
                    from_fallback=model_id != models[0],
                    provider="groq",
                )
            else:
                continue
        raise classify_groq_error(last_exc or CloudServiceError("Groq request failed."))

    @property
    def model_chain(self) -> list[str]:
        chain = []
        for m in (self.settings.model, self.settings.fallback_model):
            if m and m not in chain:
                chain.append(m)
        return chain


def with_timeout(fn: Callable[[], Any], seconds: float, what: str) -> Any:
    """Run ``fn`` in a worker thread with a hard timeout.

    Used for local blocking inference so a wedged model can not hang the app.
    """
    result: dict[str, Any] = {}
    done = threading.Event()

    def runner() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=runner, name=f"timeout:{what}", daemon=True)
    thread.start()
    if not done.wait(seconds):
        raise AURAError(f"{what} timed out after {seconds:.0f}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")
