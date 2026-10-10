"""Provider-agnostic LLM client contract.

Every chat backend AURA can use (Groq, OpenAI, Claude, Kimi, DeepSeek,
MiMo, any OpenAI-compatible server) implements :class:`BaseLLMClient`:

* ``chat(messages, **kw) -> LLMResult``
* ``chat_json(messages, **kw) -> (parsed, LLMResult)``
* ``ping() -> bool``
* ``offline: bool``   – True when no usable API key is configured
* ``model_chain: list[str]``
* ``provider: str``

New providers plug in through
:func:`src.intelligence.llm.registry.create_llm_client` (see that module).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from src.utils.exceptions import (
    AURAError,
    AuthenticationError,
    CloudServiceError,
    InvalidModelError,
    RateLimitError,
    classify_groq_error,
)
from src.utils.metrics import RequestMetrics


@dataclass
class LLMResult:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    tool_calls: list[Any] | None = None
    finish_reason: str = ""
    from_fallback: bool = False
    provider: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def json(self) -> Any:
        """Parse the reply as JSON, tolerating code fences and preamble."""
        return parse_json_reply(self.text)


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def parse_json_reply(text: str) -> Any:
    """Parse JSON out of an LLM reply, tolerating code fences and prose."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if m:
            return json.loads(m.group(0))
        raise


def classify_provider_error(exc: BaseException, provider: str = "LLM") -> AURAError:
    """Normalise any SDK/HTTP error into the AURA exception hierarchy.

    Falls back to :func:`classify_groq_error` (already status-code driven,
    works for every OpenAI-style error object) and re-words the message with
    the provider name.
    """
    err = classify_groq_error(exc)
    label = provider or "LLM"
    text = str(err)
    text = text.replace("Groq", label)
    if isinstance(err, AuthenticationError):
        return AuthenticationError(text, hint=f"Set the {label.upper()}_API_KEY (or LLM_API_KEY) in AURA/.env.", cause=exc)
    if isinstance(err, RateLimitError):
        return RateLimitError(text, hint="Wait before retrying; AURA retries with bounded backoff.", cause=exc)
    if isinstance(err, InvalidModelError):
        return InvalidModelError(text, hint="Set LLM_MODEL to a model id offered by this provider.", cause=exc)
    return type(err)(text, hint=getattr(err, "hint", None), cause=exc)


class BaseLLMClient(ABC):
    """Common interface every chat provider must implement."""

    #: registry name, e.g. ``"groq"`` or ``"openai"``.
    provider: str = "llm"

    @abstractmethod
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
        """One chat completion with bounded retries + model fallback."""

    def chat_json(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> tuple[Any, LLMResult]:
        """Chat and parse the reply as JSON."""
        result = self.chat(messages, **kwargs)
        try:
            return result.json(), result
        except (ValueError, json.JSONDecodeError) as exc:
            raise CloudServiceError(
                f"LLM returned unparseable JSON: {exc}",
                hint="The raw reply is available in the logs.",
            ) from exc

    @abstractmethod
    def ping(self) -> bool:
        """Cheap connectivity probe (True when the provider answered)."""

    @property
    @abstractmethod
    def offline(self) -> bool:
        """True when no usable credentials are configured."""

    @property
    @abstractmethod
    def model_chain(self) -> list[str]:
        """Ordered model ids to try (primary first)."""

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "models": self.model_chain,
            "offline": self.offline,
        }


def build_model_chain(primary: str, fallback: str, default: str = "") -> list[str]:
    """Unique, ordered models to try: primary, fallback, then default if either is empty."""
    chain: list[str] = []
    for m in (primary or default, fallback or (default if primary else "")):
        if m and m not in chain:
            chain.append(m)
    return chain
