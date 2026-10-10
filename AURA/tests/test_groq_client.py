"""Groq client unit tests (no network): URL handling, retries, classification."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from config.settings import LLMSettings
from src.intelligence.groq_client import (
    GroqClient,
    LLMResult,
    sdk_base_url,
    _parse_json,
    with_timeout,
)
from src.utils.exceptions import (
    AuthenticationError,
    CloudServiceError,
    InvalidModelError,
    RateLimitError,
    classify_groq_error,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _HTTPError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message or f"status {status_code}")
        self.status_code = status_code


def _response(text: str = "hello", model: str = "fake-model", tool_calls=None, usage=True):
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage_obj = (
        SimpleNamespace(prompt_tokens=11, completion_tokens=7, completion_tokens_details=None)
        if usage
        else None
    )
    return SimpleNamespace(choices=[choice], model=model, usage=usage_obj)


class _FakeSDK:
    """Mimics ``groq.Groq().chat.completions.create``."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(kwargs)
        return item


def _client(script, **overrides) -> tuple[GroqClient, _FakeSDK]:
    kwargs: dict[str, Any] = {
        "api_key": "gsk_dummy",
        "model": "primary-model",
        "fallback_model": "fallback-model",
        "max_retries": 1,
        "timeout_s": 5.0,
    }
    kwargs.update(overrides)
    settings = LLMSettings(**kwargs)
    client = GroqClient(settings)
    sdk = _FakeSDK(script)
    client._client = sdk          # inject before any real call
    return client, sdk


# ---------------------------------------------------------------------------
# URL / JSON helpers
# ---------------------------------------------------------------------------

def test_sdk_base_url_strips_openai_suffix():
    assert sdk_base_url("https://api.groq.com/openai/v1") == "https://api.groq.com"
    assert sdk_base_url("https://api.groq.com/openai/v1/") == "https://api.groq.com"
    assert sdk_base_url("https://api.groq.com") == "https://api.groq.com"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here is the plan:\n{"intent": "greeting"}', {"intent": "greeting"}),
        ('```\n[1, 2]\n```', [1, 2]),
    ],
)
def test_parse_json_tolerates_markdown(raw, expected):
    assert _parse_json(raw) == expected


def test_parse_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _parse_json("")


# ---------------------------------------------------------------------------
# error classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status,exc_type",
    [
        (401, AuthenticationError),
        (429, RateLimitError),
        (404, InvalidModelError),
        (500, CloudServiceError),
    ],
)
def test_classify_groq_error(status, exc_type):
    assert isinstance(classify_groq_error(_HTTPError(status)), exc_type)


def test_classify_network_error():
    assert isinstance(classify_groq_error(ConnectionError("no route")), CloudServiceError)


def test_every_error_carries_a_hint():
    for status in (401, 404, 429, 500):
        err = classify_groq_error(_HTTPError(status))
        assert err.hint, f"no hint for {status}"


# ---------------------------------------------------------------------------
# client behaviour
# ---------------------------------------------------------------------------

def test_offline_client_refuses():
    client = GroqClient(LLMSettings(api_key=""))
    assert client.offline
    with pytest.raises(CloudServiceError):
        client.chat([{"role": "user", "content": "hi"}])


def test_successful_chat_reports_usage_and_latency():
    client, sdk = _client([_response("hi there")])
    result = client.chat([{"role": "user", "content": "hi"}])
    assert isinstance(result, LLMResult)
    assert result.text == "hi there"
    assert result.prompt_tokens == 11 and result.completion_tokens == 7
    assert result.latency_s >= 0
    assert sdk.calls[0]["model"] == "primary-model"


def test_404_falls_back_to_secondary_model():
    client, sdk = _client([_HTTPError(404), _response("from fallback", model="fallback-model")])
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.model == "fallback-model"
    assert result.from_fallback
    assert [c["model"] for c in sdk.calls] == ["primary-model", "fallback-model"]


def test_rate_limit_is_retried_not_looped_forever():
    # 429 → backoff → retry (max_retries=1 → exactly two attempts)
    client, sdk = _client([_HTTPError(429), _HTTPError(429)])
    with pytest.raises(RateLimitError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(sdk.calls) <= 4  # primary+fallback per attempt, bounded


def test_auth_error_is_not_retried():
    client, sdk = _client([_HTTPError(401)])
    with pytest.raises(AuthenticationError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(sdk.calls) == 1


def test_json_helper_parses_reply():
    payload = {"intent": "visual_question", "requires_camera": True}
    client, _ = _client([_response(json.dumps(payload))])
    parsed, result = client.chat_json([{"role": "user", "content": "x"}])
    assert parsed == payload


def test_json_helper_rejects_non_json():
    client, _ = _client([_response("I am not going to answer in JSON.")])
    with pytest.raises(CloudServiceError):
        client.chat_json([{"role": "user", "content": "x"}])


def test_model_chain_configuration():
    client, _ = _client([])
    assert client.model_chain == ["primary-model", "fallback-model"]


def test_tool_calls_are_propagated():
    calls = [SimpleNamespace(id="t1", function=SimpleNamespace(name="route", arguments="{}"))]
    client, sdk = _client([_response("", tool_calls=calls)])
    result = client.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert result.tool_calls and result.tool_calls[0].function.name == "route"
    assert "tools" in sdk.calls[0]


def test_budget_stops_endless_loops():
    client, _ = _client([_HTTPError(500)] * 50, max_retries=1)
    with pytest.raises(CloudServiceError):
        client.chat([{"role": "user", "content": "x"}], budget_s=0.5)


def test_with_timeout_returns_value():
    assert with_timeout(lambda: 42, 1.0, "unit") == 42


def test_with_timeout_raises_on_hang():
    import time as _time

    from src.utils.exceptions import AURAError

    with pytest.raises(AURAError) as excinfo:
        with_timeout(lambda: _time.sleep(3), 0.2, "slow-thing")
    assert "timed out" in str(excinfo.value)


def test_empty_content_is_represented():
    client, _ = _client([_response("")])
    assert client.chat([{"role": "user", "content": "x"}]).text == ""
