"""Memory provider tests use mocked transport and isolated SQLite databases."""
import io
import json
import urllib.error
from dataclasses import replace

import pytest

from config.settings import Settings
from src.memory.engine import MultimodalMemoryEngine
from src.memory.fact_extractor import extract_facts
from src.memory.gemini_memory import GeminiMemoryClient, select_context
from src.memory.indexing_worker import process_pending_jobs
from src.memory.models import RememberFactRequest


def client_reply(reply):
    client = GeminiMemoryClient("test-secret", "gemini-3.8-flash")
    client.chat_json = lambda *a, **kw: (reply, None)
    return client


def durable_reply():
    return {"facts": [{"subject": "current_speaker", "predicate": "occupation",
                        "object": "engineer", "evidence": "I work as an engineer", "durable": True}],
            "correction": False}


def test_native_transport_and_secret_redaction(monkeypatch):
    def urlopen(request, timeout):
        assert "test-secret" not in request.full_url
        assert request.get_header("X-goog-api-key") == "test-secret"
        body = json.loads(request.data)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert timeout == 8
        return io.BytesIO(json.dumps({"candidates": [{"finishReason": "STOP", "content": {
            "parts": [{"text": '{"facts": []}'}]}}]}).encode())
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = GeminiMemoryClient("test-secret", "gemini-3.8-flash")
    assert client.chat_json([{"role": "user", "content": "hello"}])[0] == {"facts": []}
    settings = Settings().with_overrides(**{"memory.gemini_api_key": "test-secret"})
    assert "test-secret" not in repr(settings.memory)
    assert "test-secret" not in json.dumps(settings.as_dict())


@pytest.mark.parametrize("error", [TimeoutError("test-secret"),
    urllib.error.HTTPError("https://example/test-secret", 403, "test-secret", {}, None)])
def test_provider_errors_do_not_leak_key(monkeypatch, error):
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError) as exc:
        GeminiMemoryClient("test-secret", "gemini-3.8-flash").chat_json([])
    assert "test-secret" not in str(exc.value)


def test_durable_extraction_rejects_transient_and_unsupported_facts():
    reply = durable_reply()
    reply["facts"] += [dict(reply["facts"][0], durable=False),
                       dict(reply["facts"][0], evidence="I live on Mars")]
    result = extract_facts("I work as an engineer", client=client_reply(reply), use_llm=True)
    assert len(result.facts) == 1
    assert result.facts[0].object_value == "engineer"
    assert result.method == "gemini"


def test_empty_gemini_extraction_overrides_local_guess():
    result = extract_facts("I am Happy", client=client_reply({"facts": [], "correction": False}), use_llm=True)
    assert not result.facts


def test_failure_keeps_local_learning():
    client = client_reply({})
    result = extract_facts("My name is Alex", client=client, use_llm=True)
    assert result.facts[0].object_value == "Alex"


def test_filter_preserves_fresh_exchanges_and_only_known_ids(engine):
    engine.remember_fact(RememberFactRequest(transcript="My name is Alex", use_llm=False))
    facts = engine.store.list_facts()
    history = [m for i in range(4) for m in (
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"})]
    client = client_reply({"fact_ids": ["invented", facts[0].fact_id, facts[0].fact_id],
                           "turn_ids": [0, 99, True]})
    chosen, context = select_context(client, "Who am I?", history, facts, engine.store)
    assert chosen == history[:2] + history[-4:]
    assert [f.fact_id for f in context.retrieved_facts] == [facts[0].fact_id]


def test_filter_failure_keeps_history_and_local_retrieval(engine):
    engine.remember_fact(RememberFactRequest(transcript="My name is Alex", use_llm=False))
    engine.gemini_client = client_reply({})
    history = [{"role": "user", "content": "hello"}]
    chosen, block = engine.prepare_conversation_context("What is my name?", history)
    assert chosen == history
    assert "Alex" in block


def test_gemini_facts_survive_database_reopen(engine):
    engine.llm_client = engine.gemini_client = client_reply(durable_reply())
    engine.settings = replace(engine.settings, auto_learn_facts=True, use_llm_fact_extraction=True)
    engine.maybe_learn_facts("I work as an engineer", "durable-turn")
    settings = engine.settings
    engine.shutdown()
    reopened = MultimodalMemoryEngine(settings)
    try:
        reopened.initialize()
        facts = reopened.store.list_facts()
        assert any(f.object_value == "engineer" and f.source_turn_id == "durable-turn" for f in facts)
    finally:
        reopened.shutdown()


def test_pending_extraction_survives_restart(engine):
    engine.llm_client = engine.gemini_client = client_reply(durable_reply())
    engine.settings = replace(engine.settings, auto_learn_facts=True, use_llm_fact_extraction=True,
                              background_indexing=True)
    engine.record_turn(transcript="I work as an engineer", response="Thanks", turn_id="pending-turn")
    engine.maybe_learn_facts("I work as an engineer", "pending-turn")
    assert any(j.job_type == "extract_facts" for j in engine.store.pending_index_jobs())
    settings = replace(engine.settings, background_indexing=False)
    engine.shutdown()
    reopened = MultimodalMemoryEngine(settings)
    reopened.llm_client = reopened.gemini_client = client_reply(durable_reply())
    try:
        reopened.initialize()
        process_pending_jobs(reopened)
        assert any(f.object_value == "engineer" for f in reopened.store.list_facts())
    finally:
        reopened.shutdown()


def test_offline_disables_gemini(memory_settings):
    settings = replace(memory_settings.memory, gemini_api_key="test-secret")
    engine = MultimodalMemoryEngine(settings, llm_client=object(), allow_cloud=False)
    assert engine.gemini_client is None
    assert engine.llm_client is None
