"""Orchestrator integration: routing, tool dispatch, prompt evidence."""

from __future__ import annotations

import pytest

from config.settings import Settings, reset_settings_cache
from src.intelligence.intent_classifier import DeterministicRouter
from src.intelligence.prompt_builder import build_response_messages
from src.memory.models import RememberFactRequest
from src.memory.retriever import MemoryRetriever
from tests.conftest import FakeGroqClient, make_agent, plan_dict


# ---------------------------------------------------------------------------
# deterministic routing
# ---------------------------------------------------------------------------

class TestMemoryRouting:
    @pytest.fixture
    def route(self):
        return DeterministicRouter().route

    def test_remember_fact(self, route):
        d = route("Please remember that my project is called NATERIDA")
        assert d.plan.intent.value == "remember_fact"
        assert d.plan.tool == "remember_fact"
        assert not d.plan.requires_camera

    def test_correct_memory(self, route):
        d = route("Actually the project is now called NATERIDA-X")
        assert d.plan.intent.value == "correct_memory"
        assert d.plan.tool == "correct_memory"

    def test_learn_identity(self, route):
        d = route("Please remember my face as Alex")
        assert d.plan.intent.value == "learn_identity"
        assert d.plan.tool == "remember_identity"
        assert d.plan.requires_camera
        assert not d.plan.requires_vlm

    def test_learn_object(self, route):
        d = route("Remember this object: my custom motor controller")
        assert d.plan.intent.value == "learn_visual_object"
        assert d.plan.tool == "learn_visual_object"
        assert d.plan.requires_camera
        assert "controller" in d.plan.tool_arguments.get("object_name", "")

    def test_forget(self, route):
        d = route("Forget everything you know about my face")
        assert d.plan.intent.value == "forget_memory"

    def test_list(self, route):
        d = route("What do you remember about me?")
        assert d.plan.intent.value in {"list_memories", "query_memory"}

    def test_query_memory(self, route):
        d = route("What did I tell you about my project?")
        assert d.plan.intent.value == "query_memory"

    def test_who_am_i_uses_face(self, route):
        d = route("Who am I?")
        assert d.plan.requires_camera
        assert d.plan.tool == "recognize_person"
        assert not d.plan.requires_vlm

    def test_plain_chat_not_memory(self, route):
        d = route("What is the capital of France?")
        assert d.plan.intent.value not in {
            "remember_fact",
            "correct_memory",
            "learn_identity",
            "learn_visual_object",
            "forget_memory",
            "query_memory",
            "list_memories",
        }


# ---------------------------------------------------------------------------
# prompt evidence
# ---------------------------------------------------------------------------

class TestPromptInjection:
    def test_memory_block_reaches_response_messages(self):
        from src.orchestrator.schemas import RequestRecord

        record = RequestRecord(transcript="Who am I?")
        record.metadata["memory_block"] = "MEMORY (untrusted):\n- name: Alex"
        messages = build_response_messages(record)
        joined = "\n".join(m["content"] for m in messages)
        assert "MEMORY (untrusted)" in joined

    def test_no_memory_block_no_section(self):
        from src.orchestrator.schemas import RequestRecord

        record = RequestRecord(transcript="Hello")
        messages = build_response_messages(record)
        joined = "\n".join(m["content"] for m in messages)
        assert "MEMORY (untrusted)" not in joined


# ---------------------------------------------------------------------------
# AIAgent end-to-end with a real (tmp) memory engine
# ---------------------------------------------------------------------------

def _memory_settings(tmp_path, **extra):
    overrides = {
        "memory.enabled": True,
        "memory.db_path": str(tmp_path / "mem.db"),
        "memory.asset_dir": str(tmp_path / "assets"),
        "memory.face_recognition_enabled": False,
        "memory.object_grounder": "center-prior",
        "memory.background_indexing": False,
        "memory.auto_learn_facts": False,
        "memory.clip_allow_download": False,
    }
    overrides.update(extra)
    return Settings().with_overrides(**overrides)


@pytest.fixture
def mem_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "1")
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "mem.db"))
    monkeypatch.setenv("MEMORY_ASSET_DIR", str(tmp_path / "assets"))
    reset_settings_cache()
    try:
        plan = plan_dict(
            "remember_fact",
            tool="remember_fact",
            tool_arguments={"transcript": "My robotics project is called NATERIDA"},
        )
        a = make_agent(
            plan=plan,
            settings=_memory_settings(tmp_path),
            response_strategy="off",
        )
        a.start(warm_models=False)
        yield a
        a.shutdown()
    finally:
        reset_settings_cache()


class TestAgentMemory:
    def test_remember_fact_turn_answers_directly(self, mem_agent):
        record = mem_agent.handle_text("Please remember that my project is called NATERIDA")
        assert record.plan.intent.value == "remember_fact"
        assert record.response
        assert record.tool_results and record.tool_results[0].ok
        assert "NATERIDA" in mem_agent.memory_engine.retrieve("project").message

    def test_llm_plan_without_transcript_falls_back_to_turn(self, tmp_path, monkeypatch):
        """LLM routers may omit tool_arguments; memory tools must still work."""
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        reset_settings_cache()
        try:
            # Simulate an LLM plan: correct intent/tool, but no tool_arguments.
            plan = plan_dict("remember_fact", tool="remember_fact", tool_arguments={})
            a = make_agent(
                plan=plan,
                settings=_memory_settings(tmp_path),
                response_strategy="off",
            )
            a.start(warm_models=False)
            try:
                record = a.handle_text("Remember that my project is called ORION")
                assert record.plan.intent.value == "remember_fact"
                assert record.tool_results and record.tool_results[0].ok, (
                    record.tool_results[0].error if record.tool_results else "no result"
                )
                assert "ORION" in a.memory_engine.retrieve("project").message
            finally:
                a.shutdown()
        finally:
            reset_settings_cache()

    def test_engine_status_in_agent_status(self, mem_agent):
        status = mem_agent.status()
        assert status["memory"]["enabled"] is True

    def test_shutdown_stops_engine(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        reset_settings_cache()
        try:
            a = make_agent(settings=_memory_settings(tmp_path))
            a.start(warm_models=False)
            assert a.memory_engine is not None
            a.shutdown()
            assert a.memory_engine.store is None
        finally:
            reset_settings_cache()


class TestOfflineDirectAnswer:
    def test_query_answered_from_memory_without_llm(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        monkeypatch.setenv("AURA_OFFLINE", "1")
        reset_settings_cache()
        try:
            settings = _memory_settings(tmp_path, **{"orchestrator.offline": True})
            plan = plan_dict(
                "query_memory",
                tool="query_memory",
                tool_arguments={"query": "What did I tell you my robotics project was called?"},
            )
            a = make_agent(plan=plan, settings=settings, response_strategy="off")
            a.start(warm_models=False)
            try:
                a.memory_engine.remember_fact(
                    RememberFactRequest(
                        transcript="My robotics project is called NATERIDA", turn_id="t1"
                    )
                )
                record = a.handle_text(
                    "What did I tell you my robotics project was called?"
                )
                assert record.response
                assert "NATERIDA" in record.response
                assert record.response_strategy in {"memory_offline", "direct_answer"}
            finally:
                a.shutdown()
        finally:
            monkeypatch.delenv("AURA_OFFLINE", raising=False)
            reset_settings_cache()
