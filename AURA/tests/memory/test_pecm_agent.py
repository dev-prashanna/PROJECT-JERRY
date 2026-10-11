"""Agent-level PECM integration: tool registry, offline recall, event log."""

from __future__ import annotations

import pytest

from config.settings import Settings, reset_settings_cache
from src.memory.indexing_worker import process_pending_jobs
from src.orchestrator.schemas import Intent
from tests.conftest import FakeGroqClient, make_agent, plan_dict


@pytest.fixture
def pecm_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "1")
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "agent_pecm.db"))
    monkeypatch.setenv("MEMORY_ASSET_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("FACE_RECOGNITION_ENABLED", "0")
    monkeypatch.setenv("OBJECT_MEMORY_ENABLED", "0")
    monkeypatch.setenv("MEMORY_BACKGROUND_INDEXING", "0")
    monkeypatch.setenv("MEMORY_AUTO_LEARN_FACTS", "0")
    monkeypatch.setenv("CONVERSATION_MEMORY_ENABLED", "1")
    monkeypatch.setenv("MEMORY_SEMANTIC_SEARCH_ENABLED", "1")
    monkeypatch.setenv("MEMORY_TEXT_EMBEDDER", "hash")
    reset_settings_cache()
    s = Settings()
    yield s
    reset_settings_cache()


def _seed_turn(agent) -> str:
    """Record a VQA turn directly through the engine (no camera needed)."""
    r = agent.memory_engine.record_turn(
        transcript="What color is the curtain?",
        observation="The curtain appears white.",
        response="The curtain appears white.",
        frame_id="frame_1042",
        vlm_model="moondream2",
    )
    process_pending_jobs(agent.memory_engine)
    return r["turn_id"]


class TestRecallToolRegistration:
    def test_recall_tools_registered(self, pecm_settings):
        a = make_agent(settings=pecm_settings)
        a.start(warm_models=False)
        try:
            assert a.registry.get("recall_memory") is not None
            assert a.registry.get("recall_visual_memory") is not None
        finally:
            a.shutdown()

    def test_memory_tool_dispatches_recall(self, pecm_settings):
        a = make_agent(settings=pecm_settings)
        a.start(warm_models=False)
        try:
            _seed_turn(a)
            result = a.ctx.memory_tool("recall", {"query": "curtain color"})
            assert result.ok
            assert result.tool == "recall_memory"
            assert result.payload.get("events")
        finally:
            a.shutdown()

    def test_memory_tool_dispatches_recall_visual(self, pecm_settings):
        a = make_agent(settings=pecm_settings)
        a.start(warm_models=False)
        try:
            _seed_turn(a)
            result = a.ctx.memory_tool(
                "recall_visual", {"query": "what color was the curtain earlier?"}
            )
            assert result.ok
            assert result.tool == "recall_visual_memory"
            assert "white" in result.text.lower()
            assert result.payload.get("camera_required") is False
        finally:
            a.shutdown()


class TestOfflineRecall:
    """Offline: recall answers directly from the event log — no cloud needed."""

    @pytest.fixture
    def offline_agent(self, pecm_settings):
        s = pecm_settings.with_overrides(**{"orchestrator.offline": True})
        a = make_agent(
            plan=plan_dict(
                "recall_visual_memory",
                tool="recall_visual_memory",
                tool_arguments={"query": "what color was the curtain earlier?"},
            ),
            answer="should not be used offline",
            settings=s,
            router_mode="deterministic",
        )
        a.start(warm_models=False)
        yield a
        a.shutdown()

    def test_offline_recall_answers_from_event_log(self, offline_agent):
        _seed_turn(offline_agent)
        record = offline_agent.handle_text(
            "what color was the curtain earlier?", speak=False
        )
        assert record.error is None
        assert "white" in record.response.lower()
        assert record.response_strategy in {"memory_offline", "memory_direct"}

    def test_offline_recall_no_camera_snapshot(self, offline_agent):
        _seed_turn(offline_agent)
        before = offline_agent.camera.snapshots
        offline_agent.handle_text(
            "what color was the curtain when the camera was off?", speak=False
        )
        assert offline_agent.camera.snapshots == before


class TestOnlineRecallContext:
    """Online: recall attaches episodic context for the LLM generator."""

    def test_online_recall_attaches_episodic_context(self, pecm_settings):
        a = make_agent(
            plan=plan_dict(
                "recall_visual_memory",
                tool="recall_visual_memory",
                tool_arguments={"query": "what color was the curtain earlier?"},
            ),
            answer="The curtain was white, as I observed earlier.",
            settings=pecm_settings,
        )
        a.start(warm_models=False)
        try:
            _seed_turn(a)
            record = a.handle_text(
                "what color was the curtain earlier?", speak=False
            )
            assert record.metadata.get("episodic_context")
            assert "white" in record.metadata["episodic_context"].lower()
            # generator produced the final answer (not the raw event)
            assert record.response == "The curtain was white, as I observed earlier."
        finally:
            a.shutdown()

    def test_episodic_context_reaches_prompt(self, pecm_settings):
        from src.intelligence.prompt_builder import build_response_messages
        from src.orchestrator.schemas import RequestRecord

        a = make_agent(settings=pecm_settings)
        a.start(warm_models=False)
        try:
            _seed_turn(a)
            block = a.memory_engine.build_rag_context("curtain color")
            record = RequestRecord(
                transcript="what color was the curtain?",
                metadata={"episodic_context": block},
            )
            messages = build_response_messages(record)
            assert any("local memory" in (m.get("content") or "") for m in messages)
        finally:
            a.shutdown()


class TestDurableEventRecordingViaAgent:
    def test_handle_text_records_events(self, pecm_settings):
        a = make_agent(
            plan=plan_dict("general_question"),
            answer="France's capital is Paris.",
            settings=pecm_settings,
        )
        a.start(warm_models=False)
        try:
            a.handle_text("What is the capital of France?", speak=False)
            store = a.memory_engine.store
            assert store.count_events() >= 2
            types = {e.event_type.value for e in store.recent_events(limit=10)}
            assert "user_transcript" in types
            assert "assistant_response" in types
        finally:
            a.shutdown()

    def test_vqa_turn_records_observation(self, pecm_settings):
        a = make_agent(
            plan=plan_dict(
                "visual_question",
                tool="analyze_camera",
                tool_arguments={"question": "what color is the bottle?"},
            ),
            answer="The bottle is blue.",
            settings=pecm_settings,
        )
        a.ctx._vlm.answer = "the bottle is blue"
        a.start(warm_models=False)
        try:
            record = a.handle_text("What color is the bottle?", speak=False)
            store = a.memory_engine.store
            types = {e.event_type.value for e in store.recent_events(limit=10)}
            assert "vlm_observation" in types
            obs = next(
                e for e in store.recent_events(limit=10)
                if e.event_type.value == "vlm_observation"
            )
            assert obs.frame_id
            assert "blue" in obs.content.lower()
        finally:
            a.shutdown()


class TestStatusIncludesPecm:
    def test_status_reports_conversation_fields(self, pecm_settings):
        a = make_agent(settings=pecm_settings)
        a.start(warm_models=False)
        try:
            status = a.memory_engine.status()
            assert status["conversation_enabled"] is True
            assert status["episodic_enabled"] is True
            assert status["text_embedder"] == "hash-embed-v1"
            assert status["schema_version"] >= 3
        finally:
            a.shutdown()
