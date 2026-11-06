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

    def test_remember_face_and_name_in_one_command(self, route):
        d = route("Remember my face, my name is Prashanna")
        assert d.plan.intent.value == "learn_identity"
        assert d.plan.tool == "remember_identity"
        assert d.plan.requires_camera

    def test_remember_name_and_face(self, route):
        d = route("Remember my name and face")
        assert d.plan.intent.value == "learn_identity"
        assert d.plan.tool == "remember_identity"

    def test_bare_name_statement_stays_a_fact(self, route):
        d = route("My name is Prashanna")
        assert d.plan.intent.value == "remember_fact"
        assert d.plan.tool == "remember_fact"
        assert not d.plan.requires_camera

    def test_asking_about_face_is_not_enrollment(self, route):
        d = route("Do you remember my face?")
        assert d.plan.intent.value == "query_memory"
        assert d.plan.tool == "recognize_person"
        assert d.plan.requires_camera

    def test_enable_face_recognition_reports_status(self, route):
        d = route("Enable face recognition.")
        assert d.plan.intent.value == "system_status"
        assert d.plan.tool == "get_system_status"

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
    def test_router_prompt_documents_face_enrollment(self):
        from src.intelligence.prompt_builder import ROUTER_SYSTEM

        assert "remember_identity" in ROUTER_SYSTEM
        assert "entity_name" in ROUTER_SYSTEM
        # a bare name statement must not silently become an enrollment
        assert "stays remember_fact" in ROUTER_SYSTEM

    def test_router_prompt_routes_face_toggle_to_status(self):
        from src.intelligence.prompt_builder import ROUTER_SYSTEM

        assert "face recognition" in ROUTER_SYSTEM.lower()
        assert "get_system_status" in ROUTER_SYSTEM

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


# ---------------------------------------------------------------------------
# memory error messages
# ---------------------------------------------------------------------------

class TestMemoryErrorMessage:
    def test_engine_unavailable_is_not_blamed_on_face(self):
        from src.orchestrator.agent import _memory_error_message
        from src.orchestrator.schemas import ToolResult

        result = ToolResult.failure(
            "remember_fact", "multimodal memory is not enabled or not initialized"
        )
        message = _memory_error_message(result)
        assert "memory system" in message.lower()
        assert "face recognition" not in message.lower()

    def test_face_disabled_still_says_face(self):
        from src.orchestrator.agent import _memory_error_message
        from src.orchestrator.schemas import ToolResult

        result = ToolResult.failure(
            "remember_identity", "face recognition is not enabled"
        )
        assert "face recognition" in _memory_error_message(result).lower()


# ---------------------------------------------------------------------------
# face enrollment: name fact → offer → "yes" → enrollment
# ---------------------------------------------------------------------------

@pytest.fixture
def face_agent(tmp_path, monkeypatch):
    from src.memory.face_memory import FaceMatchThresholds, FaceMemoryService
    from tests.conftest import FakeCamera
    from tests.memory.conftest import FakeFaceBackend, face_image

    monkeypatch.setenv("MEMORY_ENABLED", "1")
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "mem.db"))
    monkeypatch.setenv("MEMORY_ASSET_DIR", str(tmp_path / "assets"))
    reset_settings_cache()
    try:
        plan = plan_dict("remember_fact", tool="remember_fact", tool_arguments={})
        a = make_agent(
            plan=plan,
            camera=FakeCamera(face_image()),
            settings=_memory_settings(tmp_path),
            response_strategy="off",
        )
        a.start(warm_models=False)
        # Face recognition is off in these settings; inject the fake backend
        # so the enrollment path runs without ONNX weights.
        a.memory_engine.face_service = FaceMemoryService(
            a.memory_engine.store, FakeFaceBackend(), FaceMatchThresholds()
        )
        yield a
        a.shutdown()
    finally:
        reset_settings_cache()


class TestFaceEnrollmentFlow:
    def test_name_statement_offers_face_enrollment(self, face_agent):
        record = face_agent.handle_text("My name is Prashanna")
        assert record.plan.intent.value == "remember_fact"
        assert "Prashanna" in record.response
        assert "face" in record.response.lower()
        assert face_agent._pending_face_enrollment == "Prashanna"

    def test_affirmation_enrolls_the_face(self, face_agent):
        face_agent.handle_text("My name is Prashanna")
        record = face_agent.handle_text("yes")

        assert record.plan.intent.value == "learn_identity"
        assert record.plan.tool == "remember_identity"
        assert record.plan.tool_arguments["entity_name"] == "Prashanna"
        assert record.tool_results, "enrollment tool never ran"
        result = record.tool_results[0]
        assert result.ok, result.error
        assert "prashanna" in record.response.lower()
        assert face_agent._pending_face_enrollment is None

        enrolled = [
            p["name"].lower() for p in face_agent.memory_engine.face_service.list_enrolled()
        ]
        assert "prashanna" in enrolled

    def test_other_answer_drops_the_offer(self, face_agent):
        face_agent.handle_text("My name is Prashanna")
        record = face_agent.handle_text("What is the capital of France?")
        assert face_agent._pending_face_enrollment is None
        assert record.plan.intent.value != "learn_identity"

    def test_third_person_name_does_not_offer(self, face_agent):
        record = face_agent.handle_text("The person's name is Ninja")
        assert face_agent._pending_face_enrollment is None
        assert "face" not in record.response.lower()


# ---------------------------------------------------------------------------
# memory tool failures must keep the *tool* name (not the engine operation)
# ---------------------------------------------------------------------------

class TestMemoryToolFailureNaming:
    @pytest.fixture
    def face_agent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        reset_settings_cache()
        try:
            plan = plan_dict(
                "learn_identity",
                tool="remember_identity",
                tool_arguments={"entity_name": "Prashanna"},
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

    @staticmethod
    def _stub_snapshot(agent, monkeypatch):
        import numpy as np

        monkeypatch.setattr(
            agent.ctx,
            "_memory_snapshot",
            lambda mode="fresh": np.zeros((64, 64, 3), dtype=np.uint8),
        )

    def test_enroll_failure_keeps_the_tool_name(self, face_agent, monkeypatch):
        self._stub_snapshot(face_agent, monkeypatch)
        result = face_agent.ctx.memory_tool(
            "enroll_person", {"entity_name": "Prashanna"}
        )
        assert not result.ok
        assert result.tool == "remember_identity"

    def test_recognize_failure_keeps_the_tool_name(self, face_agent, monkeypatch):
        self._stub_snapshot(face_agent, monkeypatch)
        result = face_agent.ctx.memory_tool("recognize_faces", {})
        assert not result.ok
        assert result.tool == "recognize_person"

    def test_failed_enrollment_answers_a_friendly_error(self, face_agent, monkeypatch):
        """The routed plan matches the failed result, so no LLM round trip."""
        self._stub_snapshot(face_agent, monkeypatch)
        record = face_agent.handle_text("yes")
        assert record.tool_results, record.error
        result = record.tool_results[0]
        assert result.tool == "remember_identity"
        assert not result.ok
        assert record.response_strategy == "memory_error"
        assert record.response


# ---------------------------------------------------------------------------
# fact extraction: the engine must actually reach the LLM, and never leak
# extractor diagnostics into the spoken reply
# ---------------------------------------------------------------------------

class TestFactExtractionWiring:
    def test_memory_engine_shares_the_chat_client(self, mem_agent):
        """Without a client the engine only ever runs deterministic patterns."""
        assert mem_agent.memory_engine is not None
        assert mem_agent.memory_engine.llm_client is mem_agent.generator.client

    def test_remember_fact_asks_the_llm_to_extract(self, mem_agent):
        record = mem_agent.handle_text("Please remember that my project is called NATERIDA")
        assert record.tool_results and record.tool_results[0].ok
        calls = mem_agent.generator.client.calls
        fact_calls = [
            c
            for c in calls
            if c.get("messages")
            and "extract durable facts" in c["messages"][0].get("content", "")
        ]
        assert fact_calls, "the engine never asked the LLM to extract facts"

    def test_nothing_stored_never_leaks_extractor_diagnostics(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        reset_settings_cache()
        try:
            plan = plan_dict(
                "remember_fact",
                tool="remember_fact",
                tool_arguments={"transcript": "zorp quibble wobble nebula"},
            )
            a = make_agent(
                plan=plan,
                settings=_memory_settings(tmp_path),
                answer="I did not catch anything durable in that.",
            )
            a.start(warm_models=False)
            try:
                record = a.handle_text("zorp quibble wobble nebula")
                assert record.tool_results, record.error
                assert record.tool_results[0].ok
                spoken = (record.response or "").lower()
                assert record.response
                assert "deterministic" not in spoken
                assert "skipped" not in spoken
                # the generator is told nothing was stored, so it cannot claim
                # a successful save
                assert "MEMORY WRITE NOTE" in record.metadata.get("memory_note", "")
            finally:
                a.shutdown()
        finally:
            reset_settings_cache()


def test_response_temperature_env_override(monkeypatch):
    from config.settings import get_settings

    monkeypatch.setenv("LLM_RESPONSE_TEMPERATURE", "0.35")
    reset_settings_cache()
    try:
        assert get_settings().llm.response_temperature == 0.35
    finally:
        reset_settings_cache()


class TestRememberFactReplies:
    def test_robot_name_fact_reads_back_naturally(self, mem_agent):
        from src.orchestrator.schemas import RoutePlan, ToolResult

        result = ToolResult(
            tool="remember_fact",
            ok=True,
            text="stored 1 fact(s)",
            payload={
                "stored": [
                    {
                        "predicate": "name",
                        "object_value": "Jerry",
                        "subject_name": "assistant",
                    }
                ],
                "corrected": [],
            },
        )
        plan = RoutePlan(
            **plan_dict(
                "remember_fact",
                tool="remember_fact",
                tool_arguments={"transcript": "Remember that your name is Jerry"},
            )
        )
        reply = mem_agent._remember_fact_reply(result, plan)
        assert reply == "Got it — you can call me Jerry."
        # AURA's name, not the speaker's: no face-enrolment offer.
        assert mem_agent._pending_face_enrollment is None

    def test_nothing_stored_returns_an_empty_reply(self, mem_agent):
        from src.orchestrator.schemas import RoutePlan, ToolResult

        result = ToolResult(
            tool="remember_fact",
            ok=True,
            text="no deterministic fact pattern matched",
            payload={"stored": [], "corrected": []},
        )
        plan = RoutePlan(
            **plan_dict(
                "remember_fact",
                tool="remember_fact",
                tool_arguments={"transcript": "zorp quibble wobble"},
            )
        )
        assert mem_agent._remember_fact_reply(result, plan) == ""

    @pytest.mark.parametrize("subject", ["assistant", "self", "aura"])
    def test_robot_name_subject_variants(self, mem_agent, subject):
        from src.orchestrator.schemas import RoutePlan, ToolResult

        result = ToolResult(
            tool="remember_fact",
            ok=True,
            payload={
                "stored": [
                    {
                        "predicate": "name",
                        "object_value": "Jerry",
                        "subject_name": subject,
                    }
                ],
                "corrected": [],
            },
        )
        plan = RoutePlan(
            **plan_dict(
                "remember_fact",
                tool="remember_fact",
                tool_arguments={"transcript": "Remember that your name is Jerry"},
            )
        )
        assert mem_agent._remember_fact_reply(result, plan) == "Got it — you can call me Jerry."

    def test_addressed_to_aura_beats_an_unknown_subject(self, mem_agent):
        from src.orchestrator.schemas import RoutePlan, ToolResult

        result = ToolResult(
            tool="remember_fact",
            ok=True,
            payload={
                "stored": [
                    {
                        "predicate": "name",
                        "object_value": "Jerry",
                        "subject_name": "some odd subject",
                    }
                ],
                "corrected": [],
            },
        )
        plan = RoutePlan(
            **plan_dict(
                "remember_fact",
                tool="remember_fact",
                tool_arguments={"transcript": "Your name is Jerry"},
            )
        )
        assert mem_agent._remember_fact_reply(result, plan) == "Got it — you can call me Jerry."

    def test_memory_note_reaches_the_response_messages(self):
        from src.orchestrator.schemas import RequestRecord

        record = RequestRecord(transcript="zorp quibble")
        record.metadata["memory_note"] = "MEMORY WRITE NOTE: nothing durable"
        messages = build_response_messages(record)
        assert "MEMORY WRITE NOTE" in messages[-1]["content"]
