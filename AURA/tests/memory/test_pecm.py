"""PECM acceptance tests — durable events, hybrid recall, camera-free history.

Covers the curtain-colour scenario (Steps 1-9) and the remaining acceptance
criteria: idempotent writes, index-pending status, deletion, RAG context,
routing without camera, and the two-level retrieval model.
"""

from __future__ import annotations

import pytest

from config.settings import Settings, reset_settings_cache
from src.memory.engine import MultimodalMemoryEngine
from src.memory.event_models import (
    ConversationEvent,
    EventType,
    RecallStatus,
)
from src.memory.indexing_worker import process_pending_jobs
from src.memory.temporal_resolver import parse_temporal


@pytest.fixture
def pecm_engine(tmp_path):
    reset_settings_cache()
    s = Settings().with_overrides(
        **{
            "memory.enabled": True,
            "memory.db_path": str(tmp_path / "pecm.db"),
            "memory.asset_dir": str(tmp_path / "assets"),
            "memory.face_recognition_enabled": False,
            "memory.object_memory_enabled": False,
            "memory.background_indexing": False,
            "memory.auto_learn_facts": False,
            "memory.conversation_memory_enabled": True,
            "memory.episodic_memory_enabled": True,
            "memory.memory_semantic_search_enabled": True,
            "memory.memory_text_embedder": "hash",
            "memory.memory_allow_historical_vision_recall": True,
        }
    )
    eng = MultimodalMemoryEngine(s.memory)
    eng.initialize()
    yield eng
    eng.shutdown()
    reset_settings_cache()


def _curtain_turn(engine) -> dict:
    """Steps 1-2: live VQA turn that must be persisted."""
    return engine.record_turn(
        transcript="What color is the curtain?",
        observation="The curtain appears white.",
        response="The curtain appears white.",
        frame_id="frame_1042",
        vlm_model="moondream2",
    )


class TestCurtainScenario:
    """The end-to-end acceptance scenario from the PECM spec."""

    def test_step1_turn_is_persisted(self, pecm_engine):
        r = _curtain_turn(pecm_engine)
        assert r["turn_id"]
        assert len(r["events"]) == 3
        assert all(ev["ok"] for ev in r["events"])
        assert pecm_engine.store.count_events() == 3

    def test_step2_events_have_correct_types_and_frame(self, pecm_engine):
        r = _curtain_turn(pecm_engine)
        events = pecm_engine.store.events_for_turn(r["turn_id"])
        types = [e.event_type for e in events]
        assert EventType.USER_TRANSCRIPT in types
        assert EventType.VLM_OBSERVATION in types
        assert EventType.ASSISTANT_RESPONSE in types
        obs = next(e for e in events if e.event_type == EventType.VLM_OBSERVATION)
        assert obs.frame_id == "frame_1042"
        assert obs.source_model == "moondream2"

    def test_step3_fresh_recall_before_embeddings(self, pecm_engine):
        """Level-1 retrieval must work before any embedding job runs."""
        _curtain_turn(pecm_engine)
        # Do NOT process pending jobs — embeddings are still pending
        assert pecm_engine.store.count_pending_index_jobs() > 0
        result = pecm_engine.recall_last_visual_observation(
            "what color was the curtain earlier?"
        )
        assert result.ok
        assert result.payload["status"] == RecallStatus.FOUND.value
        assert "white" in result.message.lower()
        assert result.payload["camera_required"] is False

    def test_step4_episode_built_after_indexing(self, pecm_engine):
        r = _curtain_turn(pecm_engine)
        process_pending_jobs(pecm_engine)
        ep = pecm_engine.store.episode_for_turn(r["turn_id"])
        assert ep is not None
        assert "white" in ep.episode_summary.lower()
        assert ep.frame_id == "frame_1042"
        assert ep.episode_type.value == "visual_observation"

    def test_step5_semantic_search_after_embedding(self, pecm_engine):
        _curtain_turn(pecm_engine)
        process_pending_jobs(pecm_engine)
        assert pecm_engine.text_index is not None
        assert len(pecm_engine.text_index) > 0
        result = pecm_engine.search_conversations("curtain colour")
        assert result.ok
        assert result.payload["events"]
        assert result.payload["status"] == RecallStatus.FOUND.value

    def test_step6_historical_recall_with_camera_off(self, pecm_engine):
        """Camera unavailable → answer from the durable event log."""
        _curtain_turn(pecm_engine)
        process_pending_jobs(pecm_engine)
        result = pecm_engine.recall_last_visual_observation(
            "what color was the curtain when the camera was off?"
        )
        assert result.ok
        assert "white" in result.message.lower()
        assert result.payload["status"] == RecallStatus.FOUND.value

    def test_step7_rag_context_block(self, pecm_engine):
        _curtain_turn(pecm_engine)
        process_pending_jobs(pecm_engine)
        block = pecm_engine.build_rag_context("what color was the curtain?")
        assert block
        assert "white" in block.lower()
        assert "local memory" in block.lower()

    def test_step8_no_false_positive_on_unknown_query(self, pecm_engine):
        result = pecm_engine.recall_last_visual_observation(
            "what color was the sofa yesterday?"
        )
        # May fall back to the most recent observation (curtain) or not-found;
        # must never fabricate a sofa colour.
        if result.ok and result.payload["status"] == RecallStatus.FOUND.value:
            assert "sofa" not in result.message.lower()

    def test_step9_second_turn_and_delete(self, pecm_engine):
        r1 = _curtain_turn(pecm_engine)
        r2 = pecm_engine.record_turn(transcript="thanks", response="you're welcome")
        # 3 events from the VQA turn + 2 from the follow-up (no observation)
        assert pecm_engine.store.count_events() == 5
        ev_id = r2["events"][0]["event_id"]
        d = pecm_engine.delete_conversation(event_id=ev_id)
        assert d.ok
        assert pecm_engine.store.count_events() == 4
        # first turn still intact
        assert pecm_engine.store.events_for_turn(r1["turn_id"])


class TestDurableWritePath:
    def test_event_committed_before_index_jobs(self, pecm_engine):
        r = _curtain_turn(pecm_engine)
        events = pecm_engine.store.events_for_turn(r["turn_id"])
        assert events  # committed immediately
        # jobs queued but not yet run
        assert pecm_engine.store.count_pending_index_jobs() >= 2

    def test_duplicate_turn_id_is_idempotent(self, pecm_engine):
        r = _curtain_turn(pecm_engine)
        # recording the same event_id again is a no-op at the store level
        from src.memory.event_models import ConversationEvent

        event = pecm_engine.store.events_for_turn(r["turn_id"])[0]
        again = ConversationEvent(
            event_id=event.event_id,
            session_id=event.session_id,
            turn_id=event.turn_id,
            event_sequence=event.event_sequence,
            event_type=event.event_type,
            content=event.content,
        )
        assert pecm_engine.store.insert_event(again) is False
        assert pecm_engine.store.count_events() == 3

    def test_record_event_accepts_string_type(self, pecm_engine):
        ack = pecm_engine.record_event("user_text", "hello there")
        assert ack["ok"]
        assert ack["event_id"]

    def test_empty_content_rejected(self, pecm_engine):
        ack = pecm_engine.record_event(EventType.USER_TRANSCRIPT, "   ")
        assert ack["ok"] is False


class TestRecallStatuses:
    def test_not_found_when_store_empty(self, pecm_engine):
        result = pecm_engine.recall_last_visual_observation("what color is anything?")
        assert result.ok
        assert result.payload["status"] in {
            RecallStatus.NOT_FOUND.value,
            RecallStatus.INDEX_PENDING.value,
        }

    def test_storage_unavailable(self, pecm_engine):
        pecm_engine.store = None
        result = pecm_engine.search_conversations("anything")
        assert not result.ok

    def test_historical_recall_disabled(self, tmp_path):
        reset_settings_cache()
        s = Settings().with_overrides(
            **{
                "memory.enabled": True,
                "memory.db_path": str(tmp_path / "off.db"),
                "memory.asset_dir": str(tmp_path / "assets"),
                "memory.face_recognition_enabled": False,
                "memory.object_memory_enabled": False,
                "memory.background_indexing": False,
                "memory.memory_allow_historical_vision_recall": False,
                "memory.conversation_memory_enabled": True,
            }
        )
        eng = MultimodalMemoryEngine(s.memory)
        eng.initialize()
        try:
            result = eng.recall_last_visual_observation("what did you see?")
            assert not result.ok
        finally:
            eng.shutdown()
            reset_settings_cache()


class TestTemporalResolver:
    @pytest.mark.parametrize("phrase", [
        "what color was the curtain earlier?",
        "what did you see before?",
        "what color was the curtain when the camera was off?",
        "repeat what you saw",
        "what was on the table last time?",
    ])
    def test_history_phrases_detected(self, phrase):
        hint = parse_temporal(phrase)
        assert hint.refers_to_history

    def test_last_observation_detected(self):
        hint = parse_temporal("what did you last see?")
        assert hint.refers_to_last_observation

    def test_previous_response_detected(self):
        hint = parse_temporal("what did you just say?")
        assert hint.refers_to_previous_response

    def test_recent_turn_detected(self):
        hint = parse_temporal("what happened just now?")
        assert hint.refers_to_recent_turn

    def test_plain_question_has_no_history_hint(self):
        hint = parse_temporal("what is the capital of France?")
        assert not hint.refers_to_history
        assert not hint.refers_to_last_observation


class TestHybridFusion:
    def test_rrf_prefers_items_ranked_by_multiple_sources(self):
        from src.memory.hybrid_search import rrf_fuse

        a, b, c = "a", "b", "c"
        fused = rrf_fuse(
            {"keyword": [a, b, c], "semantic": [b, a, c], "recent": [c, b, a]},
            k=60,
        )
        ids = [item for item, _score, _meta in fused]
        # b is #2 in keyword, #1 in semantic, #2 in recent → strongest overall
        assert ids[0] == "b"

    def test_rrf_weights_scale_contribution(self):
        from src.memory.hybrid_search import rrf_fuse

        fused = rrf_fuse(
            {"recent": ["x"], "keyword": ["y"]},
            weights={"recent": 3.0, "keyword": 1.0},
        )
        assert fused[0][0] == "x"


class TestRoutingWithoutCamera:
    """Deterministic router must send historical questions to recall tools."""

    def _route(self, transcript: str):
        from src.intelligence.intent_classifier import DeterministicRouter

        return DeterministicRouter().route(transcript)

    def test_historical_visual_question_routes_to_recall(self):
        d = self._route("what color was the curtain earlier?")
        assert d.plan.tool == "recall_visual_memory"
        assert d.plan.requires_camera is False
        assert d.plan.requires_vlm is False

    def test_historical_conversation_routes_to_recall(self):
        d = self._route("what did you say earlier about the project?")
        assert d.plan.tool == "recall_memory"
        assert d.plan.requires_camera is False

    def test_camera_off_phrase_routes_to_recall(self):
        d = self._route("what did you see when the camera was off?")
        assert d.plan.tool == "recall_visual_memory"
        assert d.plan.requires_camera is False

    def test_live_visual_question_still_uses_camera(self):
        d = self._route("what do you see right now?")
        assert d.plan.requires_camera is True
        assert d.plan.tool in {"analyze_camera", "describe_scene"}


class TestDeletion:
    def test_delete_session_clears_events_and_episodes(self, pecm_engine):
        r = _curtain_turn(pecm_engine)
        process_pending_jobs(pecm_engine)
        session_id = pecm_engine.store.events_for_turn(r["turn_id"])[0].session_id
        d = pecm_engine.delete_conversation(session_id=session_id)
        assert d.ok
        assert pecm_engine.store.count_events() == 0

    def test_delete_all_history(self, pecm_engine):
        _curtain_turn(pecm_engine)
        pecm_engine.record_turn(transcript="hello", response="hi")
        d = pecm_engine.delete_conversation(all_history=True)
        assert d.ok
        assert pecm_engine.store.count_events() == 0
        # FTS cleared too
        assert pecm_engine.store.search_events_fts("curtain") == []
