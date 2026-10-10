"""Conversation memory tests: bounds, separation of observations, context."""

from __future__ import annotations

import time

from src.intelligence.conversation_memory import ConversationMemory
from src.orchestrator.schemas import VisualEvidence


def _evidence(frame_id: str, answer: str, ts: float | None = None) -> VisualEvidence:
    return VisualEvidence(
        frame_id=frame_id,
        question="what is visible?",
        raw_answer=answer,
        status="completed",
        frame_timestamp=ts if ts is not None else time.time(),
    )


def test_records_turns_in_order():
    mem = ConversationMemory(max_turns=5)
    for i in range(3):
        mem.record(user_transcript=f"q{i}", final_response=f"a{i}")
    assert [t.user_transcript for t in mem.turns] == ["q0", "q1", "q2"]
    assert mem.last_response == "a2"


def test_bounded_history_drops_oldest():
    mem = ConversationMemory(max_turns=3)
    for i in range(10):
        mem.record(user_transcript=f"q{i}", final_response=f"a{i}")
    assert len(mem) == 3
    assert mem.turns[0].user_transcript == "q7"


def test_chat_messages_alternate_roles():
    mem = ConversationMemory(max_turns=4)
    mem.record(user_transcript="hi", final_response="hello")
    msgs = mem.chat_messages()
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_visual_observations_kept_separately_and_bounded():
    mem = ConversationMemory(max_turns=4)
    mem.record(user_transcript="what do you see", final_response="a bottle")
    for i in range(10):
        mem.record_evidence(_evidence(f"frame_{i:04d}", f"object {i}"))
    assert len(mem) == 1                      # turns are separate
    assert len(mem.visual_observations) <= 6  # bounded deque
    assert mem.visual_observations[-1].frame_id == "frame_0009"


def test_past_observation_never_becomes_current():
    mem = ConversationMemory(max_turns=4)
    past = _evidence("frame_0001", "a red bottle", ts=time.time() - 600)
    mem.record_evidence(past)
    stored = mem.visual_observations[0]
    assert stored.frame_id == "frame_0001"
    assert stored.frame_timestamp is not None and time.time() - stored.frame_timestamp > 300
    # The memory only reports it as a *previous* observation in context.
    ctx = mem.context_summary()
    assert "Previous observation" in ctx


def test_context_summary_is_bounded():
    mem = ConversationMemory(max_turns=8)
    for i in range(8):
        mem.record(user_transcript=f"question number {i} " + "x" * 200,
                   final_response=f"answer {i} " + "y" * 200)
    assert len(mem.context_summary(max_chars=300)) <= 301


def test_find_frame_for_reference():
    mem = ConversationMemory(max_turns=4)
    mem.record_evidence(_evidence("frame_0001", "a laptop and a red bottle"))
    found = mem.find_frame_for_reference("red bottle")
    assert found is not None and found.frame_id == "frame_0001"
    assert mem.find_frame_for_reference("unicorn") is None


def test_clear():
    mem = ConversationMemory()
    mem.record(user_transcript="a", final_response="b")
    mem.record_evidence(_evidence("f", "x"))
    mem.clear()
    assert len(mem) == 0 and mem.visual_observations == [] and mem.last_response == ""


def test_persistence_is_opt_in(tmp_path):
    path = tmp_path / "history.json"
    mem = ConversationMemory(max_turns=5)          # no persist_path
    mem.record(user_transcript="a", final_response="b")
    assert not path.exists()

    mem2 = ConversationMemory(max_turns=5, persist_path=path)
    mem2.record(user_transcript="a", final_response="b")
    assert path.exists()
    restored = ConversationMemory.load(path, max_turns=5)
    assert len(restored) == 1
    assert restored.turns[0].final_response == "b"
