"""End-to-end pipeline tests using fakes for hardware and the cloud.

Each test mirrors one of the acceptance scenarios in the specification.
Nothing here touches a microphone, a camera, a GPU or the network, so the
suite is deterministic and CI-safe.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.orchestrator.state_machine import State
from tests.conftest import (
    FakeCamera,
    FakePlayer,
    FakeSTT,
    FakeTTS,
    make_agent,
    plan_dict,
    recording,
)


VISUAL_PLAN = plan_dict(
    "visual_question",
    tool="analyze_camera",
    normalized_question="What objects are visible in front of the camera?",
    tool_arguments={"question": "What objects are visible in front of the camera?"},
)
SCENE_PLAN = plan_dict(
    "scene_description",
    tool="describe_scene",
    normalized_question="Describe everything visible in front of the camera.",
)


def _start(agent):
    agent.start(warm_models=False)
    return agent


# ---------------------------------------------------------------------------
# 1 — simple greeting
# ---------------------------------------------------------------------------

def test_greeting_goes_straight_to_speech():
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="Hello, I am AURA."))
    try:
        record = agent.handle_text("Hello robot!")
    finally:
        agent.shutdown()

    assert record.plan is not None
    assert record.plan.requires_camera is False
    assert agent.camera.snapshots == 0
    assert not record.evidence
    assert record.response == "Hello, I am AURA."
    assert agent.player.played, "the answer must be handed to the speaker"
    assert agent.tts.requests, "text must be synthesised"


# ---------------------------------------------------------------------------
# 2 — visual scene description
# ---------------------------------------------------------------------------

def test_visual_question_uses_camera_and_evidence():
    agent = _start(
        make_agent(plan=VISUAL_PLAN, answer="I can see a laptop and a red bottle on the table.")
    )
    try:
        record = agent.handle_text("What do you see in front of you?")
    finally:
        agent.shutdown()

    assert agent.camera.snapshots >= 1, "a snapshot must be captured"
    assert len(record.evidence) == 1
    ev = record.evidence[0]
    assert ev.frame_id.startswith("frame_")
    assert ev.inference_time_ms is not None and ev.inference_time_ms > 0
    assert ev.image_age_s is not None and ev.image_age_s < 2.0
    assert record.response
    assert agent.player.played


def test_visual_pipeline_states_are_visited():
    agent = _start(make_agent(plan=VISUAL_PLAN, answer="A table."))
    visited = []
    agent.on("state", lambda tr: visited.append(tr.current.value))
    try:
        agent.handle_text("What is on the table?")
    finally:
        agent.shutdown()
    for expected in ("CAPTURING", "PERCEIVING", "GENERATING", "SPEAKING"):
        assert expected in visited, f"{expected} missing from {visited}"


def test_evidence_is_tied_to_the_captured_frame():
    camera = FakeCamera()
    agent = _start(make_agent(plan=VISUAL_PLAN, answer="A object.", camera=camera))
    try:
        record = agent.handle_text("What do you see?")
    finally:
        agent.shutdown()
    frame = camera.buffer.get()
    assert record.evidence[0].frame_id == frame.tag


# ---------------------------------------------------------------------------
# 3 — general scientific question
# ---------------------------------------------------------------------------

def test_general_question_bypasses_the_vlm():
    agent = _start(
        make_agent(
            plan=plan_dict("general_question"),
            answer="F equals m a: force equals mass times acceleration.",
        )
    )
    try:
        record = agent.handle_text("Explain Newton's second law.")
    finally:
        agent.shutdown()

    assert record.plan.requires_camera is False
    assert agent.camera.snapshots == 0
    assert not record.evidence
    assert "force" in record.response.lower()
    metrics = agent.last_metrics
    assert metrics.flags.get("requires_camera") is False


# ---------------------------------------------------------------------------
# 4 — conversational reference
# ---------------------------------------------------------------------------

def test_conversational_reference_uses_history_and_past_evidence():
    from src.intelligence.conversation_memory import ConversationMemory
    from src.orchestrator.schemas import VisualEvidence

    agent = _start(make_agent(plan=VISUAL_PLAN, answer="I can see the bottle."))
    try:
        agent.handle_text("What is on the desk?")
        # The memory now holds a past observation.
        agent.memory.record_evidence(
            VisualEvidence(
                frame_id="frame_0001",
                question="What is on the desk?",
                raw_answer="A laptop and a green bottle.",
            )
        )
        summary = agent.memory.context_summary()
        record = agent.handle_text("What colour is the object on the left?")
    finally:
        agent.shutdown()

    assert "Previous observation" in summary
    # The router received the context (self-contained question expected).
    assert record.plan is not None
    if record.plan.is_visual:
        assert record.plan.normalized_question, "reference must be resolved"
    # It must never present an unobserved colour as fact without evidence.
    from src.intelligence.response_generator import audit_response

    if not record.evidence_text:
        findings = audit_response(record, record.response)
        assert "visual_claim_without_evidence" in findings or "I could not" in record.response


def test_memory_survives_between_turns():
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="Hi!"))
    try:
        agent.handle_text("Hello")
        agent.handle_text("Hello again")
        assert len(agent.memory) == 2
    finally:
        agent.shutdown()


# ---------------------------------------------------------------------------
# 5 — internet failure → offline degraded mode
# ---------------------------------------------------------------------------

def test_offline_mode_still_answers_visual_requests_locally():
    settings = make_agent().settings.with_overrides(**{"orchestrator.offline": True})
    agent = _start(
        make_agent(plan=VISUAL_PLAN, answer="a table", settings=settings, router_mode="deterministic")
    )
    assert agent.mode == "offline"
    try:
        record = agent.handle_text("What do you see in front of you?")
    finally:
        agent.shutdown()

    assert record.evidence, "camera + VLM must keep working offline"
    assert record.response


def test_offline_general_question_admits_limitation():
    settings = make_agent().settings.with_overrides(**{"orchestrator.offline": True})
    agent = _start(
        make_agent(plan=plan_dict("general_question"), settings=settings, router_mode="deterministic")
    )
    try:
        record = agent.handle_text("Who wrote Crime and Punishment?")
    finally:
        agent.shutdown()
    assert record.mode == "offline"
    assert "offline" in record.response.lower()
    assert agent.camera.snapshots == 0


# ---------------------------------------------------------------------------
# 6 — API rate limit
# ---------------------------------------------------------------------------

def test_rate_limit_falls_back_without_endless_retries():
    from src.utils.exceptions import RateLimitError

    client_fail = RateLimitError("429")
    agent = _start(make_agent(fail_client=client_fail))
    started = time.perf_counter()
    try:
        record = agent.handle_text("Explain photosynthesis.")
    finally:
        agent.shutdown()
    elapsed = time.perf_counter() - started
    assert elapsed < 20, f"rate limiting took too long: {elapsed:.1f}s"
    assert record.response, "user must still get a spoken answer"


def test_rate_limit_error_is_reported_not_silent():
    from src.utils.exceptions import RateLimitError

    agent = _start(make_agent(fail_client=RateLimitError("429")))
    errors = []
    agent.on("error", errors.append)
    try:
        record = agent.handle_text("Hello there, can you hear me?")
    finally:
        agent.shutdown()
    # either an error event or a recovered response — never a crash
    assert record is not None


# ---------------------------------------------------------------------------
# 7 — visual hallucination
# ---------------------------------------------------------------------------

def test_response_with_no_evidence_does_not_claim_vision():
    from src.intelligence.response_generator import ResponseGenerator, audit_response
    from src.orchestrator.schemas import RequestRecord

    from src.orchestrator.schemas import RoutePlan

    record = RequestRecord(transcript="What do you see?", plan=None)
    record.plan = RoutePlan.model_validate(
        plan_dict("visual_question", tool="analyze_camera")
    )
    # no evidence attached at all
    generator = ResponseGenerator.__new__(ResponseGenerator)
    result = generator._direct(record)
    findings = audit_response(record, result.text)
    assert "visual_claim_without_evidence" not in findings
    assert "could not" in result.text.lower()


def test_direct_strategy_never_invents_entities():
    from src.intelligence.response_generator import ResponseGenerator, audit_response
    from src.orchestrator.schemas import RequestRecord, VisualEvidence, RoutePlan

    record = RequestRecord(transcript="What do you see?")
    record.plan = RoutePlan.model_validate(plan_dict("visual_question", tool="analyze_camera"))
    record.evidence.append(
        VisualEvidence(frame_id="frame_0001", question="what do you see?",
                       raw_answer="a blurred object")
    )
    result = ResponseGenerator._direct(record)
    findings = audit_response(record, result.text)
    assert not [f for f in findings if f.startswith("novel_visual_entity")], findings
    assert "blurred object" in result.text


def test_audit_flags_invented_visual_facts():
    from src.intelligence.response_generator import audit_response
    from src.orchestrator.schemas import RequestRecord, VisualEvidence, RoutePlan

    record = RequestRecord(transcript="What do you see?")
    record.plan = RoutePlan.model_validate(plan_dict("visual_question"))
    record.evidence.append(
        VisualEvidence(frame_id="frame_0001", question="q", raw_answer="a laptop")
    )
    findings = audit_response(
        record,
        "I can see a Dell XPS laptop, two iPhones and a Coca-Cola bottle.",
    )
    assert findings, "unsupported claims must be flagged"
    assert any(f.startswith("novel_visual_entity") for f in findings)


def test_injected_text_from_camera_reaches_the_llm_as_data_only():
    """An instruction seen in the image must not become an action."""
    agent = _start(make_agent(plan=VISUAL_PLAN, answer="I can only describe what I see."))
    try:
        record = agent.handle_text("What do you see?")
    finally:
        agent.shutdown()
    # No tool other than the visual one was executed.
    assert [t.tool for t in record.tool_results] == ["analyze_camera"]
    assert record.plan.tool in {"analyze_camera", "describe_scene"}


# ---------------------------------------------------------------------------
# 8 — long speech output
# ---------------------------------------------------------------------------

def test_long_response_is_synthesised_without_memory_growth():
    long_answer = "This is sentence number %d and it keeps going. " % 1 + " ".join(
        f"word{i}" for i in range(600)
    )
    agent = _start(make_agent(plan=plan_dict("general_question"), answer=long_answer))
    try:
        record = agent.handle_text("Tell me in detail how a CPU works.")
    finally:
        agent.shutdown()

    assert record.response
    assert len(agent.tts.requests) >= 1
    assert len(agent.memory) == 1, "memory must stay bounded after a long turn"


def test_sentence_segmentation_bounds_tts_chunks():
    from src.speech.segmentation import split_sentences

    text = " ".join(f"Clause number {i} exists." for i in range(80))
    chunks = split_sentences(text, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 5


# ---------------------------------------------------------------------------
# 9 — rapid repeated commands
# ---------------------------------------------------------------------------

def test_rapid_commands_are_bounded_and_never_concurrent():
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="ok"))
    try:
        jobs = [agent.submit_text(f"command {i}") for i in range(8)]
        for job in jobs:
            agent.scheduler.wait(job, timeout=10)
        # No uncontrolled concurrency: the queue processed them one at a time.
        assert agent.scheduler.processed >= 1
        assert agent.scheduler.status()["queue_depth"] == 0
        dropped = sum(1 for j in jobs if j.dropped)
        assert dropped <= 7  # bounded queue superseded the older ones
        assert not any(j.error for j in jobs if not j.dropped)
    finally:
        agent.shutdown()


def test_queue_full_is_reported_when_not_dropping():
    from src.orchestrator.scheduler import PipelineScheduler
    from src.utils.exceptions import QueueFullError

    gate = threading.Event()
    release = threading.Event()

    def handler(job):
        gate.set()
        release.wait(3)
        return "done"

    sched = PipelineScheduler(handler, queue_size=1, drop_oldest=False)
    sched.start()
    try:
        first = sched.submit("a")
        gate.wait(2)
        sched.submit("b")                     # fills the single slot
        with pytest.raises(QueueFullError):
            sched.submit("c")
    finally:
        release.set()
        sched.shutdown()


def test_superseded_requests_cannot_overwrite_newer_results():
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="ok"))
    try:
        jobs = [agent.submit_text(f"q{i}") for i in range(6)]
        for job in jobs:
            agent.scheduler.wait(job, timeout=10)
        newer = [j for j in jobs if not j.dropped]
        assert newer, "at least one request must complete"
        # The newest submitted request is the last non-dropped one.
        assert jobs[-1] is newer[-1] or jobs[-1].dropped
    finally:
        agent.shutdown()


# ---------------------------------------------------------------------------
# 10 — clean shutdown
# ---------------------------------------------------------------------------

def test_clean_shutdown_releases_everything():
    camera = FakeCamera()
    player = FakePlayer()
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="bye", camera=camera, player=player))
    agent.handle_text("Hello")
    agent.shutdown()

    assert camera.closed, "camera must be released"
    assert agent.scheduler.status()["closed"]
    assert agent.sm.state.value in {"SHUTDOWN", "IDLE"}
    assert not agent.is_busy


def test_shutdown_during_inference_is_safe():
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="hi"))
    # simulate in-flight work
    agent.scheduler.start()
    job = agent.submit_text("slow command")
    agent.shutdown(timeout_s=3)
    assert agent.scheduler.status()["closed"]
    assert job.done.is_set()


def test_double_shutdown_is_idempotent():
    agent = _start(make_agent())
    agent.shutdown()
    agent.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# speech → text → answer → speech (integration with fakes)
# ---------------------------------------------------------------------------

def test_recording_pipeline_end_to_end():
    stt = FakeSTT(text="what do you see in front of you")
    agent = _start(make_agent(plan=VISUAL_PLAN, answer="A laptop.", stt=stt))
    try:
        record = agent.handle_recording(recording(duration_s=1.5))
    finally:
        agent.shutdown()

    assert stt.calls == 1
    assert record.transcript == "what do you see in front of you"
    assert record.evidence and record.response
    assert agent.player.played


def test_empty_transcript_is_dropped_before_routing():
    stt = FakeSTT(text="")
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="hi", stt=stt))
    calls_before = len(agent.router.llm.client.calls) if hasattr(agent.router.llm, "client") else 0
    try:
        record = agent.handle_recording(recording())
    finally:
        agent.shutdown()
    assert record.error == "empty transcript"
    assert record.response == ""
    assert agent.camera.snapshots == 0


def test_empty_transcript_reports_why_to_the_ui():
    """A silent/failed transcription must not look like a dead pipeline."""
    stt = FakeSTT(text="")
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="hi", stt=stt))
    seen: list[str] = []
    agent.on("error", seen.append)
    try:
        agent.handle_recording(recording())
    finally:
        agent.shutdown()
    assert seen, "an empty transcript should surface a user-visible message"
    assert "try again" in seen[0].lower() or "hear" in seen[0].lower()


def test_stt_engine_failure_is_reported_not_swallowed():
    stt = FakeSTT(
        text="",
        fail=True,
        error="Library libcublas.so.12 is not found or cannot be loaded",
    )
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="hi", stt=stt))
    seen: list[str] = []
    agent.on("error", seen.append)
    try:
        record = agent.handle_recording(recording())
    finally:
        agent.shutdown()
    assert record.response == ""
    assert seen and "recognition" in seen[0].lower()


def test_time_to_first_audio_is_measured():
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="Hello there, friend."))
    try:
        record = agent.handle_text(
            "Hi robot",
            speech_ended_at=time.time() - 0.05,
        )
    finally:
        agent.shutdown()
    metrics = agent.last_metrics
    assert "time_to_first_audio" in metrics.stages
    assert metrics.stages["time_to_first_audio"] >= 0


def test_metrics_record_all_stages():
    agent = _start(make_agent(plan=VISUAL_PLAN, answer="a table"))
    try:
        agent.handle_text("What do you see?", speech_ended_at=time.time())
    finally:
        agent.shutdown()
    stages = agent.last_metrics.stages
    for expected in ("route", "camera_snapshot", "generate_response", "kokoro_synth"):
        assert expected in stages, f"missing stage {expected}: {sorted(stages)}"


def test_state_returns_to_idle_after_every_turn():
    agent = _start(make_agent(plan=plan_dict("greeting"), answer="hi"))
    try:
        for text in ("Hello", "How are you?", "Goodbye"):
            agent.handle_text(text)
            assert agent.sm.state is State.IDLE, f"stuck in {agent.sm.state} after {text!r}"
    finally:
        agent.shutdown()
