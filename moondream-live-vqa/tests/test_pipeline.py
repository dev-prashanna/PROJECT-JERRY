"""Pipeline and inference-worker tests using a fake engine.

No GPU, no model weights and no camera hardware are required: frames are
injected straight into the latest-frame buffer and the engine is stubbed, so
these tests exercise orchestration logic (modes, queueing, shutdown, events)
deterministically.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest

from config.settings import build_config
from src.pipeline.inference_worker import InferenceJob, InferenceWorker
from src.pipeline.vqa_pipeline import PipelineMode, VQAPipeline
from src.vlm.moondream_engine import EngineInfo


def fake_info() -> EngineInfo:
    return EngineInfo(
        model_path="/fake/models/moondream2",
        device="cpu",
        dtype="float32",
        api="fake",
        model_class="FakeEngine",
        torch_version="0",
        transformers_version="0",
        load_seconds=0.0,
        parameter_count=0,
    )


class FakeEngine:
    """Minimal stand-in for :class:`MoondreamEngine`."""

    def __init__(
        self,
        *,
        loaded: bool = True,
        delay: float = 0.02,
        gate: Optional[threading.Event] = None,
        fail: bool = False,
    ) -> None:
        self.is_loaded = loaded
        self.device = "cpu"
        self.delay = delay
        self.gate = gate
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []
        self.cache_key: Optional[str] = None

    def load(self) -> EngineInfo:
        self.is_loaded = True
        return fake_info()

    def info(self):
        """Mirrors ``MoondreamEngine.info`` (None until loaded)."""
        return fake_info() if self.is_loaded else None

    def clear_image_cache(self) -> None:
        self.cache_key = None

    def ask(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        if self.gate is not None:
            self.gate.wait(timeout=10)
        else:
            time.sleep(self.delay)
        self.calls.append(
            {
                "question": question,
                "frame_id": kwargs.get("frame_id"),
                "cache_key": kwargs.get("cache_key"),
                "mode_frame_shape": None
                if kwargs.get("frame") is None
                else tuple(np.asarray(kwargs["frame"]).shape),
            }
        )
        if self.fail:
            raise RuntimeError("synthetic engine failure")
        return {
            "status": "success",
            "answer": f"echo: {question}",
            "error": None,
            "inference_time_seconds": 0.01,
            "encode_time_seconds": 0.0,
            "generate_time_seconds": 0.01,
            "preprocess_time_seconds": 0.0,
            "encoding_reused": False,
            "device": self.device,
            "frame_id": kwargs.get("frame_id"),
            "frame_timestamp": kwargs.get("frame_timestamp"),
            "question": question,
            "model_path": "/fake/models/moondream2",
            "api": "fake",
        }


@pytest.fixture
def events() -> List[tuple]:
    return []


@pytest.fixture
def make_pipeline(events):
    """Factory that always shuts the pipeline down afterwards."""
    created: List[VQAPipeline] = []

    def factory(
        engine: Optional[FakeEngine] = None,
        *,
        queue_size: int = 1,
        grounding: bool = False,
        auto_load: bool = True,
    ) -> VQAPipeline:
        config = build_config(load_env=False, queue_size=queue_size)
        config.inference.grounding_prompt = grounding
        pipeline = VQAPipeline(
            config,
            on_event=lambda kind, payload: events.append((kind, dict(payload))),
            engine=engine or FakeEngine(),
        )
        created.append(pipeline)
        if auto_load:
            pipeline.load_model_blocking()
        return pipeline

    yield factory
    for pipeline in created:
        pipeline.stop(timeout=5)


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def results_of(events: List[tuple]) -> List[dict]:
    return [payload for kind, payload in events if kind == "inference_result"]


def inject_frame(pipeline: VQAPipeline, value: int = 1) -> int:
    frame = np.full((24, 32, 3), value, dtype=np.uint8)
    return pipeline.camera.buffer.update(frame).frame_id


class TestQuestionValidation:
    def test_empty_question_rejected(self, make_pipeline):
        pipeline = make_pipeline()
        ok, message = pipeline.ask("   ")
        assert ok is False
        assert "question" in message.lower()

    def test_too_long_question_rejected(self, make_pipeline):
        pipeline = make_pipeline()
        ok, message = pipeline.ask("x" * 4001)
        assert ok is False
        assert "too long" in message

    def test_model_must_be_loaded(self, make_pipeline):
        pipeline = make_pipeline(engine=FakeEngine(loaded=False), auto_load=False)
        inject_frame(pipeline)
        ok, message = pipeline.ask("hello")
        assert ok is False
        assert "loading" in message


class TestLiveMode:
    def test_question_requires_a_frame(self, make_pipeline):
        pipeline = make_pipeline()
        ok, message = pipeline.ask("what is this?")
        assert ok is False
        assert "no camera frame" in message

    def test_answer_uses_latest_frame_and_reports_it(self, make_pipeline, events):
        pipeline = make_pipeline()
        frame_id = inject_frame(pipeline, value=10)
        ok, _ = pipeline.ask("what color is this?")
        assert ok is True
        assert wait_for(lambda: results_of(events))
        result = results_of(events)[0]
        assert result["status"] == "success"
        assert result["frame_id"] == frame_id
        assert result["question"] == "what color is this?"
        assert result["answer"] == "echo: what color is this?"
        assert result["mode"] == PipelineMode.LIVE

    def test_each_question_gets_the_newest_frame(self, make_pipeline, events):
        pipeline = make_pipeline()
        first_id = inject_frame(pipeline, value=1)
        assert pipeline.ask("first")[0]
        assert wait_for(lambda: len(results_of(events)) == 1)

        second_id = inject_frame(pipeline, value=2)
        assert pipeline.ask("second")[0]
        assert wait_for(lambda: len(results_of(events)) == 2)

        ids = [r["frame_id"] for r in results_of(events)]
        assert ids == [first_id, second_id]
        assert second_id != first_id

    def test_frame_snapshot_is_an_independent_copy(self, make_pipeline):
        pipeline = make_pipeline()
        frame_id = inject_frame(pipeline, value=42)
        assert pipeline.ask("q")[0]
        job = wait_for(lambda: pipeline.worker.active_job is not None)
        assert job
        stored = pipeline.camera.buffer.get_latest()
        assert stored.frame_id == frame_id
        assert stored.frame[0, 0, 0] == 42


class TestFrozenMode:
    def test_freeze_uses_the_same_frame_for_every_question(
        self, make_pipeline, events
    ):
        pipeline = make_pipeline()
        frame_id = inject_frame(pipeline, value=9)
        frozen = pipeline.freeze()
        assert frozen is not None
        assert frozen.frame_id == frame_id
        assert pipeline.mode == PipelineMode.FROZEN

        assert pipeline.ask("one")[0]
        assert wait_for(lambda: len(results_of(events)) == 1)
        new_id = inject_frame(pipeline, value=99)  # camera keeps running
        assert pipeline.ask("two")[0]
        assert wait_for(lambda: len(results_of(events)) == 2)

        ids = [r["frame_id"] for r in results_of(events)]
        assert ids == [frame_id, frame_id]
        assert new_id != frame_id

        # identical cache keys let the engine reuse the vision encoding
        keys = [call["cache_key"] for call in pipeline.engine.calls]
        assert keys[0] == keys[1] == f"frame:{frame_id}"

    def test_unfreeze_returns_to_live_mode(self, make_pipeline, events):
        pipeline = make_pipeline()
        inject_frame(pipeline, value=1)
        pipeline.freeze()
        pipeline.unfreeze()
        assert pipeline.mode == PipelineMode.LIVE
        assert pipeline.frozen_info is None

        first = inject_frame(pipeline, value=2)
        assert pipeline.ask("live?")[0]
        assert wait_for(lambda: results_of(events))
        assert results_of(events)[0]["frame_id"] == first

    def test_freeze_without_frames_fails_gracefully(self, make_pipeline, events):
        pipeline = make_pipeline()
        assert pipeline.freeze() is None
        assert any(kind == "error" for kind, _ in events)


class TestQueueing:
    def test_queue_is_bounded(self, make_pipeline, events):
        gate = threading.Event()
        engine = FakeEngine(gate=gate)
        pipeline = make_pipeline(engine=engine, queue_size=1)
        inject_frame(pipeline)

        assert pipeline.ask("job-1")[0]
        assert wait_for(lambda: pipeline.worker.active_job is not None)

        assert pipeline.ask("job-2")[0]  # one may wait behind the active job
        ok, message = pipeline.ask("job-3")  # capacity exceeded
        assert ok is False
        assert "queued" in message.lower()

        gate.set()
        assert wait_for(lambda: len(results_of(events)) == 2)
        assert [r["question"] for r in results_of(events)] == ["job-1", "job-2"]

    def test_zero_queue_rejects_while_busy(self, make_pipeline):
        gate = threading.Event()
        engine = FakeEngine(gate=gate)
        pipeline = make_pipeline(engine=engine, queue_size=0)
        inject_frame(pipeline)
        assert pipeline.ask("first")[0]
        assert wait_for(lambda: pipeline.worker.active_job is not None)
        ok, message = pipeline.ask("second")
        gate.set()
        assert ok is False
        assert "already" in message or "queue" in message

    def test_rejected_event_is_emitted(self, make_pipeline, events):
        pipeline = make_pipeline()
        # no frame -> rejection path emits nothing; force a queue rejection
        inject_frame(pipeline)
        assert pipeline.ask("ok")[0]
        wait_for(lambda: pipeline.worker.active_job is not None)
        pipeline.ask("also ok")  # queued
        pipeline.ask("too much")  # rejected -> event
        wait_for(
            lambda: any(kind == "rejected" for kind, _ in events), timeout=3
        )
        assert any(kind == "rejected" for kind, _ in events)


class TestFailureHandling:
    def test_engine_exception_becomes_error_result(self, make_pipeline, events):
        pipeline = make_pipeline(engine=FakeEngine(fail=True))
        inject_frame(pipeline)
        assert pipeline.ask("boom")[0]
        assert wait_for(lambda: results_of(events))
        result = results_of(events)[0]
        assert result["status"] == "error"
        assert "synthetic engine failure" in result["error"]
        # the worker survives the failure
        assert pipeline.worker.is_running

    def test_error_result_carries_the_frame_it_used(self, make_pipeline, events):
        pipeline = make_pipeline(engine=FakeEngine(fail=True))
        frame_id = inject_frame(pipeline)
        pipeline.ask("boom")
        assert wait_for(lambda: results_of(events))
        assert results_of(events)[0]["frame_id"] == frame_id


class TestEvents:
    def test_submitted_event_contains_question_and_frame(self, make_pipeline, events):
        pipeline = make_pipeline()
        frame_id = inject_frame(pipeline)
        pipeline.ask("what now?")
        submitted = [p for k, p in events if k == "submitted"]
        assert submitted and submitted[0]["question"] == "what now?"
        assert submitted[0]["frame_id"] == frame_id
        assert submitted[0]["mode"] == PipelineMode.LIVE

    def test_mode_changed_event_on_freeze(self, make_pipeline, events):
        pipeline = make_pipeline()
        inject_frame(pipeline)
        pipeline.freeze()
        payloads = [p for k, p in events if k == "mode_changed"]
        assert payloads and payloads[-1]["mode"] == PipelineMode.FROZEN
        assert payloads[-1]["frozen"]["frame_id"] == 1

    def test_model_status_event(self, make_pipeline, events):
        make_pipeline()
        payloads = [p for k, p in events if k == "model_status"]
        assert payloads
        assert payloads[-1]["state"] == "loaded"
        assert payloads[-1]["model_class"] == "FakeEngine"

    def test_status_summary(self, make_pipeline):
        pipeline = make_pipeline()
        summary = pipeline.status_summary
        assert summary["model_state"] == "loaded"
        assert summary["inference"] == "idle"
        assert summary["mode"] == PipelineMode.LIVE


class TestInferenceWorkerUnit:
    def test_submit_before_start_auto_starts(self, events):
        engine = FakeEngine(delay=0.01)
        worker = InferenceWorker(
            engine,
            on_result=lambda result: events.append(result),
            queue_size=1,
        )
        snapshot = None
        from src.camera.frame_buffer import FrameSnapshot

        snapshot = FrameSnapshot(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            frame_id=1,
            timestamp=time.time(),
            width=4,
            height=4,
        )
        job = InferenceJob(
            job_id=1, question="hi", snapshot=snapshot, cache_key="frame:1", mode="live"
        )
        ok, _ = worker.submit(job)
        assert ok
        assert wait_for(lambda: len(events) == 1)
        worker.stop(timeout=5)
        assert not worker.is_running

    def test_stop_drops_late_results(self):
        gate = threading.Event()
        engine = FakeEngine(gate=gate)
        delivered: List[dict] = []
        worker = InferenceWorker(engine, on_result=delivered.append, queue_size=1)
        from src.camera.frame_buffer import FrameSnapshot

        snapshot = FrameSnapshot(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            frame_id=1,
            timestamp=time.time(),
            width=4,
            height=4,
        )
        worker.start()
        worker.submit(
            InferenceJob(
                job_id=1,
                question="q",
                snapshot=snapshot,
                cache_key="frame:1",
                mode="live",
            )
        )
        assert wait_for(lambda: worker.active_job is not None)
        worker.stop(timeout=2)  # request still blocked on the gate
        gate.set()
        time.sleep(0.2)
        assert delivered == []


class TestModelLoadFailure:
    def test_blocking_load_failure_is_reported(self, events):
        engine = FakeEngine()

        def boom() -> EngineInfo:
            raise RuntimeError("weights missing")

        engine.load = boom  # type: ignore[method-assign]
        config = build_config(load_env=False)
        pipeline = VQAPipeline(
            config,
            on_event=lambda kind, payload: events.append((kind, dict(payload))),
            engine=engine,
        )
        try:
            with pytest.raises(RuntimeError, match="weights missing"):
                pipeline.load_model_blocking()
            assert pipeline.model_state == "error"
            assert "weights missing" in (pipeline.model_error or "")
            assert any(kind == "error" for kind, _ in events)
            statuses = [p for k, p in events if k == "model_status"]
            assert statuses and statuses[-1]["state"] == "error"
            # the pipeline still refuses questions
            inject_frame(pipeline)
            ok, _ = pipeline.ask("anything")
            assert ok is False
        finally:
            pipeline.stop(timeout=3)
