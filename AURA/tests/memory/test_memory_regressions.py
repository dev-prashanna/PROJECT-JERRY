"""Real API-shaped detections and memory behavior missed by the fake backend."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from src.memory.face_memory import OpenCVFaceBackend, FaceRecognitionApiBackend
from src.memory.fact_extractor import extract_deterministic
from src.memory.models import RememberFactRequest, MatchStatus
from tests.memory.conftest import face_image


def test_yunet_box_and_landmarks_reach_sface(tmp_path):
    row = np.array([120, 60, 80, 90, 140, 85, 180, 85, 160, 105,
                    145, 125, 175, 125, .98], dtype=np.float32)
    aligned_rows = []
    backend = OpenCVFaceBackend(tmp_path, allow_download=False)
    backend._detector = SimpleNamespace(
        setInputSize=lambda size: None,
        detect=lambda image: (1, row[None, :]),
    )
    def align(image, detection):
        aligned_rows.append(np.asarray(detection).reshape(-1))
        return np.zeros((112, 112, 3), dtype=np.uint8)
    backend._recognizer = SimpleNamespace(
        alignCrop=align, feature=lambda image: np.ones((1, 128), dtype=np.float32)
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    faces = backend.detect(image)
    assert len(faces) == 1
    assert faces[0].bbox_xyxy == (120, 60, 200, 150)
    backend.embed(image, faces[0])
    np.testing.assert_array_equal(aligned_rows[0], row)


def test_dlib_box_round_trip():
    locations = []
    backend = FaceRecognitionApiBackend()
    def encode(image, known_face_locations):
        locations.extend(known_face_locations)
        return [np.ones(128, dtype=np.float32)]
    backend._mod = SimpleNamespace(
        face_locations=lambda image, model: [(20, 150, 100, 70)],
        face_encodings=encode,
    )
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    face = backend.detect(image)[0]
    assert face.bbox_xyxy == (70, 20, 150, 100)
    backend.embed(image, face)
    assert locations == [(20, 150, 100, 70)]


def test_reenrollment_does_not_compete_with_same_person(face_service):
    image = face_image(seed=1)
    for _ in range(2):
        face_service.enroll_identity("Alex", image, transcript="remember my face as Alex")
    match = face_service.match_face(face_service.extract_embeddings(image)[0].embedding)
    assert match.status == MatchStatus.MATCHED
    assert match.entity_name == "Alex"


def test_face_deletion_updates_live_index(face_service):
    image = face_image(seed=1)
    entity, _, _ = face_service.enroll_identity("Alex", image, transcript="remember my face as Alex")
    face_service.remove_identity(entity.entity_id)
    assert len(face_service._index) == 0


@pytest.mark.parametrize("text", ["I am tired", "I'm happy", "I am working on my robot"])
def test_state_is_not_a_person_name(text):
    result = extract_deterministic(text)
    assert not any(f.predicate == "name" for f in result.facts)


def test_name_ends_before_another_clause():
    result = extract_deterministic("My name is Alex and I like robotics")
    assert next(f.object_value for f in result.facts if f.predicate == "name") == "Alex"


def test_ownership_is_additive(engine):
    for text in ("I own a bicycle", "I own a telescope"):
        assert engine.remember_fact(RememberFactRequest(transcript=text, use_llm=False)).ok
    facts = engine.store.search_facts(predicate="own", limit=20)
    assert {f.object_value.lower() for f in facts} == {"bicycle", "telescope"}


def test_person_correction_cannot_overwrite_project(engine):
    engine.remember_fact(RememberFactRequest(transcript="My project is called ORION", use_llm=False))
    engine.remember_fact(RememberFactRequest(transcript="Actually my name is Alex", use_llm=False))
    assert any(f.object_value == "ORION" for f in engine.store.search_facts(limit=20))


def test_project_correction_preserves_person_name(engine):
    for text in ("My project is called ORION", "My name is Alex",
                 "Actually the project is now called HELIOS"):
        assert engine.remember_fact(RememberFactRequest(transcript=text, use_llm=False)).ok
    values = {f.object_value for f in engine.store.search_facts(limit=20)}
    assert "Alex" in values
    assert "HELIOS" in values
    assert "ORION" not in values


def test_learning_without_background_indexing(engine):
    engine.settings = replace(engine.settings, auto_learn_facts=True)
    engine.maybe_learn_facts("I like green tea")
    assert any(f.object_value == "green tea" for f in engine.store.search_facts(limit=20))


def test_background_learning_uses_configured_llm(engine):
    from src.memory.engine import _JobHandler
    from src.memory.resource_scheduler import MemoryJobScheduler

    engine.settings = replace(engine.settings, auto_learn_facts=True,
                              background_indexing=True, use_llm_fact_extraction=True)
    engine.llm_client = SimpleNamespace(chat_json=lambda *args, **kwargs: (
        {"facts": [{"predicate": "favorite_drink", "object": "oolong tea"}]}, None
    ))
    engine.scheduler = MemoryJobScheduler(_JobHandler(engine))
    engine.scheduler.start()
    job = engine.maybe_learn_facts("Oolong tea is my go-to drink")
    assert job.done.wait(3)
    assert job.error is None
    assert any(f.object_value == "oolong tea" for f in engine.store.search_facts(limit=20))


@pytest.mark.parametrize("speak", [True, False])
def test_failed_face_enrollment_returns_to_idle(tmp_path, speak):
    from src.memory.face_memory import FaceMemoryService
    from src.orchestrator.state_machine import State
    from tests.conftest import FakeCamera, make_agent
    from tests.memory.conftest import FakeFaceBackend
    from tests.memory.test_orchestrator_integration import _memory_settings

    agent = make_agent(settings=_memory_settings(tmp_path), router_mode="deterministic",
                       camera=FakeCamera(np.zeros((120, 160, 3), dtype=np.uint8)))
    agent.start(warm_models=False)
    try:
        agent.memory_engine.face_service = FaceMemoryService(agent.memory_engine.store, FakeFaceBackend())
        record = agent.handle_text("remember my face as Alex", speak=speak)
        assert record.tool_results and not record.tool_results[0].ok
        assert agent.sm.state == State.IDLE
        assert any(t.current == State.GENERATING for t in agent.sm.history)
        if speak:
            assert any(t.current == State.SPEAKING for t in agent.sm.history)
    finally:
        agent.shutdown()


def test_general_reply_receives_saved_facts_after_restart(tmp_path):
    from tests.conftest import make_agent
    from tests.memory.test_orchestrator_integration import _memory_settings

    settings = _memory_settings(tmp_path)
    first = make_agent(settings=settings)
    first.start(warm_models=False)
    first.memory_engine.remember_fact(RememberFactRequest(
        transcript="My project is called ORION", use_llm=False))
    first.shutdown()
    agent = make_agent(settings=settings)
    agent.start(warm_models=False)
    try:
        record = agent.handle_text("Help me plan my project", speak=False)
        assert record.plan.intent.value == "general_question"
        messages = agent.generator.client.calls[-1]["messages"]
        assert any("ORION" in m["content"] for m in messages)
    finally:
        agent.shutdown()


def test_response_uses_configured_history_window():
    from tests.conftest import make_agent

    agent = make_agent()
    agent.start(warm_models=False)
    try:
        for n in range(6):
            agent.memory.record(user_transcript=f"context marker {n}", final_response="Understood.")
        agent.handle_text("Let's continue", speak=False)
        messages = agent.generator.client.calls[-1]["messages"]
        assert any(m["content"] == "context marker 0" for m in messages)
    finally:
        agent.shutdown()


def test_depth_input_matches_actual_model_dtype():
    import torch
    from src.perception.depth_engine import DepthEngine

    class TinyDepth(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1).to(torch.bfloat16)
        def forward(self, pixel):
            assert pixel.dtype == self.conv.weight.dtype
            return SimpleNamespace(predicted_depth=self.conv(pixel)[:, 0])

    engine = DepthEngine()
    engine._model = TinyDepth()
    engine._processor = lambda **kwargs: {"pixel_values": torch.ones((1, 3, 8, 8))}
    result = engine.infer(np.zeros((16, 16, 3), dtype=np.uint8))
    assert result.depth.shape == (16, 16)
    assert np.isfinite(result.depth).all()
