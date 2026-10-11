"""Face memory: service behavior, consent enforcement, engine-level flows."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.exceptions import ConsentRequired, FaceNotDetected, MemoryPermissionDenied
from src.memory.face_memory import DetectedFace, FaceMatchThresholds, FaceMemoryService
from src.memory.models import EnrollPersonRequest, MatchStatus
from tests.memory.conftest import FakeFaceBackend, face_image


class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_detect_finds_blue_patch(face_service):
    faces = face_service.detect_faces(face_image(seed=1))
    assert len(faces) == 1


def test_detect_plain_image_finds_nothing(face_service):
    plain = np.full((120, 160, 3), 100, dtype=np.uint8)
    assert face_service.detect_faces(plain) == []


def test_enroll_requires_self_enrollment_utterance(face_service):
    with pytest.raises(ConsentRequired):
        face_service.enroll_identity("Alex", face_image(), transcript="hello")


def test_enroll_third_party_blocked(face_service):
    with pytest.raises(MemoryPermissionDenied):
        face_service.enroll_identity(
            "Alice", face_image(seed=5), transcript="this is Alice"
        )


def test_enroll_then_same_face_matches(face_service):
    img = face_image(seed=1)
    entity, match, _ = face_service.enroll_identity(
        "Alex", img, transcript="remember my face as Alex"
    )
    faces = face_service.extract_embeddings(img)
    m = face_service.match_face(faces[0].embedding)
    assert m.status == MatchStatus.MATCHED
    assert m.entity_id == entity.entity_id
    assert m.entity_name == "Alex"


def test_unenrolled_face_unknown(face_service):
    face_service.enroll_identity(
        "Alex", face_image(seed=1), transcript="remember my face as Alex"
    )
    rng = np.random.default_rng(3)
    stranger = rng.standard_normal(128).astype(np.float32)
    stranger /= np.linalg.norm(stranger)
    m = face_service.match_face(stranger)
    assert m.status == MatchStatus.UNKNOWN
    assert m.entity_id is None


def test_remove_identity_clears_embeddings(face_service, store):
    entity, _, _ = face_service.enroll_identity(
        "Alex", face_image(seed=1), transcript="remember my face as Alex"
    )
    face_service.remove_identity(entity.entity_id)
    assert store.get_entity(entity.entity_id) is None
    assert face_service.rebuild_index() == 0


def test_engine_face_enroll_and_recognize(engine):
    backend = FakeFaceBackend()
    engine.face_service = FaceMemoryService(
        engine.store, backend, FaceMatchThresholds()
    )
    img = face_image(seed=1)
    res = engine.enroll_person(
        EnrollPersonRequest(
            entity_name="Alex",
            frame=img,
            consent_context={"transcript": "remember my face as Alex"},
        )
    )
    assert res.ok
    rec = engine.recognize_faces(_Req(frame=img))
    assert rec.ok
    assert rec.payload.get("entity_name") == "Alex"
    assert "Alex" in rec.message


def test_engine_recognize_without_enrollment_is_graceful(engine):
    backend = FakeFaceBackend()
    engine.face_service = FaceMemoryService(engine.store, backend)
    rec = engine.recognize_faces(_Req(frame=face_image(seed=1)))
    assert rec.ok
    assert "do not recognise" in rec.message or "not certain" in rec.message


def test_engine_face_disabled_reports_unavailable(engine):
    res = engine.enroll_person(
        EnrollPersonRequest(
            entity_name="Alex",
            frame=face_image(),
            consent_context={"transcript": "remember my face as Alex"},
        )
    )
    assert not res.ok
    assert "not enabled" in (res.message or "").lower() or "not enabled" in (res.error_code or "").lower()


def test_multi_face_enrollment_refused(face_service):
    img = face_image(seed=1)
    # Force two detections via a stubbed backend
    face_service._backend.detect = lambda image: [
        DetectedFace(bbox_xyxy=(10, 10, 60, 60), confidence=0.9),
        DetectedFace(bbox_xyxy=(70, 10, 120, 60), confidence=0.9),
    ]
    with pytest.raises(ConsentRequired):
        face_service.enroll_identity(
            "Alex", img, transcript="remember my face as Alex"
        )
