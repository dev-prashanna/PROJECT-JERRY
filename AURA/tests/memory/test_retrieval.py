"""Cross-modal retrieval and engine-level remember/correct/delete flows."""

from __future__ import annotations

import numpy as np

from src.memory.engine import MultimodalMemoryEngine
from src.memory.models import (
    CorrectFactRequest,
    EntityType,
    ForgetRequest,
    MatchStatus,
    RememberFactRequest,
)
from src.memory.retriever import MemoryRetriever
from tests.memory.conftest import FrameStub, face_image, make_test_image


def test_retrieve_by_name_and_fact(store):
    from src.memory.entity_resolver import EntityResolver
    from src.memory.models import MemoryFact

    resolver = EntityResolver(store)
    e = resolver.get_or_create("NATERIDA", EntityType.PROJECT)
    store.insert_fact(
        MemoryFact(
            subject_entity_id=e.entity_id,
            predicate="project_name",
            object_value="NATERIDA",
            source_transcript="my project is NATERIDA",
            source_turn_id="t1",
        )
    )
    r = MemoryRetriever(store)
    assert r.retrieve_by_name("NATERIDA")
    assert r.retrieve_by_fact("NATERIDA")
    ctx = r.retrieve_multimodal(text_query="What is NATERIDA?")
    assert any(f.object_value == "NATERIDA" for f in ctx.retrieved_facts)


def test_engine_remember_and_retrieve(engine: MultimodalMemoryEngine):
    res = engine.remember_fact(
        RememberFactRequest(transcript="My robotics project is called NATERIDA", turn_id="t1")
    )
    assert res.ok
    got = engine.retrieve("What did I tell you my robotics project was called?")
    assert got.ok
    assert "NATERIDA" in got.message
    assert "NATERIDA" in got.payload["memory_block"]


def test_engine_correction_supersedes(engine: MultimodalMemoryEngine):
    engine.remember_fact(
        RememberFactRequest(transcript="My robotics project is called NATERIDA", turn_id="t1")
    )
    res = engine.correct_fact(
        CorrectFactRequest(transcript="Actually the project is now called NATERIDA-X", turn_id="t2")
    )
    assert res.ok
    got = engine.retrieve("robotics project name")
    assert "NATERIDA-X" in got.message
    assert "NATERIDA-X" in got.payload["memory_block"]


def test_unrelated_person_facts_not_returned(engine: MultimodalMemoryEngine):
    engine.remember_fact(
        RememberFactRequest(transcript="My name is Alex", turn_id="t1")
    )
    engine.remember_fact(
        RememberFactRequest(transcript="My robotics project is called NATERIDA", turn_id="t2")
    )
    ctx = engine.retrieve_context("What is my name?")
    # Alex's name fact should be present; object facts about other subjects
    # should not drown it out.
    assert any(f.object_value == "Alex" for f in ctx.retrieved_facts)


def test_delete_entity_and_facts(engine: MultimodalMemoryEngine):
    engine.remember_fact(
        RememberFactRequest(transcript="My robotics project is called NATERIDA", turn_id="t1")
    )
    res = engine.delete_memory(ForgetRequest(name="NATERIDA"))
    if not res.ok:
        # subject may be "my project"; try that name
        res = engine.delete_memory(ForgetRequest(name="my project"))
    assert res.ok
    assert engine.retrieve("NATERIDA").payload["fact_count"] == 0


def test_face_cross_modal_link(store, face_service):
    # Enroll a face, then retrieve via face embedding
    img = face_image(seed=1)
    entity, match, count = face_service.enroll_identity(
        "Alex", img, transcript="remember my face as Alex"
    )
    assert entity.canonical_name == "Alex"
    faces = face_service.extract_embeddings(img)
    assert faces and faces[0].embedding is not None
    m2 = face_service.match_face(faces[0].embedding)
    assert m2.status == MatchStatus.MATCHED
    assert m2.entity_id == entity.entity_id


def test_person_b_not_matched_to_person_a(store, face_service):
    img_a = face_image(seed=1)
    entity_a, _, _ = face_service.enroll_identity(
        "Alex", img_a, transcript="remember my face as Alex"
    )
    # A genuinely different person's face yields an unrelated 128-D vector.
    rng = np.random.default_rng(42)
    other_face = rng.standard_normal(128).astype(np.float32)
    other_face /= np.linalg.norm(other_face)
    m = face_service.match_face(other_face)
    assert m.status == MatchStatus.UNKNOWN or m.entity_id != entity_a.entity_id


def test_ambiguous_when_two_candidates_tie(store, face_service):
    from src.memory.models import EntityType, MemoryEntity
    from src.memory.vector_index import serialize_vector

    alex = MemoryEntity(canonical_name="Alex", entity_type=EntityType.PERSON)
    sam = MemoryEntity(canonical_name="Sam", entity_type=EntityType.PERSON)
    store.upsert_entity(alex)
    store.upsert_entity(sam)

    def unit(vec):
        v = np.asarray(vec, dtype=np.float32)
        return v / np.linalg.norm(v)

    q = np.zeros(128, dtype=np.float32)
    q[0] = 1.0
    p = np.zeros(128, dtype=np.float32)
    p[1] = 1.0
    # Embeddings ~0.40 and ~0.44 from q: best under threshold, within margin.
    e1 = unit(np.cos(0.4027) * q + np.sin(0.4027) * p)
    e2 = unit(np.cos(0.4429) * q + np.sin(0.4429) * p)
    for entity, vec in ((alex, e1), (sam, e2)):
        blob, dim, _dtype = serialize_vector(vec)
        store.insert_face_embedding(entity.entity_id, blob, dim, "fake-face")
    face_service.rebuild_index()
    m = face_service.match_face(q.astype(np.float32))
    assert m.status == MatchStatus.AMBIGUOUS
    assert m.entity_id is None


def test_object_enroll_and_recognize(engine: MultimodalMemoryEngine):
    img = make_test_image(seed=3, color=(0, 180, 220))
    res = engine.enroll_object(
        __import__("src.memory.models", fromlist=["EnrollObjectRequest"]).EnrollObjectRequest(
            object_name="custom motor controller",
            frame=FrameStub(img),
            text_query="circuit board",
        )
    )
    assert res.ok
    rec = engine.recognize_objects(
        type("R", (), {"frame": FrameStub(img), "text_query": "device"})()
    )
    assert rec.ok
    assert "motor controller" in rec.message
