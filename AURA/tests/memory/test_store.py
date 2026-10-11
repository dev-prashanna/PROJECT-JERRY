"""Entity store CRUD, facts, embeddings, deletion and merge."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.models import (
    EntityType,
    EntityStatus,
    FactStatus,
    MemoryEntity,
    MemoryFact,
)
from src.memory.store import MemoryStore
from src.memory.vector_index import serialize_vector


def _entity(name="Alex", etype=EntityType.PERSON, **kw) -> MemoryEntity:
    return MemoryEntity(canonical_name=name, entity_type=etype, **kw)


def test_upsert_and_get(store: MemoryStore):
    e = _entity()
    store.upsert_entity(e)
    got = store.get_entity(e.entity_id)
    assert got is not None and got.canonical_name == "Alex"


def test_alias_lookup_case_insensitive(store: MemoryStore):
    e = _entity(aliases=["Al", "Big Al"])
    store.upsert_entity(e)
    assert store.find_entities_by_name("big al")[0].entity_id == e.entity_id


def test_two_people_can_share_a_name(store: MemoryStore):
    a = _entity("Sam")
    b = _entity("Sam")
    store.upsert_entity(a)
    store.upsert_entity(b)
    assert len(store.find_entities_by_name("Sam")) == 2


def test_fact_roundtrip_and_supersede(store: MemoryStore):
    e = _entity()
    store.upsert_entity(e)
    f1 = MemoryFact(
        subject_entity_id=e.entity_id,
        predicate="project_name",
        object_value="NATERIDA",
        source_transcript="my project is NATERIDA",
        source_turn_id="t1",
    )
    store.insert_fact(f1)
    f2 = MemoryFact(
        subject_entity_id=e.entity_id,
        predicate="project_name",
        object_value="NATERIDA-X",
        source_transcript="now NATERIDA-X",
        source_turn_id="t2",
    )
    store.supersede_fact(f1.fact_id, f2)
    active = store.facts_for_entity(e.entity_id)
    assert len(active) == 1
    assert active[0].object_value == "NATERIDA-X"
    assert active[0].supersedes_fact_id == f1.fact_id
    assert store.get_fact(f1.fact_id).status == FactStatus.SUPERSEDED


def test_foreign_key_cascade_on_entity_delete(store: MemoryStore):
    e = _entity()
    store.upsert_entity(e)
    blob, dim, _ = serialize_vector(np.random.rand(128).astype(np.float32))
    store.insert_face_embedding(e.entity_id, blob, dim, "m")
    store.insert_fact(
        MemoryFact(
            subject_entity_id=e.entity_id,
            predicate="p",
            object_value="v",
            source_transcript="x",
            source_turn_id="t",
        )
    )
    counts = store.delete_entity(e.entity_id)
    assert counts["entities"] == 1
    assert counts["face_embeddings"] == 1
    assert counts["facts"] == 1
    assert store.get_entity(e.entity_id) is None


def test_corrupt_embedding_rejected_on_read(store: MemoryStore):
    e = _entity()
    store.upsert_entity(e)
    blob, dim, _ = serialize_vector(np.zeros(128, dtype=np.float32) + 0.5)
    # Corrupt: wrong length
    store.insert_face_embedding(e.entity_id, blob[:50], 128, "m")
    rows = store.all_face_embeddings("m")
    assert len(rows) == 1
    from src.memory.exceptions import InvalidEmbedding
    from src.memory.vector_index import deserialize_vector

    with pytest.raises(InvalidEmbedding):
        deserialize_vector(rows[0][2], rows[0][3])


def test_merge_entities(store: MemoryStore):
    a = _entity("Alex")
    b = _entity("Alexandra", aliases=["Alex"])
    store.upsert_entity(a)
    store.upsert_entity(b)
    store.insert_fact(
        MemoryFact(
            subject_entity_id=a.entity_id,
            predicate="name",
            object_value="Alex",
            source_transcript="my name is Alex",
            source_turn_id="t1",
        )
    )
    stats = store.merge_entities(a.entity_id, b.entity_id)
    assert stats["facts"] == 1
    assert store.get_entity(a.entity_id) is None
    assert len(store.facts_for_entity(b.entity_id)) == 1


def test_merge_type_mismatch_rejected(store: MemoryStore):
    a = _entity("A", EntityType.PERSON)
    b = _entity("B", EntityType.OBJECT)
    store.upsert_entity(a)
    store.upsert_entity(b)
    with pytest.raises(ValueError):
        store.merge_entities(a.entity_id, b.entity_id)
