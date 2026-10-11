"""Entity resolution: names, subjects, associations, provisional entities."""

from __future__ import annotations

from src.memory.entity_resolver import EntityResolver
from src.memory.models import EntityType, EntityStatus


def test_get_or_create_then_reuse(store):
    r = EntityResolver(store)
    a = r.get_or_create("Alex", EntityType.PERSON)
    b = r.get_or_create("alex", EntityType.PERSON)
    assert a.entity_id == b.entity_id
    assert len(store.list_entities()) == 1


def test_resolve_by_alias(store):
    r = EntityResolver(store)
    e = r.get_or_create("Alex", EntityType.PERSON, aliases=["Al"])
    res = r.resolve_by_name("Al", EntityType.PERSON)
    assert res.entity is not None and res.entity.entity_id == e.entity_id


def test_ambiguous_names_reported(store):
    r = EntityResolver(store)
    r.get_or_create("Sam", EntityType.PERSON)
    r.get_or_create("Sam", EntityType.PERSON)  # resolves to first; force second
    from src.memory.models import MemoryEntity

    store.upsert_entity(MemoryEntity(canonical_name="Sam", entity_type=EntityType.PERSON))
    res = r.resolve_by_name("Sam", EntityType.PERSON)
    assert res.status == "ambiguous"
    assert len(res.candidates) >= 2


def test_subject_resolution_speaker(store):
    r = EntityResolver(store)
    speaker = r.get_or_create("current speaker", EntityType.PERSON, provisional=True)
    res = r.resolve_subject("current_speaker", speaker_entity=speaker)
    assert res.entity is not None and res.entity.entity_id == speaker.entity_id
    assert res.entity.status == EntityStatus.PROVISIONAL


def test_subject_resolution_project(store):
    r = EntityResolver(store)
    e = r.get_or_create("NATERIDA", EntityType.PROJECT)
    res = r.resolve_subject("my_project")
    # No prior binding: unknown until named
    assert res.status in {"unknown", "matched"}


def test_association_deduplicated(store):
    r = EntityResolver(store)
    a = r.get_or_create("Alex", EntityType.PERSON)
    b = r.get_or_create("motor controller", EntityType.OBJECT)
    r.link_association(a, b, "owns")
    r.link_association(a, b, "owns")
    assert len(store.associations_for(a.entity_id)) == 1


def test_merge_via_resolver(store):
    r = EntityResolver(store)
    a = r.get_or_create("Alex", EntityType.PERSON)
    b = r.get_or_create("Alexandra", EntityType.PERSON)
    stats = r.merge(a.entity_id, b.entity_id)
    assert "facts" in stats
    assert store.get_entity(a.entity_id) is None
