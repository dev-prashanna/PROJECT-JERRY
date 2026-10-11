"""Migration behaviour: ordering, backups, preservation of records."""

from __future__ import annotations

from src.memory.migrations import MIGRATIONS, current_version
from src.memory.models import EntityType, MemoryEntity, MemoryFact
from src.memory.store import MemoryStore


def test_migrations_ordered():
    versions = [v for v, _, _ in MIGRATIONS]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


def test_records_survive_reopen(tmp_path):
    path = tmp_path / "keep.db"
    s = MemoryStore(path)
    e = MemoryEntity(canonical_name="Keepme", entity_type=EntityType.CONCEPT)
    s.upsert_entity(e)
    s.insert_fact(
        MemoryFact(
            subject_entity_id=e.entity_id,
            predicate="note",
            object_value="survives",
            source_transcript="note that",
            source_turn_id="t1",
        )
    )
    s.close()
    s2 = MemoryStore(path)
    assert s2.get_entity(e.entity_id).canonical_name == "Keepme"
    assert len(s2.facts_for_entity(e.entity_id)) == 1
    s2.close()


def test_backup_creates_file(store: MemoryStore, tmp_path):
    dest = tmp_path / "backups" / "copy.db"
    out = store.backup(dest)
    assert out.is_file() and out.stat().st_size > 0


def test_current_version_matches_store(store: MemoryStore):
    assert current_version(store) == store.schema_version
