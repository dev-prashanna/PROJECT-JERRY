"""Schema creation and migration behaviour."""

from __future__ import annotations

import sqlite3

import pytest

from src.memory.exceptions import MemoryMigrationError
from src.memory.migrations import MIGRATIONS, current_version, run_migrations
from src.memory.store import MemoryStore


def test_schema_version_applied(store: MemoryStore):
    assert store.schema_version == max(v for v, _, _ in MIGRATIONS)


def test_core_tables_exist(store: MemoryStore):
    rows = store.execute_fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    names = {r[0] for r in rows}
    for expected in (
        "entities",
        "entity_aliases",
        "face_embeddings",
        "object_embeddings",
        "declarative_memory",
        "observations",
        "entity_observations",
        "entity_associations",
        "enrollment_consent",
        "identity_bindings",
        "memory_revisions",
        "embedding_models",
        "schema_migrations",
    ):
        assert expected in names, expected


def test_foreign_keys_enabled(store: MemoryStore):
    row = store.execute_fetchone("PRAGMA foreign_keys")
    assert row is not None and int(row[0]) == 1


def test_wal_mode(store: MemoryStore):
    row = store.execute_fetchone("PRAGMA journal_mode")
    assert row is not None and str(row[0]).lower() == "wal"


def test_migration_idempotent(tmp_path):
    path = tmp_path / "idem.db"
    s1 = MemoryStore(path)
    v1 = s1.schema_version
    s1.close()
    s2 = MemoryStore(path)
    assert s2.schema_version == v1
    # Re-running migrations is a no-op
    assert run_migrations(s2) == v1
    s2.close()


def test_newer_schema_rejected(store: MemoryStore):
    store.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (999, "future", "2030-01-01T00:00:00+00:00"),
    )
    with pytest.raises(MemoryMigrationError):
        run_migrations(store)


def test_entity_check_constraint(store: MemoryStore):
    with pytest.raises(sqlite3.IntegrityError):
        store.execute(
            "INSERT INTO entities (entity_id, canonical_name, entity_type, status, created_at, updated_at)"
            " VALUES ('e1', 'x', 'dragon', 'active', 't', 't')"
        )
