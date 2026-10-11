"""SQLite persistence layer for the Multimodal Memory System.

Design:

* One authoritative SQLite file (WAL journal, foreign keys on).
* Writes are serialized through a dedicated lock — WAL does not make
  arbitrary concurrent writers safe.
* Readers use the same connection under the same lock (SQLite connections
  are not shared across threads without synchronization); the lock scope is
  short because queries are small.
* All public methods are plain sync calls; higher layers schedule them on
  worker threads.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from src.memory.exceptions import MemoryStoreUnavailable
from src.memory.migrations import run_migrations
from src.memory.models import (
    ConsentRecord,
    EntityAssociation,
    EntityType,
    EntityStatus,
    FactSourceType,
    FactStatus,
    MemoryEntity,
    MemoryFact,
    dt_to_iso,
    normalize_name,
    parse_dt,
    utcnow,
)
from src.utils.logger import get_logger

log = get_logger("memory.store")

PRAGMAS = (
    ("foreign_keys", "ON"),
    ("journal_mode", "WAL"),
    ("busy_timeout", "5000"),
    ("synchronous", "NORMAL"),
)


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return path


class MemoryStore:
    """Transactional SQLite store for entities, facts and embeddings."""

    def __init__(self, db_path: str | Path):
        self.db_path = expand_path(db_path)
        ensure_private_dir(self.db_path.parent)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._closed = False
        self.open()

    # -- lifecycle ------------------------------------------------------
    def open(self) -> None:
        with self._lock:
            if self._conn is not None:
                return
            try:
                conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    isolation_level=None,
                )
            except sqlite3.Error as exc:
                raise MemoryStoreUnavailable(f"cannot open memory database: {exc}") from exc
            conn.row_factory = sqlite3.Row
            for pragma, value in PRAGMAS:
                try:
                    conn.execute(f"PRAGMA {pragma}={value}")
                except sqlite3.Error as exc:  # pragma: no cover
                    log.debug("pragma %s failed: %s", pragma, exc)
            self._conn = conn
            run_migrations(self)
            log.info("memory store ready at %s (schema v%s)", self.db_path, self.schema_version)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
                self._conn.close()
                self._conn = None
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def schema_version(self) -> int:
        row = self.execute_fetchone("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        return int(row[0]) if row else 0

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise MemoryStoreUnavailable("memory database is not open")
        return self._conn

    # -- low-level SQL --------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._require()
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def executescript(self, script: str) -> None:
        with self._lock:
            self._require().executescript(script)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._require().execute(sql, tuple(params))

    def execute_fetchone(self, sql: str, params: Iterable[Any] = ()) -> Optional[tuple]:
        with self._lock:
            row = self._require().execute(sql, tuple(params)).fetchone()
            return tuple(row) if row is not None else None

    def execute_fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[tuple]:
        with self._lock:
            rows = self._require().execute(sql, tuple(params)).fetchall()
            return [tuple(r) for r in rows]

    def execute_fetchone_dict(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
        with self._lock:
            cur = self._require().execute(sql, tuple(params))
            row = cur.fetchone()
            if row is None:
                return None
            return {d[0]: row[i] for i, d in enumerate(cur.description)}

    def execute_fetchall_dicts(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            cur = self._require().execute(sql, tuple(params))
            rows = cur.fetchall()
            if not rows:
                return []
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in rows]

    # -- backup ---------------------------------------------------------
    def backup(self, dest: str | Path) -> Path:
        dest_path = expand_path(dest)
        ensure_private_dir(dest_path.parent)
        with self._lock:
            conn = self._require()
            target = sqlite3.connect(str(dest_path))
            try:
                conn.backup(target)
            finally:
                target.close()
        try:
            os.chmod(dest_path, 0o600)
        except OSError:  # pragma: no cover
            pass
        return dest_path

    def vacuum(self) -> None:
        with self._lock:
            self._require().execute("VACUUM")

    # -- entities -------------------------------------------------------
    def upsert_entity(self, entity: MemoryEntity) -> MemoryEntity:
        now = dt_to_iso(entity.updated_at) or utcnow().isoformat()
        with self.transaction():
            self.execute(
                """
                INSERT INTO entities (
                    entity_id, canonical_name, entity_type, status,
                    created_at, updated_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    canonical_name=excluded.canonical_name,
                    entity_type=excluded.entity_type,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    last_seen_at=COALESCE(excluded.last_seen_at, entities.last_seen_at)
                """,
                (
                    entity.entity_id,
                    entity.canonical_name,
                    entity.entity_type.value,
                    entity.status.value,
                    dt_to_iso(entity.created_at),
                    now,
                    dt_to_iso(entity.last_seen_at),
                ),
            )
            for alias in entity.aliases:
                self._insert_alias_locked(entity.entity_id, alias)
        return entity

    def _insert_alias_locked(self, entity_id: str, alias: str) -> None:
        alias = (alias or "").strip()
        if not alias:
            return
        normalized = normalize_name(alias)
        now = utcnow().isoformat()
        self.execute(
            """
            INSERT INTO entity_aliases (alias_id, entity_id, alias, normalized_alias, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (f"al_{entity_id}_{abs(hash(normalized)) % (10**10)}", entity_id, alias, normalized, now),
        )

    def add_alias(self, entity_id: str, alias: str) -> None:
        with self.transaction():
            self._insert_alias_locked(entity_id, alias)

    def get_entity(self, entity_id: str) -> Optional[MemoryEntity]:
        row = self.execute_fetchone(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        )
        return self._row_to_entity(row) if row else None

    def find_entities_by_name(self, name: str, entity_type: Optional[EntityType] = None) -> list[MemoryEntity]:
        normalized = normalize_name(name)
        sql = """
            SELECT DISTINCT e.* FROM entities e
            LEFT JOIN entity_aliases a ON a.entity_id = e.entity_id
            WHERE (LOWER(e.canonical_name) = ? OR a.normalized_alias = ?)
        """
        params: list[Any] = [normalized, normalized]
        if entity_type is not None:
            sql += " AND e.entity_type = ?"
            params.append(entity_type.value)
        sql += " ORDER BY e.created_at"
        rows = self.execute_fetchall(sql, params)
        out = []
        for row in rows:
            entity = self._row_to_entity(row)
            entity.aliases = self.aliases_for(entity.entity_id)
            out.append(entity)
        return out

    def list_entities(
        self,
        entity_type: Optional[EntityType] = None,
        status: Optional[EntityStatus] = None,
        limit: int = 200,
    ) -> list[MemoryEntity]:
        sql = "SELECT * FROM entities WHERE 1=1"
        params: list[Any] = []
        if entity_type is not None:
            sql += " AND entity_type = ?"
            params.append(entity_type.value)
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        out = []
        for row in self.execute_fetchall(sql, params):
            entity = self._row_to_entity(row)
            entity.aliases = self.aliases_for(entity.entity_id)
            out.append(entity)
        return out

    def aliases_for(self, entity_id: str) -> list[str]:
        rows = self.execute_fetchall(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY created_at",
            (entity_id,),
        )
        return [r[0] for r in rows]

    def touch_entity(self, entity_id: str, when: Optional[datetime] = None) -> None:
        stamp = dt_to_iso(when or utcnow())
        self.execute(
            "UPDATE entities SET last_seen_at = ?, updated_at = ? WHERE entity_id = ?",
            (stamp, stamp, entity_id),
        )

    def delete_entity(self, entity_id: str) -> dict[str, int]:
        """Delete an entity and everything that cascades from it."""
        counts = {"entities": 0, "face_embeddings": 0, "object_embeddings": 0, "facts": 0}
        crop_paths = self.execute_fetchall(
            "SELECT crop_path FROM object_embeddings WHERE entity_id = ? AND crop_path IS NOT NULL",
            (entity_id,),
        )
        with self.transaction():
            counts["face_embeddings"] = self.execute(
                "DELETE FROM face_embeddings WHERE entity_id = ?", (entity_id,)
            ).rowcount
            counts["object_embeddings"] = self.execute(
                "DELETE FROM object_embeddings WHERE entity_id = ?", (entity_id,)
            ).rowcount
            counts["facts"] = self.execute(
                "DELETE FROM declarative_memory WHERE subject_entity_id = ?", (entity_id,)
            ).rowcount
            counts["entities"] = self.execute(
                "DELETE FROM entities WHERE entity_id = ?", (entity_id,)
            ).rowcount
        for (path,) in crop_paths:
            _safe_unlink(path)
        return counts

    def _row_to_entity(self, row: tuple) -> MemoryEntity:
        d = self._row_to_dict(row, "entities")
        return MemoryEntity(
            entity_id=d["entity_id"],
            canonical_name=d["canonical_name"],
            entity_type=EntityType(d["entity_type"]),
            status=EntityStatus(d["status"]),
            created_at=parse_dt(d["created_at"]) or utcnow(),
            updated_at=parse_dt(d["updated_at"]) or utcnow(),
            last_seen_at=parse_dt(d.get("last_seen_at")),
        )

    @staticmethod
    def _row_to_dict(row: tuple, table: str) -> dict[str, Any]:
        # column order matches CREATE TABLE for the tables we select with *
        columns = {
            "entities": [
                "entity_id", "canonical_name", "entity_type", "status",
                "created_at", "updated_at", "last_seen_at",
            ],
            "declarative_memory": [
                "fact_id", "subject_entity_id", "predicate", "object_value",
                "value_type", "raw_transcript", "source_turn_id", "source_type",
                "asserted_by_entity_id", "status", "supersedes_fact_id",
                "extraction_method", "created_at", "updated_at",
            ],
        }[table]
        return dict(zip(columns, row))

    # -- facts ----------------------------------------------------------
    def insert_fact(self, fact: MemoryFact) -> MemoryFact:
        with self.transaction():
            self._insert_fact_locked(fact)
        return fact

    def _insert_fact_locked(self, fact: MemoryFact) -> None:
        self.execute(
            """
            INSERT INTO declarative_memory (
                fact_id, subject_entity_id, predicate, object_value, value_type,
                raw_transcript, source_turn_id, source_type, asserted_by_entity_id,
                status, supersedes_fact_id, extraction_method, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fact_id) DO UPDATE SET
                status=excluded.status,
                updated_at=excluded.updated_at,
                supersedes_fact_id=excluded.supersedes_fact_id
            """,
            (
                fact.fact_id,
                fact.subject_entity_id,
                fact.predicate,
                fact.object_value,
                fact.value_type,
                fact.source_transcript,
                fact.source_turn_id,
                fact.source_type.value,
                fact.asserted_by_entity_id,
                fact.status.value,
                fact.supersedes_fact_id,
                fact.extraction_method,
                dt_to_iso(fact.created_at),
                dt_to_iso(fact.updated_at),
            ),
        )

    def supersede_fact(self, old_fact_id: str, new_fact: MemoryFact) -> MemoryFact:
        with self.transaction():
            row = self.execute_fetchone(
                "SELECT status FROM declarative_memory WHERE fact_id = ?", (old_fact_id,)
            )
            if row is None:
                raise KeyError(old_fact_id)
            new_fact.supersedes_fact_id = old_fact_id
            self.execute(
                """
                UPDATE declarative_memory
                SET status = 'superseded', updated_at = ?
                WHERE fact_id = ?
                """,
                (utcnow().isoformat(), old_fact_id),
            )
            self._insert_fact_locked(new_fact)
            self.record_revision(
                entity_id=new_fact.subject_entity_id,
                fact_id=new_fact.fact_id,
                action="supersede",
                detail=f"supersedes {old_fact_id}",
            )
        return new_fact

    def get_fact(self, fact_id: str) -> Optional[MemoryFact]:
        row = self.execute_fetchone(
            "SELECT * FROM declarative_memory WHERE fact_id = ?", (fact_id,)
        )
        return self._row_to_fact(row) if row else None

    def facts_for_entity(
        self,
        entity_id: str,
        *,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[MemoryFact]:
        sql = "SELECT * FROM declarative_memory WHERE subject_entity_id = ?"
        params: list[Any] = [entity_id]
        if active_only:
            sql += " AND status = 'active'"
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        return [self._row_to_fact(r) for r in self.execute_fetchall(sql, params)]

    def search_facts(
        self,
        *,
        needle: str = "",
        predicate: Optional[str] = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[MemoryFact]:
        sql = "SELECT * FROM declarative_memory WHERE 1=1"
        params: list[Any] = []
        if active_only:
            sql += " AND status = 'active'"
        if predicate:
            sql += " AND LOWER(predicate) = ?"
            params.append(predicate.lower())
        if needle:
            words = [w for w in str(needle).lower().split() if len(w) > 2]
            phrase = f"%{needle.lower()}%"
            sql += (
                " AND (LOWER(object_value) LIKE ? OR LOWER(raw_transcript) LIKE ?"
                " OR LOWER(predicate) LIKE ?"
            )
            params.extend([phrase, phrase, phrase])
            if words:
                sql += " OR " + " OR ".join(
                    "(LOWER(object_value) LIKE ? OR LOWER(predicate) LIKE ?)" for _ in words
                )
                for w in words:
                    params.extend([f"%{w}%", f"%{w}%"])
            sql += ")"
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        return [self._row_to_fact(r) for r in self.execute_fetchall(sql, params)]

    def list_facts(self, *, active_only: bool = True, limit: int = 200) -> list[MemoryFact]:
        return self.search_facts(active_only=active_only, limit=limit)

    def delete_fact(self, fact_id: str) -> bool:
        with self.transaction():
            cur = self.execute("DELETE FROM declarative_memory WHERE fact_id = ?", (fact_id,))
            return cur.rowcount > 0

    def _row_to_fact(self, row: tuple) -> MemoryFact:
        d = self._row_to_dict(row, "declarative_memory")
        return MemoryFact(
            fact_id=d["fact_id"],
            subject_entity_id=d["subject_entity_id"],
            predicate=d["predicate"],
            object_value=d["object_value"],
            value_type=d["value_type"],
            source_transcript=d["raw_transcript"],
            source_turn_id=d["source_turn_id"],
            source_type=FactSourceType(d["source_type"]),
            asserted_by_entity_id=d["asserted_by_entity_id"],
            status=FactStatus(d["status"]),
            supersedes_fact_id=d["supersedes_fact_id"],
            extraction_method=d["extraction_method"],
            created_at=parse_dt(d["created_at"]) or utcnow(),
            updated_at=parse_dt(d["updated_at"]) or utcnow(),
        )

    # -- embeddings -----------------------------------------------------
    def insert_face_embedding(
        self,
        entity_id: str,
        blob: bytes,
        dimension: int,
        model_id: str,
        *,
        quality_score: Optional[float] = None,
        embedding_id: Optional[str] = None,
    ) -> str:
        from src.memory.models import new_id

        emb_id = embedding_id or new_id("femb")
        with self.transaction():
            self.execute(
                """
                INSERT INTO face_embeddings (
                    embedding_id, entity_id, embedding_data, embedding_dimension,
                    embedding_dtype, model_id, quality_score, created_at
                ) VALUES (?, ?, ?, ?, 'float32', ?, ?, ?)
                """,
                (
                    emb_id,
                    entity_id,
                    blob,
                    int(dimension),
                    model_id,
                    quality_score,
                    utcnow().isoformat(),
                ),
            )
        return emb_id

    def insert_object_embedding(
        self,
        entity_id: str,
        blob: bytes,
        dimension: int,
        model_id: str,
        *,
        crop_path: Optional[str] = None,
        embedding_id: Optional[str] = None,
    ) -> str:
        from src.memory.models import new_id

        emb_id = embedding_id or new_id("oemb")
        with self.transaction():
            self.execute(
                """
                INSERT INTO object_embeddings (
                    embedding_id, entity_id, embedding_data, embedding_dimension,
                    embedding_dtype, model_id, crop_path, created_at
                ) VALUES (?, ?, ?, ?, 'float32', ?, ?, ?)
                """,
                (
                    emb_id,
                    entity_id,
                    blob,
                    int(dimension),
                    model_id,
                    crop_path,
                    utcnow().isoformat(),
                ),
            )
        return emb_id

    def all_face_embeddings(
        self, model_id: Optional[str] = None
    ) -> list[tuple[str, str, bytes, int, str, Optional[float]]]:
        sql = "SELECT embedding_id, entity_id, embedding_data, embedding_dimension, model_id, quality_score FROM face_embeddings"
        params: tuple = ()
        if model_id:
            sql += " WHERE model_id = ?"
            params = (model_id,)
        return list(self.execute_fetchall(sql, params))

    def all_object_embeddings(
        self, model_id: Optional[str] = None
    ) -> list[tuple[str, str, bytes, int, str, Optional[str]]]:
        sql = "SELECT embedding_id, entity_id, embedding_data, embedding_dimension, model_id, crop_path FROM object_embeddings"
        params: tuple = ()
        if model_id:
            sql += " WHERE model_id = ?"
            params = (model_id,)
        return list(self.execute_fetchall(sql, params))

    def delete_face_embeddings_for(self, entity_id: str) -> int:
        with self.transaction():
            return self.execute(
                "DELETE FROM face_embeddings WHERE entity_id = ?", (entity_id,)
            ).rowcount

    def delete_object_embeddings_for(self, entity_id: str) -> int:
        with self.transaction():
            return self.execute(
                "DELETE FROM object_embeddings WHERE entity_id = ?", (entity_id,)
            ).rowcount

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in (
            "entities",
            "entity_aliases",
            "face_embeddings",
            "object_embeddings",
            "declarative_memory",
            "observations",
            "entity_associations",
            "enrollment_consent",
            "identity_bindings",
        ):
            row = self.execute_fetchone(f"SELECT COUNT(*) FROM {table}")
            out[table] = int(row[0]) if row else 0
        return out

    def database_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.db_path) + suffix)
            if p.is_file():
                total += p.stat().st_size
        return total

    # -- observations ---------------------------------------------------
    def insert_observation(
        self,
        observation_id: str,
        frame_id: str,
        kind: str,
        bbox: tuple[int, int, int, int],
        *,
        quality_score: Optional[float] = None,
        source_model: str = "",
        candidate_label: Optional[str] = None,
        request_id: str = "",
    ) -> None:
        with self.transaction():
            self.execute(
                """
                INSERT INTO observations (
                    observation_id, frame_id, kind, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    quality_score, source_model, candidate_label, captured_at, request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO NOTHING
                """,
                (
                    observation_id,
                    frame_id,
                    kind,
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    quality_score,
                    source_model,
                    candidate_label,
                    utcnow().isoformat(),
                    request_id,
                ),
            )

    def link_observation(self, entity_id: str, observation_id: str, role: str = "enrollment") -> None:
        with self.transaction():
            self.execute(
                """
                INSERT INTO entity_observations (entity_id, observation_id, role, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (entity_id, observation_id, role, utcnow().isoformat()),
            )

    # -- associations ---------------------------------------------------
    def add_association(self, assoc: EntityAssociation) -> EntityAssociation:
        with self.transaction():
            self.execute(
                """
                INSERT INTO entity_associations (
                    association_id, source_entity_id, target_entity_id, relation,
                    weight, source_transcript, source_turn_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(association_id) DO NOTHING
                """,
                (
                    assoc.association_id,
                    assoc.source_entity_id,
                    assoc.target_entity_id,
                    assoc.relation,
                    assoc.weight,
                    assoc.source_transcript,
                    assoc.source_turn_id,
                    dt_to_iso(assoc.created_at),
                ),
            )
        return assoc

    def associations_for(self, entity_id: str) -> list[EntityAssociation]:
        rows = self.execute_fetchall(
            "SELECT * FROM entity_associations WHERE source_entity_id = ? OR target_entity_id = ?",
            (entity_id, entity_id),
        )
        out = []
        for r in rows:
            (
                association_id,
                source_entity_id,
                target_entity_id,
                relation,
                weight,
                source_transcript,
                source_turn_id,
                created_at,
            ) = r
            out.append(
                EntityAssociation(
                    association_id=association_id,
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                    relation=relation,
                    weight=weight,
                    source_transcript=source_transcript,
                    source_turn_id=source_turn_id,
                    created_at=parse_dt(created_at) or utcnow(),
                )
            )
        return out

    # -- consent & bindings ---------------------------------------------
    def save_consent(self, consent: ConsentRecord) -> ConsentRecord:
        with self.transaction():
            self.execute(
                """
                INSERT INTO enrollment_consent (
                    consent_id, subject_description, scope, granted, granted_at,
                    expires_at, method, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(consent_id) DO UPDATE SET
                    granted=excluded.granted,
                    granted_at=excluded.granted_at,
                    expires_at=excluded.expires_at,
                    metadata=excluded.metadata
                """,
                (
                    consent.consent_id,
                    consent.subject_description,
                    consent.scope,
                    1 if consent.granted else 0,
                    dt_to_iso(consent.granted_at),
                    dt_to_iso(consent.expires_at),
                    consent.method,
                    json.dumps(consent.metadata),
                ),
            )
        return consent

    def get_consent(self, consent_id: str) -> Optional[ConsentRecord]:
        row = self.execute_fetchone(
            "SELECT * FROM enrollment_consent WHERE consent_id = ?", (consent_id,)
        )
        if row is None:
            return None
        (
            consent_id,
            subject_description,
            scope,
            granted,
            granted_at,
            expires_at,
            method,
            metadata,
        ) = row
        return ConsentRecord(
            consent_id=consent_id,
            subject_description=subject_description,
            scope=scope,
            granted=bool(granted),
            granted_at=parse_dt(granted_at),
            expires_at=parse_dt(expires_at),
            method=method,
            metadata=json.loads(metadata or "{}"),
        )

    def record_identity_binding(
        self, entity_id: str, consent_id: Optional[str], method: str = "explicit_enrollment"
    ) -> str:
        from src.memory.models import new_id

        binding_id = new_id("bind")
        with self.transaction():
            self.execute(
                """
                INSERT INTO identity_bindings (binding_id, entity_id, consent_id, method, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (binding_id, entity_id, consent_id, method, utcnow().isoformat()),
            )
        return binding_id

    def revoke_identity_bindings(self, entity_id: str) -> int:
        with self.transaction():
            return self.execute(
                "UPDATE identity_bindings SET revoked_at = ? WHERE entity_id = ? AND revoked_at IS NULL",
                (utcnow().isoformat(), entity_id),
            ).rowcount

    # -- revisions ------------------------------------------------------
    def record_revision(
        self,
        *,
        entity_id: Optional[str],
        fact_id: Optional[str],
        action: str,
        detail: str = "",
    ) -> None:
        from src.memory.models import new_id

        self.execute(
            """
            INSERT INTO memory_revisions (revision_id, entity_id, fact_id, action, detail, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (new_id("rev"), entity_id, fact_id, action, detail, utcnow().isoformat()),
        )

    # -- merge ----------------------------------------------------------
    def merge_entities(self, source_id: str, target_id: str) -> dict[str, int]:
        """Reassign facts, embeddings, aliases and associations; drop source."""
        if source_id == target_id:
            raise ValueError("cannot merge an entity with itself")
        source = self.get_entity(source_id)
        target = self.get_entity(target_id)
        if source is None or target is None:
            raise KeyError("entity not found")
        if source.entity_type != target.entity_type:
            raise ValueError("entity type mismatch")
        stats = {"facts": 0, "face_embeddings": 0, "object_embeddings": 0, "associations": 0}
        with self.transaction():
            stats["facts"] = self.execute(
                "UPDATE declarative_memory SET subject_entity_id = ?, updated_at = ? WHERE subject_entity_id = ?",
                (target_id, utcnow().isoformat(), source_id),
            ).rowcount
            stats["face_embeddings"] = self.execute(
                "UPDATE face_embeddings SET entity_id = ? WHERE entity_id = ?",
                (target_id, source_id),
            ).rowcount
            stats["object_embeddings"] = self.execute(
                "UPDATE object_embeddings SET entity_id = ? WHERE entity_id = ?",
                (target_id, source_id),
            ).rowcount
            stats["associations"] = self.execute(
                "UPDATE entity_associations SET source_entity_id = ? WHERE source_entity_id = ?",
                (target_id, source_id),
            ).rowcount
            self.execute(
                "UPDATE entity_associations SET target_entity_id = ? WHERE target_entity_id = ?",
                (target_id, source_id),
            )
            for alias in source.aliases:
                self._insert_alias_locked(target_id, alias)
            self.execute("DELETE FROM entities WHERE entity_id = ?", (source_id,))
            self.record_revision(
                entity_id=target_id,
                fact_id=None,
                action="merge",
                detail=f"merged {source_id} into {target_id}",
            )
        return stats

    # ------------------------------------------------------------------
    # persistent episodic & conversational memory (PECM)
    # ------------------------------------------------------------------
    # -- sessions -------------------------------------------------------
    def upsert_session(self, session_id: str, *, started_at: Optional[str] = None,
                       metadata: Optional[dict] = None) -> None:
        now = utcnow().isoformat()
        with self.transaction():
            self.execute(
                """
                INSERT INTO conversation_sessions (session_id, started_at, session_metadata)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    session_metadata = COALESCE(excluded.session_metadata, session_metadata)
                """,
                (session_id, started_at or now, json.dumps(metadata or {})),
            )

    def close_session(self, session_id: str, *, ended_at: Optional[str] = None) -> None:
        with self.transaction():
            self.execute(
                "UPDATE conversation_sessions SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL",
                (ended_at or utcnow().isoformat(), session_id),
            )

    def get_session(self, session_id: str) -> Optional[dict]:
        return self.execute_fetchone_dict(
            "SELECT * FROM conversation_sessions WHERE session_id = ?", (session_id,)
        )

    def list_sessions(self, limit: int = 50) -> list[dict]:
        return self.execute_fetchall_dicts(
            "SELECT * FROM conversation_sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )

    # -- events ---------------------------------------------------------
    def insert_event(self, event) -> bool:
        """Insert a ConversationEvent; idempotent on event_id.

        Returns True if a new row was written, False on duplicate.
        FTS index row is written in the same transaction.
        """
        from src.memory.event_models import ConversationEvent  # local import

        assert isinstance(event, ConversationEvent)
        with self.transaction():
            # FTS rowid: reuse event's or allocate a fresh one.
            if event.fts_rowid is None:
                row = self.execute_fetchone(
                    "SELECT COALESCE(MAX(fts_rowid), 0) + 1 FROM conversation_events"
                )
                event.fts_rowid = int(row[0]) if row else 1
            cur = self.execute(
                """
                INSERT INTO conversation_events (
                    event_id, session_id, turn_id, event_sequence, event_type,
                    content, source_model, source_tool, frame_id, timestamp,
                    metadata_json, persistence_status, fts_rowid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    event.event_id, event.session_id, event.turn_id,
                    event.event_sequence, event.event_type.value, event.content,
                    event.source_model, event.source_tool, event.frame_id,
                    event.timestamp.isoformat(), event.metadata_json(),
                    event.persistence_status, event.fts_rowid,
                ),
            )
            if cur.rowcount == 0:
                return False  # duplicate — FTS row already present
            self.execute(
                "INSERT INTO conversation_fts(rowid, content) VALUES (?, ?)",
                (event.fts_rowid, event.content),
            )
            return True

    def get_event(self, event_id: str):
        from src.memory.event_models import ConversationEvent

        row = self.execute_fetchone_dict(
            "SELECT * FROM conversation_events WHERE event_id = ?", (event_id,)
        )
        return ConversationEvent.from_row(row) if row else None

    def recent_events(self, *, session_id: Optional[str] = None,
                      event_type: Optional[str] = None, limit: int = 20) -> list:
        from src.memory.event_models import ConversationEvent

        clauses, params = [], []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.execute_fetchall_dicts(
            f"SELECT * FROM conversation_events {where} "
            f"ORDER BY timestamp DESC, event_sequence DESC LIMIT ?",
            (*params, limit),
        )
        return [ConversationEvent.from_row(r) for r in rows]

    def events_for_turn(self, turn_id: str) -> list:
        from src.memory.event_models import ConversationEvent

        rows = self.execute_fetchall_dicts(
            "SELECT * FROM conversation_events WHERE turn_id = ? "
            "ORDER BY event_sequence ASC",
            (turn_id,),
        )
        return [ConversationEvent.from_row(r) for r in rows]

    def search_events_fts(self, query: str, *, limit: int = 20,
                          session_id: Optional[str] = None) -> list:
        """FTS5 keyword search; BM25-ranked, best (lowest) rank first."""
        from src.memory.event_models import ConversationEvent

        # Quote each token for the FTS query grammar; fall back to no results
        # when the query contains nothing searchable.
        tokens = [t for t in "".join(
            ch if ch.isalnum() or ch.isspace() else " " for ch in query
        ).split() if t]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in tokens[:12])
        rows = self.execute_fetchall_dicts(
            """
            SELECT e.* FROM conversation_fts f
            JOIN conversation_events e ON e.fts_rowid = f.rowid
            WHERE conversation_fts MATCH ?
            ORDER BY bm25(conversation_fts)
            LIMIT ?
            """,
            (fts_query, limit * 3),
        )
        events = [ConversationEvent.from_row(r) for r in rows]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        return events[:limit]

    def count_events(self, session_id: Optional[str] = None) -> int:
        if session_id:
            row = self.execute_fetchone(
                "SELECT COUNT(*) FROM conversation_events WHERE session_id = ?",
                (session_id,),
            )
        else:
            row = self.execute_fetchone("SELECT COUNT(*) FROM conversation_events")
        return int(row[0]) if row else 0

    def delete_event(self, event_id: str) -> bool:
        with self.transaction():
            row = self.execute_fetchone(
                "SELECT fts_rowid FROM conversation_events WHERE event_id = ?",
                (event_id,),
            )
            if row is None:
                return False
            fts_rowid = row[0]
            self.execute("DELETE FROM conversation_events WHERE event_id = ?", (event_id,))
            if fts_rowid is not None:
                self.execute("DELETE FROM conversation_fts WHERE rowid = ?", (fts_rowid,))
            self.execute(
                "DELETE FROM text_memory_embeddings WHERE source_id = ?", (event_id,)
            )
            return True

    def delete_session_events(self, session_id: str) -> int:
        with self.transaction():
            rows = self.execute_fetchall_dicts(
                "SELECT fts_rowid, event_id FROM conversation_events WHERE session_id = ?",
                (session_id,),
            )
            for fts_rowid, event_id in rows:
                if fts_rowid is not None:
                    self.execute("DELETE FROM conversation_fts WHERE rowid = ?", (fts_rowid,))
                self.execute(
                    "DELETE FROM text_memory_embeddings WHERE source_id = ?", (event_id,)
                )
            cur = self.execute(
                "DELETE FROM conversation_events WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount

    # -- episodes -------------------------------------------------------
    def insert_episode(self, episode) -> bool:
        from src.memory.event_models import Episode

        assert isinstance(episode, Episode)
        with self.transaction():
            cur = self.execute(
                """
                INSERT INTO episodic_memory (
                    episode_id, session_id, turn_id, episode_summary,
                    start_timestamp, end_timestamp, source_event_ids_json,
                    frame_id, episode_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO NOTHING
                """,
                (
                    episode.episode_id, episode.session_id, episode.turn_id,
                    episode.episode_summary, episode.start_timestamp.isoformat(),
                    episode.end_timestamp.isoformat(), episode.source_event_ids_json(),
                    episode.frame_id, episode.episode_type.value,
                    episode.created_at.isoformat(),
                ),
            )
            return cur.rowcount > 0

    def episode_for_turn(self, turn_id: str):
        from src.memory.event_models import Episode

        row = self.execute_fetchone_dict(
            "SELECT * FROM episodic_memory WHERE turn_id = ? ORDER BY created_at DESC LIMIT 1",
            (turn_id,),
        )
        return Episode.from_row(row) if row else None

    def recent_episodes(self, *, session_id: Optional[str] = None,
                        episode_type: Optional[str] = None, limit: int = 10) -> list:
        from src.memory.event_models import Episode

        clauses, params = [], []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if episode_type:
            clauses.append("episode_type = ?")
            params.append(episode_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.execute_fetchall_dicts(
            f"SELECT * FROM episodic_memory {where} "
            f"ORDER BY start_timestamp DESC LIMIT ?",
            (*params, limit),
        )
        return [Episode.from_row(r) for r in rows]

    def episodes_fts(self, query: str, *, limit: int = 10) -> list:
        from src.memory.event_models import Episode

        tokens = [t for t in "".join(
            ch if ch.isalnum() or ch.isspace() else " " for ch in query
        ).split() if t]
        if not tokens:
            return []
        # Episodes are searched via their source events' FTS rows joined on
        # turn_id; also LIKE-match the summary itself as a fallback.
        likes = " OR ".join("LOWER(ep.episode_summary) LIKE ?" for _ in tokens[:8])
        params = [f"%{t.lower()}%" for t in tokens[:8]]
        rows = self.execute_fetchall_dicts(
            f"SELECT ep.* FROM episodic_memory ep WHERE {likes} "
            f"ORDER BY ep.start_timestamp DESC LIMIT ?",
            (*params, limit),
        )
        return [Episode.from_row(r) for r in rows]

    def delete_episode(self, episode_id: str) -> bool:
        with self.transaction():
            cur = self.execute(
                "DELETE FROM episodic_memory WHERE episode_id = ?", (episode_id,)
            )
            return cur.rowcount > 0

    def delete_session_episodes(self, session_id: str) -> int:
        with self.transaction():
            cur = self.execute(
                "DELETE FROM episodic_memory WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount

    # -- text embeddings ------------------------------------------------
    def insert_text_embedding(self, embedding_id: str, source_id: str, source_type: str,
                              model_id: str, dimension: int, blob: bytes) -> bool:
        with self.transaction():
            cur = self.execute(
                """
                INSERT INTO text_memory_embeddings (
                    embedding_id, source_id, source_type, embedding_model_id,
                    embedding_dimension, embedding_data, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(embedding_id) DO NOTHING
                """,
                (embedding_id, source_id, source_type, model_id, dimension, blob,
                 utcnow().isoformat()),
            )
            return cur.rowcount > 0

    def text_embeddings_for_source(self, source_id: str) -> list[dict]:
        return self.execute_fetchall_dicts(
            "SELECT * FROM text_memory_embeddings WHERE source_id = ?", (source_id,)
        )

    def all_text_embeddings(self, model_id: Optional[str] = None) -> list[dict]:
        if model_id:
            return self.execute_fetchall_dicts(
                "SELECT * FROM text_memory_embeddings WHERE embedding_model_id = ?",
                (model_id,),
            )
        return self.execute_fetchall_dicts("SELECT * FROM text_memory_embeddings")

    def delete_text_embeddings_for(self, source_id: str) -> int:
        with self.transaction():
            cur = self.execute(
                "DELETE FROM text_memory_embeddings WHERE source_id = ?", (source_id,)
            )
            return cur.rowcount

    # -- indexing jobs --------------------------------------------------
    def upsert_index_job(self, job) -> None:
        from src.memory.event_models import IndexJob

        assert isinstance(job, IndexJob)
        now = utcnow().isoformat()
        with self.transaction():
            self.execute(
                """
                INSERT INTO memory_index_jobs (
                    job_id, job_type, source_id, status, attempts,
                    last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    attempts = excluded.attempts,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (job.job_id, job.job_type, job.source_id, job.status,
                 job.attempts, job.last_error, job.created_at.isoformat(), now),
            )

    def pending_index_jobs(self, *, limit: int = 50) -> list:
        from src.memory.event_models import IndexJob

        rows = self.execute_fetchall_dicts(
            "SELECT * FROM memory_index_jobs WHERE status = 'pending' "
            "ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        out = []
        for d in rows:
            out.append(IndexJob(
                job_id=d["job_id"], job_type=d["job_type"], source_id=d["source_id"],
                status=d["status"], attempts=int(d["attempts"] or 0),
                last_error=d.get("last_error"),
                created_at=d["created_at"], updated_at=d["updated_at"],
            ))
        return out

    def finish_index_job(self, job_id: str, *, status: str = "done",
                         error: Optional[str] = None) -> None:
        with self.transaction():
            self.execute(
                "UPDATE memory_index_jobs SET status = ?, last_error = ?, updated_at = ? "
                "WHERE job_id = ?",
                (status, error, utcnow().isoformat(), job_id),
            )

    def count_pending_index_jobs(self) -> int:
        row = self.execute_fetchone(
            "SELECT COUNT(*) FROM memory_index_jobs WHERE status = 'pending'"
        )
        return int(row[0]) if row else 0

    # -- rebuild helpers -------------------------------------------------
    def rebuild_fts(self) -> int:
        """Drop and repopulate conversation_fts from the events table."""
        with self.transaction():
            self.execute("DELETE FROM conversation_fts")
            rows = self.execute_fetchall_dicts(
                "SELECT fts_rowid, content FROM conversation_events "
                "WHERE fts_rowid IS NOT NULL ORDER BY fts_rowid"
            )
            for r in rows:
                self.execute(
                    "INSERT INTO conversation_fts(rowid, content) VALUES (?, ?)",
                    (r["fts_rowid"], r["content"]),
                )
            return len(rows)

    def clear_all_conversation_memory(self) -> dict[str, int]:
        """Delete all PECM data (events, episodes, sessions, embeddings, FTS)."""
        with self.transaction():
            counts = {}
            counts["events"] = self.execute(
                "DELETE FROM conversation_events"
            ).rowcount
            counts["episodes"] = self.execute(
                "DELETE FROM episodic_memory"
            ).rowcount
            counts["sessions"] = self.execute(
                "DELETE FROM conversation_sessions"
            ).rowcount
            counts["embeddings"] = self.execute(
                "DELETE FROM text_memory_embeddings"
            ).rowcount
            counts["jobs"] = self.execute(
                "DELETE FROM memory_index_jobs"
            ).rowcount
            self.execute("DELETE FROM conversation_fts")
            return counts

    def counts_ex(self) -> dict[str, int]:
        """Extended counts including PECM tables."""
        base = self.counts()
        for table, key in (
            ("conversation_sessions", "sessions"),
            ("conversation_events", "conversation_events"),
            ("episodic_memory", "episodes"),
            ("text_memory_embeddings", "text_embeddings"),
            ("memory_index_jobs", "pending_index_jobs"),
        ):
            try:
                row = self.execute_fetchone(f"SELECT COUNT(*) FROM {table}")
                base[key] = int(row[0]) if row else 0
            except sqlite3.Error:
                base[key] = 0
        return base


def _safe_unlink(path: str | Path) -> None:
    try:
        p = expand_path(path)
        if p.is_file():
            p.unlink()
    except OSError as exc:  # pragma: no cover
        log.debug("could not remove asset %s: %s", path, exc)
