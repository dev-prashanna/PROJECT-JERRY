"""Schema migrations for the MMS SQLite database.

Each migration is (version, name, callable(store)).  Migrations run in order
inside a transaction where SQLite allows it; applied versions are recorded so
startup is idempotent.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.memory.exceptions import MemoryMigrationError

SCHEMA_DIR = Path(__file__).resolve().parents[1]
SCHEMA_SQL = SCHEMA_DIR / "schema.sql"


def _apply_initial(store) -> None:
    # executescript() would implicitly COMMIT the enclosing migration
    # transaction, so run statements one by one instead.
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    for statement in _split_statements(sql):
        store.execute(statement)


def _split_statements(script: str) -> list[str]:
    statements: list[str] = []
    for chunk in script.split(";"):
        # strip full-line comments; keep the statement body
        lines = [
            line for line in chunk.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        body = "\n".join(lines).strip()
        if body:
            statements.append(body)
    return statements


def _apply_register_default_models(store) -> None:
    """Record the embedding model families the code expects to use."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        ("opencv-sface", "face", 128, "OpenCV FaceRecognizerSF (SFace), 128-D"),
        ("face_recognition-api", "face", 128, "dlib face_recognition API, 128-D"),
        ("sift-color-v1", "object", 512, "OpenCV SIFT + HSV histogram baseline"),
        ("clip-vit-b32", "object", 512, "CLIP ViT-B/32 image embedding"),
    ]
    for model_id, kind, dim, notes in rows:
        store.execute(
            """
            INSERT INTO embedding_models (model_id, kind, dimension, notes, registered_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO NOTHING
            """,
            (model_id, kind, dim, notes, now),
        )


def _apply_episodic_memory(store) -> None:
    """Persistent episodic + conversational memory tables (PECM)."""
    statements = [
        # -- conversation sessions -------------------------------------
        """
        CREATE TABLE IF NOT EXISTS conversation_sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            session_metadata TEXT
        )
        """,
        # -- conversation events (authoritative history) ----------------
        """
        CREATE TABLE IF NOT EXISTS conversation_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            event_sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source_model TEXT,
            source_tool TEXT,
            frame_id TEXT,
            timestamp TEXT NOT NULL,
            metadata_json TEXT,
            persistence_status TEXT NOT NULL DEFAULT 'complete',
            fts_rowid INTEGER,
            FOREIGN KEY(session_id) REFERENCES conversation_sessions(session_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_conv_events_session ON conversation_events(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_conv_events_turn ON conversation_events(turn_id)",
        "CREATE INDEX IF NOT EXISTS idx_conv_events_ts ON conversation_events(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_conv_events_type ON conversation_events(event_type)",
        # JOIN key for conversation_fts → conversation_events (BM25 ranking
        # without this index degrades to a table scan per match at ~1k+ rows)
        "CREATE INDEX IF NOT EXISTS idx_conv_events_fts_rowid ON conversation_events(fts_rowid)",
        # -- episodic memory (derived summaries) ------------------------
        """
        CREATE TABLE IF NOT EXISTS episodic_memory (
            episode_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            episode_summary TEXT NOT NULL,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT NOT NULL,
            source_event_ids_json TEXT NOT NULL,
            frame_id TEXT,
            episode_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES conversation_sessions(session_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodic_memory(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_episodes_turn ON episodic_memory(turn_id)",
        "CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodic_memory(episode_type)",
        "CREATE INDEX IF NOT EXISTS idx_episodes_start ON episodic_memory(start_timestamp)",
        # -- text memory embeddings --------------------------------------
        """
        CREATE TABLE IF NOT EXISTS text_memory_embeddings (
            embedding_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            embedding_model_id TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            embedding_data BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_text_emb_source ON text_memory_embeddings(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_text_emb_model ON text_memory_embeddings(embedding_model_id)",
        # -- FTS5 full-text search over conversation content --------------
        # A standalone (non-external-content) FTS5 table: the application
        # inserts and deletes rows in the same transaction as the source
        # events, so stale rows never survive a deletion.
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts USING fts5(
            content,
            tokenize='porter unicode61'
        )
        """,
        # -- persistent indexing jobs (crash recovery) --------------------
        """
        CREATE TABLE IF NOT EXISTS memory_index_jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_index_jobs_status ON memory_index_jobs(status)",
        "CREATE INDEX IF NOT EXISTS idx_index_jobs_source ON memory_index_jobs(source_id)",
        # register the default text embedding model family
        """
        INSERT INTO embedding_models (model_id, kind, dimension, notes, registered_at)
        VALUES ('hash-embed-v1', 'text', 256,
                'Deterministic local hashing embedder (no model download)',
                datetime('now'))
        ON CONFLICT(model_id) DO NOTHING
        """,
    ]
    for statement in statements:
        store.execute(statement)


MIGRATIONS: list[tuple[int, str, Callable]] = [
    (1, "initial_schema", _apply_initial),
    (2, "register_default_models", _apply_register_default_models),
    (3, "episodic_memory", _apply_episodic_memory),
]


def current_version(store) -> int:
    try:
        row = store.execute_fetchone("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def run_migrations(store) -> int:
    """Apply pending migrations; return the resulting schema version."""
    target = max(v for v, _, _ in MIGRATIONS)
    applied = current_version(store)
    if applied > target:
        raise MemoryMigrationError(
            f"database schema version {applied} is newer than this build supports ({target})"
        )
    for version, name, fn in MIGRATIONS:
        if version <= applied:
            continue
        try:
            with store.transaction():
                fn(store)
                store.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, datetime.now(timezone.utc).isoformat()),
                )
        except MemoryMigrationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MemoryMigrationError(f"migration {version} ({name}) failed: {exc}") from exc
    return current_version(store)
