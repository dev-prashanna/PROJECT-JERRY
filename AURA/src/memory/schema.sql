-- AURA Multimodal Memory System — initial schema (migration 001).
-- Applied idempotently by src/memory/migrations.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK (
        entity_type IN ('person', 'object', 'concept', 'location', 'project')
    ),
    status TEXT NOT NULL DEFAULT 'active' CHECK (
        status IN ('active', 'provisional', 'archived')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_seen_at TEXT
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alias_normalized ON entity_aliases(normalized_alias);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    embedding_data BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding_dtype TEXT NOT NULL DEFAULT 'float32',
    model_id TEXT NOT NULL,
    quality_score REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_face_entity ON face_embeddings(entity_id);

CREATE TABLE IF NOT EXISTS object_embeddings (
    embedding_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    embedding_data BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding_dtype TEXT NOT NULL DEFAULT 'float32',
    model_id TEXT NOT NULL,
    crop_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_object_entity ON object_embeddings(entity_id);

CREATE TABLE IF NOT EXISTS declarative_memory (
    fact_id TEXT PRIMARY KEY,
    subject_entity_id TEXT,
    predicate TEXT NOT NULL,
    object_value TEXT NOT NULL,
    value_type TEXT NOT NULL DEFAULT 'text',
    raw_transcript TEXT NOT NULL,
    source_turn_id TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'user_assertion',
    asserted_by_entity_id TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (
        status IN ('active', 'superseded', 'disputed')
    ),
    supersedes_fact_id TEXT,
    extraction_method TEXT NOT NULL DEFAULT 'deterministic',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(subject_entity_id) REFERENCES entities(entity_id) ON DELETE SET NULL,
    FOREIGN KEY(asserted_by_entity_id) REFERENCES entities(entity_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_subject ON declarative_memory(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_fact_status ON declarative_memory(status);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    frame_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('face', 'object')),
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    quality_score REAL,
    source_model TEXT NOT NULL DEFAULT '',
    candidate_label TEXT,
    captured_at TEXT NOT NULL,
    request_id TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS entity_observations (
    entity_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'enrollment',
    created_at TEXT NOT NULL,
    PRIMARY KEY (entity_id, observation_id),
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY(observation_id) REFERENCES observations(observation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entity_associations (
    association_id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'related_to',
    weight REAL NOT NULL DEFAULT 1.0,
    source_transcript TEXT NOT NULL DEFAULT '',
    source_turn_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(source_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY(target_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_assoc_source ON entity_associations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_assoc_target ON entity_associations(target_entity_id);

CREATE TABLE IF NOT EXISTS enrollment_consent (
    consent_id TEXT PRIMARY KEY,
    subject_description TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'face_enrollment',
    granted INTEGER NOT NULL DEFAULT 0,
    granted_at TEXT,
    expires_at TEXT,
    method TEXT NOT NULL DEFAULT 'explicit_utterance',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS identity_bindings (
    binding_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    consent_id TEXT,
    method TEXT NOT NULL DEFAULT 'explicit_enrollment',
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY(consent_id) REFERENCES enrollment_consent(consent_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS memory_revisions (
    revision_id TEXT PRIMARY KEY,
    entity_id TEXT,
    fact_id TEXT,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_models (
    model_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    registered_at TEXT NOT NULL
);
