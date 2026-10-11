# AURA Multimodal Memory System (MMS)

Persistent, consent-aware memory for facts, faces and objects. Memory lives in
a local SQLite database (`~/.aura/memory/aura_memory.db` by default) and is
retrieved on every turn to ground AURA's answers in what the user has
previously said, shown or enrolled.

```
                    ┌──────────────────────────────────────────┐
 text turn ────────►│ FactExtractor ─► EntityResolver ─► Store │
                    │                                      │   │
 camera frame ─────►│ FaceMemoryService / ObjectMemory      │   │
                    │        (embeddings ► VectorIndex)     │   │
                    │                                      ▼   │
                    │        MemoryRetriever ◄── associations  │
                    │              │                           │
                    │              ▼                           │
                    │   MemoryContext ─► prompt block /       │
                    │                    direct answer        │
                    └──────────────────────────────────────────┘
```

## What is remembered

| Modality | Written by | Stored as | Read back as |
|---|---|---|---|
| Facts | `remember_fact`, `correct_memory`, background `maybe_learn_facts` | `declarative_memory` rows (subject/predicate/object + provenance) | keyword retrieval + subject-entity attachment |
| Faces | `remember_identity` (explicit self-enrolment only) | `face_embeddings` (128-D, YuNet + SFace or `face_recognition`) | nearest-neighbour match with ambiguity margin |
| Objects | `learn_visual_object` | `object_embeddings` (CLIP if cached, else `sift-color-v1`) + cropped ROI asset | similarity vs novelty threshold |

Everything is linked through `entities` + `entity_associations`, so asking
"what do you know about my project?" can surface a fact *and* an enrolled
object that was linked to it.

## Consent model

* **Facts** — learned from the current user's own statements. Statements that
  are hypothetical, negated, quoted or questions are never stored.
* **Faces** — self-enrolment utterances only ("remember my face as Alex").
  Third-party enrolment ("this is Alice") is refused; multi-face frames require
  an explicit selection + confirmation token. Tokens are short-lived records in
  `enrollment_consent`. Face recognition ships **disabled**
  (`MEMORY_FACE_RECOGNITION_ENABLED=0`).
* **Objects** — the object's owner is the consenting current speaker; raw
  frames are never persisted, only content-cropped ROIs.

## Key thresholds (all provisional / configurable)

| Setting | Default | Meaning |
|---|---|---|
| `memory.face_match_threshold` | 0.55 | L2 distance below which a face counts as a match |
| `memory.face_ambiguity_margin` | 0.05 | If the runner-up is within this of the best match, the answer is "ambiguous" |
| `memory.object_novelty_threshold` | 0.65 | Cosine similarity at/below this an object is "novel" |
| `memory.context_token_budget` | 600 | Token budget for the memory block injected into the prompt |
| `memory.snapshot_freshness_s` | 5.0 | Reuse a camera snapshot younger than this |

## Database

SQLite in WAL mode with foreign keys on. Schema is versioned via
`src/memory/migrations/` (v1 initial schema, v2 default embedding-model
registry, v3 PECM tables). Migrations run automatically on `MemoryStore` open;
opening a database created by a **newer** schema raises `MemoryMigrationError`
instead of corrupting it. Deletion cascades from `entities` to embeddings, facts,
observations, consents and bindings; face/object index entries are updated in
process.

## Persistent Episodic & Conversational Memory (PECM)

On top of facts/faces/objects, AURA keeps a **durable event log** of every
interaction (schema v3). The original user utterance, VLM observation and
assistant response are committed to SQLite *before* any embedding or
summarization job runs, so fresh recall works even if derived indexes never
complete.

```
turn ──► ConversationRecorder ──► conversation_events (authoritative)
                │                        │
                │ (same txn)             │ (async jobs)
                ▼                        ▼
          conversation_fts         text_memory_embeddings
          (BM25 keyword)           (hash-embed-v1 semantic)
                                          │
                                          ▼
                              EpisodeBuilder ──► episodic_memory
```

* **Two-level retrieval** — Level 1 (active-session event history + recent
  committed events + direct keyword match) answers before embeddings finish;
  Level 2 (FTS5 BM25 + cosine semantic + temporal/entity rerank) fuses via
  Reciprocal Rank Fusion for long-term queries.
* **Camera-free historical recall** — `recall_visual_memory` answers "what
  color was the curtain earlier?" from the event log without requesting a
  fresh frame.
* **Episodes** — simple VQA turns are summarised deterministically
  (`"The user asked… moondream2 observed… AURA answered…"`); the raw events
  remain authoritative.
* **Text embeddings** — default `hash-embed-v1` (256-D deterministic hashing,
  no model download, sub-ms). Sentence-transformers (`BAAI/bge-small-en-v1.5`)
  can be selected via `MEMORY_TEXT_EMBEDDER` when cached.

| Setting | Default | Meaning |
|---|---|---|
| `conversation_memory_enabled` | true | Durable event log on/off |
| `episodic_memory_enabled` | true | Episode summarisation on/off |
| `memory_fts_enabled` | true | FTS5 keyword index |
| `memory_semantic_search_enabled` | true | Text embeddings + cosine search |
| `memory_text_embedding_model` | `BAAI/bge-small-en-v1.5` | Preferred ST model (used when cached) |
| `memory_recent_turns` | 12 | Level-1 recent-context window |
| `memory_max_retrieval_results` | 10 | Max fused events per query |
| `memory_max_context_tokens` | 800 | RAG context block budget |
| `memory_allow_historical_vision_recall` | true | Answer visual questions from history |
| `memory_rrf_k` | 60 | RRF damping constant |

Performance (hash embedder, SQLite WAL, `fts_rowid` index): hybrid search
p50 stays under 5 ms at 1 k turns and under 40 ms at 10 k turns
(`scripts/benchmark_pecm.py`).

## Vector search

`ExactNumpyIndex` (`src/memory/vector_index.py`) — exact L2/cosine search over
NumPy, model-id checked, NaN/Inf rejected at serialization time. Suitable for
the thousands-of-entries scale this assistant targets; the `VectorIndex`
protocol leaves room for an ANN backend later.

## Threading

The engine facade is synchronous (AURA is thread-based, not asyncio). All
store writes are serialised by an RLock. Optional background indexing runs on
`MemoryJobScheduler` — a bounded single-worker priority queue
(`INTERACTIVE > REQUEST_VISUAL > BACKGROUND > MAINTENANCE`).

See `docs/MMS_INTEGRATION.md` for orchestrator wiring and configuration.
