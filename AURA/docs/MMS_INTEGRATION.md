# MMS ↔ AURA integration

## Turn flow

```
handle_text
  └─ DeterministicRouter / LLM router
       └─ memory intents (see below)
            ├─ write tools  → engine op → record.response = result text
            ├─ query tools  → engine op → record.metadata["memory_block"]
            │                              └─ prompt_builder injects it
            ├─ recall tools → engine op → record.metadata["episodic_context"]
            │                              └─ prompt_builder injects it
            └─ camera tools → CAPTURING → snapshot → engine op (no VLM)
  └─ _finish_turn
       ├─ ConversationMemory.record (short-term chat window)
       ├─ engine.record_turn (durable PECM event log — before any indexing)
       └─ engine.maybe_learn_facts(transcript) in background (non-memory turns)
```

## Intents and tools

| Intent | Tool | Camera? | Engine call |
|---|---|---|---|
| `remember_fact` | `remember_fact` | no | `remember_fact` |
| `correct_memory` | `correct_memory` | no | `correct_fact` (supersedes the prior fact) |
| `query_memory` | `query_memory` | no | `retrieve` → `memory_block` |
| `recognize_person` | `recognize_person` | yes | `recognize_faces` |
| `learn_identity` | `remember_identity` | yes | `enroll_person` (consent-gated) |
| `learn_visual_object` | `learn_visual_object` | yes | `enroll_object` |
| `forget_memory` | `forget_memory` | no | `delete_memory` |
| `list_memories` | `list_memories` | no | `list_memories` |
| `recall_memory` | `recall_memory` | **no** | `search_conversations` (hybrid FTS+semantic) |
| `recall_visual_memory` | `recall_visual_memory` | **no** | `recall_last_visual_observation` (event log) |

Router patterns live in `src/intelligence/intent_classifier.py`
(`_route_memory`); tools and the `ToolContext.memory_tool()` dispatcher live in
`src/orchestrator/tools.py`. Camera-capturing memory turns go through
`AIAgent._run_memory_vision_tools` — a snapshot only, never the VLM. Historical
recall never requests a frame.

## Durable event recording (PECM)

`AIAgent._finish_turn` calls `engine.record_turn(...)` which commits the user
transcript, VLM observation and assistant response to `conversation_events`
**before** any embedding or episode job runs. The FTS column is written in the
same transaction (`indexed_immediately`), so Level-1 recall (recent context +
keyword) works even if the background indexing worker never drains.

Derived work is queued as `memory_index_jobs` rows:
* `embed_event` → hash/ST text embedding into `text_memory_embeddings`
* `build_episode` → deterministic episode summary into `episodic_memory`

Recall tools attach `record.metadata["episodic_context"]` (a formatted RAG
block); `build_response_messages` injects it next to `memory_block`. Offline
turns speak the raw event content directly (`response_strategy =
memory_offline`).

## Prompt evidence

`engine.retrieve()` returns a `MemoryContext`; `context_builder.build_memory_block`
renders it as an untrusted-data block (header, provenance per fact, footer,
prompt-injection warning). The block is attached to `record.metadata["memory_block"]`
and injected by `build_response_messages` (`prompt_builder.py`). Offline turns
answer directly from `direct_answer()` without a cloud LLM.

`engine.build_rag_context(query)` formats retrieved episodes + events into the
episodic RAG block (budget-capped by `memory_max_context_tokens`).

## Configuration (`config/settings.py` → `MemorySettings`)

| Env var | Default | Effect |
|---|---|---|
| `MEMORY_ENABLED` | 1 | Master switch; engine is `None` when off |
| `MEMORY_DB_PATH` | `~/.aura/memory/aura_memory.db` | SQLite location |
| `MEMORY_ASSET_DIR` | `~/.aura/memory/assets` | Cropped object ROIs |
| `MEMORY_FACE_RECOGNITION_ENABLED` | 0 | Face backend construction |
| `MEMORY_FACE_BACKEND` | opencv | `opencv` or `face_recognition` |
| `MEMORY_OBJECT_GROUNDER` | auto | `moondream-box` or `center-prior` |
| `MEMORY_OBJECT_EMBEDDER` | auto | CLIP if cached, else `sift-color-v1` |
| `MEMORY_CLIP_ALLOW_DOWNLOAD` | 0 | Never silently download CLIP weights |
| `MEMORY_BACKGROUND_INDEXING` | 1 | Priority-queue background jobs |
| `MEMORY_AUTO_LEARN_FACTS` | 1 | Extract facts from ordinary turns |
| `CONVERSATION_MEMORY_ENABLED` | 1 | Durable PECM event log |
| `EPISODIC_MEMORY_ENABLED` | 1 | Episode summarisation |
| `MEMORY_FTS_ENABLED` | 1 | FTS5 keyword index |
| `MEMORY_SEMANTIC_SEARCH_ENABLED` | 1 | Text embeddings + cosine search |
| `MEMORY_TEXT_EMBEDDER` | auto | `hash` (no download) or `sentence-transformers` |
| `MEMORY_TEXT_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Preferred ST model (used when cached) |
| `MEMORY_RECENT_TURNS` | 12 | Level-1 recent-context window |
| `MEMORY_MAX_RETRIEVAL_RESULTS` | 10 | Max fused events per query |
| `MEMORY_MAX_CONTEXT_TOKENS` | 800 | RAG context block budget |
| `MEMORY_ALLOW_HISTORICAL_VISION_RECALL` | 1 | Answer visual questions from history |
| `MEMORY_RRF_K` | 60 | RRF damping constant |
| `MEMORY_INDEX_WORKER_BATCH` | 16 | Index jobs processed per worker tick |

## GUI

`src/gui/memory_panel.py` — status line, entity list, fact list, **recent
conversation** list (durable event log), and a confirm-before-delete
**Forget selected** action (person deletions warn about face embeddings and
run off the GUI thread). Wired into the right column of `MainWindow`;
refreshes every 2 s from `engine.list_memories()` + `store.recent_events()`.

## Testing

`tests/memory/` — 140+ hardware-free tests (fake face backend, tmp SQLite,
`center-prior` grounder, PECM acceptance + agent integration). Run with:

```bash
python3 -m pytest tests/memory -q
```

PECM-specific suites:
* `tests/memory/test_pecm.py` — curtain scenario Steps 1-9, temporal resolver,
  RRF fusion, recall statuses, deletion, routing-without-camera.
* `tests/memory/test_pecm_agent.py` — tool registry, offline/online recall,
  durable event recording via `handle_text`, prompt injection.

## Benchmarks

```bash
python3 scripts/benchmark_memory.py    # MMS stages (facts/faces/objects)
python3 scripts/benchmark_pecm.py      # PECM stages at 100 / 1k / 10k / 100k
```

PECM reference numbers (hash embedder, SQLite WAL, `fts_rowid` index,
p50 latency): `record_turn` ~1 ms, `search_fts` ~2 ms @ 1 k turns / ~17 ms @
10 k turns, `recall_visual` ~4 ms @ 1 k / ~37 ms @ 10 k, `recent_events`
~0.1 ms at all scales.
