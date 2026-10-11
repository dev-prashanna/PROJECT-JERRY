# AURA Architecture

## Pipeline overview

```
                 ┌──────────────────────────────────────────────────────┐
                 │                    AIAgent (state machine)            │
mic ─► VAD ─► Whisper STT ─► Intent Router ─► Tool execution ─► Response │ ─► Kokoro ─► speaker
                 │   │              │                │             ▲     │
                 │   │              │                ▼             │     │
                 │   │              │         camera snapshot      │     │
                 │   │              │                │             │     │
                 │   │              │                ▼             │     │
                 │   │              │          Moondream-2 VLM ────┘     │
                 │   └──────────────┴── ConversationMemory ─────────────┘│
                 └──────────────────────────────────────────────────────┘
```

## State machine

`IDLE → LISTENING → TRANSCRIBING → INTERPRETING → [CAPTURING → PERCEIVING]
→ GENERATING → SPEAKING → IDLE`, with `ERROR` reachable from any active
state and `SHUTDOWN` terminal. Transitions are validated against an allow-list
(`src/orchestrator/state_machine.py`); every transition is recorded and
emitted to listeners (GUI, logs).

## Routing

Two routers behind one interface (`src/intelligence/intent_classifier.py`):

* **Deterministic** — keyword/regex rules with conservative ambiguity
  handling. If a reference word ("that", "other", "previous") appears without
  context, it refuses to guess and asks a clarifying question.
* **LLM** — pluggable via `LLM_PROVIDER` (Groq `openai/gpt-oss-120b` by
  default; OpenAI, Anthropic, Moonshot, DeepSeek, MiMo, or any
  OpenAI-compatible server are one env var away) returns a structured JSON
  `RoutePlan`:

```json
{
  "intent": "visual_question",
  "tool": "analyze_camera",
  "requires_camera": true,
  "requires_vlm": true,
  "normalized_question": "What objects are visible in front of the camera?",
  "confidence": 0.94,
  "rationale": "user asks about the current camera view"
}
```

The plan is **validated by pydantic** (`RoutePlan.model_validate`): unknown
intents/tools are rejected, visual intents are forced to require the camera,
and malformed JSON never crashes the pipeline — the deterministic router takes
over (`decision.error` is set and logged).

Router modes (`ROUTER_MODE`):

| mode | behaviour |
|---|---|
| `llm` | cloud routing with deterministic fallback on any error |
| `deterministic` | local rules only |
| `auto` | deterministic for obvious patterns, LLM for ambiguous ones |
| `compare` | runs both, logs the delta (debugging/evaluation) |

## Tools

Only allow-listed tools run (`src/orchestrator/tools.py`):

* `analyze_camera` — snapshot → Moondream question → `VisualEvidence`
* `describe_scene` — open-ended caption of the freshest frame
* `repeat_last_response` — memory-backed replay
* `explain_limitation` — honest "I cannot do that" answers

`ToolContext.analyze_camera` reuses the snapshot taken during `CAPTURING`
(the agent stashes it in `ctx.pending_snapshot`), so the evidence always
carries the exact `frame_id`/timestamp that was perceived — no TOCTOU gap
between "what was shown" and "what was described".

## Response composition

`ResponseGenerator` implements three strategies:

* `direct` — local template from the evidence only (no cloud call, fastest)
* `full` — LLM composes a spoken-friendly answer from evidence + memory
* `auto` — `direct` when evidence is short and unambiguous, otherwise `full`

Every generated answer is **audited** (`response_audit` flag): the generator
checks the reply against the evidence text and flags (and softens) claims that
introduce objects/numbers not present in the observation — this is what keeps
the VLM honest on dark/empty camera frames.

## Multimodal memory

`src/memory/` — a persistent, consent-aware memory engine (`MultimodalMemoryEngine`)
backed by local SQLite (`~/.aura/memory/aura_memory.db`, WAL, versioned
migrations). It stores declarative facts (subject/predicate/object with
provenance), 128-D face embeddings (consent-gated self-enrolment only) and
object embeddings (CLIP when cached, deterministic sift-color fallback), all
linked through a shared entity graph. Each turn, the retriever assembles a
`MemoryContext`; write tools answer directly, query tools attach a
prompt-injection-hardened memory block to the response prompt, and ordinary
conversation turns are mined for new facts in the background. Memory intents
(`remember_fact`, `correct_memory`, `query_memory`, `learn_identity`,
`learn_visual_object`, `recognize_person`, `forget_memory`, `list_memories`)
route through the same tool registry as everything else.

On top of that, the **Persistent Episodic & Conversational Memory** layer
(schema v3) commits every interaction — user transcript, VLM observation,
assistant response — to a durable `conversation_events` table *before* any
embedding or summarization job runs. FTS5 keyword search and deterministic
hash embeddings (Reciprocal Rank Fusion) power hybrid retrieval; historical
visual questions ("what color was the curtain earlier?") are answered from the
event log without requesting a camera frame (`recall_memory`,
`recall_visual_memory` intents). Details:
[docs/MMS_OVERVIEW.md](MMS_OVERVIEW.md), [docs/MMS_INTEGRATION.md](MMS_INTEGRATION.md).

## Threading model

* One scheduler worker thread runs queued requests (`PipelineScheduler`,
  bounded queue, drop-oldest).
* The camera runs its own capture thread into a single-slot `FrameBuffer`
  (never queues; a slow consumer always sees the newest frame).
* Audio playback is a separate player thread with a small chunk queue;
  barge-in (`stop_speaking`) drains it immediately.
* The GUI touches nothing off the main thread: agent events are re-emitted
  through a `QObject` signal bridge (`src/gui/bridge.py`), camera frames are
  polled at ~15 fps via `QTimer`, and blocking calls (push-to-talk recording)
  run on `QThreadPool`.

## Error handling

* Domain errors subclass `AURAError` (`AuthenticationError`, `RateLimitError`,
  `OfflineError`, `StaleFrameError`, `InferenceError`, …).
* `handle_text` catches `AURAError` → friendly spoken message + `ERROR` state
  → back to `IDLE`; unexpected exceptions are contained the same way.
* Groq calls have explicit timeouts, a retry budget, and model fallback;
  a 401 does **not** fall through to the fallback model (key-level failure).
* Offline mode (`--offline` or missing key) forces the deterministic router
  and tags the response with a notice.

## Configuration

`config/settings.py` resolves, in order: built-in defaults → `AURA/.env` →
real environment variables → explicit overrides (`Settings.with_overrides`).
`SecretFilter` in the logger redacts anything matching `gsk_*` / key-like
patterns, so secrets never reach logs or reports.

## Extension points

* **New tool**: register it in `build_registry` and add its name to the
  router's tool vocabulary + `RoutePlan` validation.
* **New router**: implement `route(transcript, history_context, history) →
  RouteDecision` and pass it to `AIAgent(router=...)`.
* **New LLM provider**: set `LLM_PROVIDER` to any name not already built in —
  it is treated as an OpenAI-compatible server (`LLM_BASE_URL`, `LLM_API_KEY`,
  `LLM_MODEL`). For a native protocol, implement `BaseLLMClient` and
  `register_llm(name, factory)` in `src/intelligence/llm/registry.py`.
* **New VLM**: `register_vlm(name, factory)` in
  `src/perception/vlm/registry.py` and set `VLM_PROVIDER=name`.
