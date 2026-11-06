# Gemini Flash memory

AURA can use Gemini Flash for text context selection and durable fact extraction,
while keeping its existing provider for conversational replies.

## Enable

Set these values in AURA's local `.env`, then restart AURA:

```dotenv
GEMINI_API_KEY=your_key_here
MEMORY_GEMINI_ENABLED=1
MEMORY_GEMINI_MODEL=gemini-3.8-flash
MEMORY_GEMINI_TIMEOUT=8
MEMORY_ENABLED=1
MEMORY_AUTO_LEARN_FACTS=1
MEMORY_LLM_FACT_EXTRACTION=1
```

The model is configurable; use a Gemini Flash model available to your API project.
No new Python dependency is required. Without a key, existing local retrieval and
configured fact extraction continue. Explicit orchestrator offline mode disables
cloud memory calls. `MEMORY_GEMINI_ENABLED=0` disables this integration.

## Behavior

Before a general response, Gemini selects relevant saved facts and older
conversation exchanges. It receives the current request, recent conversation text,
and a bounded candidate pool: keyword matches plus recently updated active facts
(up to 80 total). Selected IDs must exist in that pool. The newest two user turns
and their replies remain intact, regardless of the selection. The current request
is always sent to the reply generator. Existing memory tools retain their direct
retrieval paths. This is bounded retrieval, not a search of every stored fact.

After a successful conversation turn, Gemini extracts explicitly stated durable
facts, such as preferences, relationships and project details. Each accepted fact
must include a verbatim evidence substring from the user's transcript and a durable
flag. Temporary states, guesses and quoted assertions are excluded by the extraction
prompt. These checks reduce errors but do not guarantee semantic correctness.
Existing correction and single-value supersession rules apply.

Facts and their source turn/transcript are stored in the existing SQLite database
(`MEMORY_DB_PATH`, default `~/.aura/memory/aura_memory.db`). Facts have no automatic
expiration; active facts remain available across restarts until corrected or deleted.
Old conversation text is recorded separately rather than repeatedly stuffed into
all replies. The in-process recent conversation window itself resets on restart.

With background indexing and conversation recording enabled, automatic fact
extraction is queued in SQLite. Pending work resumes after restart. When background
indexing is disabled, extraction runs synchronously. Gemini failure uses deterministic
extraction; facts beyond those local patterns may not be extracted, and failed API
calls are not automatically replayed. Context-selection failure retains local
retrieval and the complete configured recent history. Calls have a configurable
socket timeout (1–30 seconds); selection adds a cloud round trip before replies.

Only text is sent through this integration, including saved facts and conversation
content; face images and embedding vectors are not sent. Keys are sent in an HTTP
header, excluded from settings repr/redacted diagnostics, and should stay in the
ignored local `.env` file.

## Verification

Run `python3 -m pytest tests/memory/test_gemini_memory.py -q` for mocked REST,
secret redaction, extraction validation, safe fallback, context selection, offline
mode, pending-job recovery and fact persistence tests. Live API verification requires
a valid key and a restart after changing the configuration.

Native API reference: https://ai.google.dev/api/generate-content
Model availability: https://ai.google.dev/gemini-api/docs/models
