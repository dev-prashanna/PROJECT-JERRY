# AURA — Multimodal Voice Assistant for Robotics

AURA is a hybrid voice assistant that listens (Whisper STT), reasons (Groq LLM
router + local fallbacks), sees (Moondream-2 VLM), and speaks (Kokoro-82M TTS)
— orchestrated by a bounded, observable state machine and presented through a
PySide6 dashboard.

```
mic ─► Whisper ─► Intent router ─┬─ general ─► Groq LLM ─► response ─► Kokoro ─► speakers
 (STT)         (Groq or local)   │
                                 └─ visual ─► camera snapshot ─► Moondream ─► grounded response
```

## Quick start

```bash
cd AURA
pip install --user --break-system-packages -r requirements.txt   # see docs/SETUP.md
cp .env.example .env                                              # add GROQ_API_KEY
python3 main.py --demo --no-speak          # scripted acceptance run
python3 main.py --text "What do you see?"  # one text turn
python3 main.py --cli                      # interactive terminal
python3 main.py --gui                      # PySide6 dashboard
```

## What each mode does

| Command | Behaviour |
|---|---|
| `main.py --text "…"` | one text command, prints the response, exits |
| `main.py --cli` | interactive REPL (text in, spoken replies out) |
| `main.py --gui` | desktop dashboard: chat, live camera, system monitor |
| `main.py --demo` | runs the three acceptance interactions + offline probe |
| `--no-speak` / `--no-listen` | disable TTS / microphone (useful headless) |
| `--offline` | force the local deterministic router, no cloud calls |
| `--router {llm,deterministic,auto,compare}` | override routing mode |
| `--strategy {auto,full,direct}` | response-composition strategy |

## Architecture at a glance

* `config/` — typed settings (defaults → `.env` → environment → overrides),
  shared model registry. Secrets are never logged.
* `src/speech/` — audio capture, VAD segmentation, Whisper STT, Kokoro TTS,
  non-blocking playback.
* `src/perception/` — camera manager (OpenCV, blank-frame detection), frame
  buffer with id-tagged snapshots, image preprocessing, Moondream-2 engine.
* `src/intelligence/` — Groq client (with fallbacks, timeouts, budget),
  deterministic + LLM intent router, prompt builder, response generator with
  hallucination auditing, conversation memory.
* `src/orchestrator/` — the `AIAgent` state machine, tool registry, bounded
  job scheduler, pydantic schemas, `build_agent()` bootstrap.
* `src/gui/` — PySide6 dashboard (thread-safe via a signal bridge).
* `scripts/` — environment inspection, per-model verification, benchmarks.
* `tests/` — 170 tests including a fully faked end-to-end acceptance suite.

Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Design principles

1. **The cloud proposes, the orchestrator disposes.** The LLM only emits a
   `RoutePlan`; the agent validates it against an allow-list before running any
   tool.
2. **Every answer is traceable.** Visual answers carry a `frame_id`, timestamp
   and inference timing; routing decisions record intent, tool and confidence.
3. **Bounded everything.** Queues, timeouts, token budgets, retries and memory
   are capped; the pipeline degrades gracefully (offline router, direct
   responses) instead of hanging.
4. **One wiring path.** `build_agent()` is the only way components are
   assembled — GUI, CLI, demo and tests all use it.

## Hardware notes (this machine)

Measured on an RTX 4060 Laptop (8 GB VRAM), Python 3.12, CUDA 13 torch with
the CUDA-12 CT2 preload fix (`src/utils/device_manager.py`):

| Stage | Typical latency |
|---|---|
| Whisper small.en (GPU) | ~100–500 ms / utterance |
| Groq `openai/gpt-oss-120b` routing | ~640–730 ms |
| Moondream-2 warm inference | ~530–880 ms (≈4.2 GB VRAM) |
| Kokoro-82M synthesis | RTF ≈ 0.01–0.31 (GPU) |
| Full text turn (general) | ~0.8–1.6 s |
| Full visual turn | ~1.5–5.6 s (incl. VLM) |

Reproduce with `python3 scripts/benchmark_pipeline.py`. More in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Known hardware caveat

On this laptop the webcam's **privacy shutter is engaged**
(`/dev/video0` reports `privacy=1`, read-only via `v4l2-ctl`), so captured
frames are uniformly black. AURA detects this and surfaces a warning in the
GUI and logs. Toggle the physical switch to enable live visual demos; the
benchmark script falls back to a synthetic test card automatically.

## Tests

```bash
python3 -m pytest tests/ -q          # 170 tests, ~7 s, no GPU/network needed
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for camera, CUDA,
audio-device and Groq-key issues.
