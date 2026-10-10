# Moondream Live VQA

Real-time **Visual Question Answering** on a live camera feed, powered by a
locally installed **Moondream 2** vision-language model. No cloud APIs, no
token, no model download — the application reuses the checkpoint that already
exists on this machine.

Point the camera at something, type a question in plain English, and get an
answer grounded in the exact frame that was captured when you pressed Enter,
while the preview keeps running.

```
Camera ──▶ capture thread ──▶ latest-frame buffer ──┬──▶ live preview (GUI)
                                                    └──▶ snapshot ──▶ Moondream 2 ──▶ answer
```

---

## Features

- **Live preview** at the camera's native rate (measured 28–30 FPS at 640×480).
- **Threaded capture** with a single-slot, lock-protected latest-frame buffer —
  old frames are discarded, never queued.
- **Background inference worker** with a *bounded* request queue (never
  unbounded); the GUI stays responsive while the GPU works.
- **Two question modes**
  - **Live snapshot** — every question captures a brand-new frame.
  - **Frozen frame** — freeze one frame and ask many questions about it; the
    vision encoding is computed once and reused (~4× faster follow-ups).
- **Answers are tied to the frame they came from**: every reply shows
  `frame #id · latency · device`, so a single still image is never mistaken
  for continuous video understanding.
- **CPU and GPU** inference (CUDA bf16 preferred, CPU fp32/bf16 fallback).
- **Offline by default** — nothing is downloaded at runtime.
- **Error handling** surfaced in the GUI: no camera, busy/disconnected device,
  model not found, CUDA OOM, invalid questions, failures during shutdown.
- **Dark, professional PySide6 UI** — usable without touching a terminal after
  launch.

---

## Quick start

```bash
cd "moondream-live-vqa"

# 1) verify the environment (model discovery, CUDA, cameras, Qt)
python3 scripts/inspect_environment.py

# 2) verify the model answers a question about a static image
python3 scripts/test_model.py

# 3) verify the camera (frames, FPS, restart, device release)
python3 scripts/test_camera.py --save-snapshot

# 4) run the desktop application
python3 main.py
```

Terminal front-end (same pipeline, no Qt):

```bash
python3 main.py --cli                      # interactive: freeze | live | status | quit
python3 main.py --cli --ask "What do you see?"
```

---

## Detected model and runtime configuration

| Item | Value on this machine |
| --- | --- |
| Model location | `../models/moondream2` (auto-detected, `MODEL_PATH` overrides) |
| Checkpoint | `model.safetensors` — 3.86 GB, bf16, **1.93 B parameters** |
| Architecture | `HfMoondream` (transformers remote code, `moondream2` 2025-06-21 release) |
| Inference API | `model.encode_image(PIL)` → `model.query(image=…, question=…)` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), CUDA 13.0, driver 580.178.04 |
| PyTorch | 2.13.0+cu130, bf16 supported |
| transformers | 5.14.1 |
| Camera | Chicony Integrated Camera (`/dev/video0`), MJPG 640×480 |
| GUI | PySide6 6.11.1, X11 (`DISPLAY=:1`) |
| Python | 3.12.3, system interpreter (no venv) |

Full report: [`docs/environment_report.txt`](docs/environment_report.txt).

The model is loaded **once** at startup (~4–5 s) and kept in memory; it is
never reloaded per question.

---

## Usage

| Action | How |
| --- | --- |
| Start/stop the camera | `Start Camera` / `Stop` buttons |
| Switch cameras | dropdown + `Refresh` |
| Ask a question | type in the input field, press **Enter** or `Ask AI` |
| Freeze the current frame | `Freeze Frame` (badge turns amber `FROZEN`) |
| Resume live snapshots | `Resume Live` (badge turns green `LIVE`) |
| Read an answer | conversation pane — each entry shows frame id, latency and device |

**Modes in one sentence:** in *live* mode each question analyses the newest
frame; in *frozen* mode every question analyses the same frozen image (and
reuses its vision encoding).

---

## Configuration

Copy `.env.example` to `.env` and edit, or use CLI flags:

```bash
python3 main.py --camera 0 --width 1280 --height 720
python3 main.py --device cpu --dtype fp32
python3 main.py --model-path /path/to/models/moondream2
python3 main.py --no-grounding            # drop the "answer only from evidence" suffix
python3 main.py --max-side 640            # optional downscale before encoding
python3 main.py --queue-size 0            # reject questions while one is running
python3 main.py --log-level DEBUG
```

All environment variables are documented in [`.env.example`](.env.example).

---

## Project layout

```text
moondream-live-vqa/
├── main.py                     # entry point (GUI by default, --cli optional)
├── config/settings.py          # defaults, .env loading, model discovery
├── src/
│   ├── camera/                 # capture thread + latest-frame buffer
│   │   ├── camera.py           #   discovery, V4L2 negotiation, reconnect
│   │   └── frame_buffer.py     #   thread-safe single-slot snapshot store
│   ├── vlm/
│   │   ├── moondream_engine.py #   load once, API detection, cache, timings
│   │   └── preprocessing.py    #   BGR→RGB, validation, optional resize
│   ├── pipeline/
│   │   ├── inference_worker.py #   bounded queue + single GPU thread
│   │   └── vqa_pipeline.py     #   orchestration, live/frozen modes, events
│   ├── gui/                    # PySide6 widgets (camera, chat, main window)
│   └── utils/                  # logging, latency measurement, GPU stats
├── scripts/                    # inspect / test_camera / test_model / benchmark / gui_selftest
├── tests/                      # pytest suite (88 tests, hardware-aware skips)
└── docs/                       # architecture, installation, troubleshooting, performance
```

Detailed design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Testing

```bash
python3 -m pytest tests/ -q          # 88 tests, ~16 s (includes a real model run)
python3 -m pytest tests/ -q -k "not RealModel"   # skip the GPU model tests
```

Hardware-dependent tests skip themselves automatically when no camera or no
checkpoint is available, so the suite is safe to run anywhere.

Reliability checks performed on this machine (see
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) and
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)):

| # | Scenario | Result |
| --- | --- | --- |
| 1 | Camera startup / restart / release | PASS (28.6 FPS, device released) |
| 2 | Multiple consecutive questions | PASS |
| 3 | Camera disconnect (forced read failures) | PASS — status `disconnected`, auto-reconnect recovers |
| 4 | Blank / invalid frames | PASS — rejected and counted, never crashes |
| 5 | Model initialisation failure | PASS — surfaced as `model_status: error`, questions refused |
| 6 | CPU fallback | PASS — fp32 CPU answer correct (14.1 s vs 0.66 s GPU) |
| 7 | GPU inference | PASS — bf16 on RTX 4060 |
| 8 | Shutdown with an in-flight request | PASS — request finishes, worker stops, camera released |
| 9 | Rapid repeated submission | PASS — queue bounded at 1 pending job, extras rejected |
| 10 | Frozen ↔ live switching | PASS — frame ids and cache keys verified |

---

## Measured performance (RTX 4060 Laptop 8 GB, 2026-10-10)

| Stage | Native 1455×1110 | Camera frame 640×480 |
| --- | --- | --- |
| Preprocess (BGR→PIL) | ~2 ms | <1 ms |
| Vision encoding (cold) | ~471 ms | ~409 ms |
| Answer generation | ~182 ms | ~186 ms |
| **End-to-end (cold)** | **~656 ms** | **~595 ms** |
| **End-to-end (frozen frame, encoding reused)** | **~264 ms** | **~268 ms** |
| Model load (once at startup) | ~4.0 s | — |
| Camera capture | — | 28.6 FPS, snapshot copy 0.07 ms |
| CUDA memory | 4.2 GB allocated / 7.8 GB total | |
| CPU fallback (fp32) | 14.1 s end-to-end | |

Numbers come from `scripts/benchmark.py --runs 5 --save` (raw JSON in
`outputs/`). Resizing to 640 px gave only ~10 % and risks losing small
detail, so resizing is **off by default**.

---

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for: no camera found,
`Device or resource busy`, permission denied, model not found, tokenizer /
offline mode, CUDA out of memory, low FPS, and Qt platform plugin issues.

Common one-liners:

```bash
ls /dev/video*                                   # is a camera node present?
fuser -v /dev/video0                             # is another app holding it?
python3 scripts/test_model.py                    # does the model answer at all?
python3 scripts/test_camera.py                   # does the camera stream?
VQA_OFFLINE=0 python3 main.py                    # allow the tokenizer cache to be fetched once
```

---

## Known limitations

- Answers describe **one captured frame**, not a video; there is no temporal
  reasoning across frames and no depth/distance measurement.
- Hallucination risk remains for small, blurry, occluded or badly lit objects,
  and for questions the image cannot answer — the grounding prompt asks the
  model to say *"unclear"* instead of guessing, but it does not eliminate
  guessing. No confidence score is shown because Moondream exposes none.
- Camera FPS depends on lighting (auto-exposure); the integrated webcam fell
  back to ~5 FPS in `YUYV`, hence the `MJPG` default.
- One inference at a time by design (GPU serialisation); a second question may
  queue behind it but never more than `VQA_QUEUE_SIZE`.
- Closing the window waits up to `VQA_SHUTDOWN_TIMEOUT` (default 10 s) for an
  in-flight request; the answer is discarded at that point.
- `pyvips` is not installed; the model's large-image tiling falls back to PIL
  (measured correct behaviour at 1455×1110).

---

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | modules, threads, data flow, concurrency rules |
| [`docs/INSTALLATION.md`](docs/INSTALLATION.md) | prerequisites, existing model setup, verification |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | failure modes and fixes |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | benchmark methodology and results |
| [`docs/environment_report.txt`](docs/environment_report.txt) | raw environment inspection output |
