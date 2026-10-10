# Architecture

## 1. Design goals

1. **Never block the GUI thread** — capture, model loading and inference all
   run outside Qt's event loop.
2. **Never block capture** — the camera thread only writes one slot; inference
   reads a copy of it.
3. **Never grow unboundedly** — one buffered frame, one active inference, one
   optional queued request.
4. **Answers must be traceable** — every result carries the `frame_id`,
   timestamp, latency and device it came from.
5. **One model instance** — loaded once, kept in memory, serialised access.

## 2. Component map

```text
┌────────────────────────────────────  GUI thread (PySide6) ───────────────────────────────────┐
│  MainWindow ──── CameraWidget (preview timer, 30 Hz) ──── ChatWidget (input, history)        │
│      │  ▲                                                                          │        │
│      │  │ Qt queued signal (PipelineBridge.event)                                   │        │
└──────┼──┼──────────────────────────────────────────────────────────────────────────┼────────┘
       │  │                                                                          │
       │  │  on_event(kind, payload)                                                 │ submit()
       ▼  │                                                                          ▼
┌───────┴─────────────────────  VQAPipeline (orchestration)  ─────────────────────────────────┐
│  • mode (live / frozen)  • frozen snapshot  • model state  • error state  • event fan-out   │
└──────┬──────────────────────────────┬──────────────────────────────┬───────────────────────┘
       │                              │                              │
       ▼                              ▼                              ▼
┌──────────────────┐        ┌──────────────────────┐      ┌─────────────────────────────────┐
│ CameraSource     │        │ MoondreamEngine      │      │ InferenceWorker                │
│ (thread)         │        │ (loaded once)        │      │ (thread + bounded Queue)       │
│  V4L2 read loop  │        │  encode / query      │      │  0..N queued, 1 active         │
│  reconnect logic │        │  encoded-frame cache │      │  serialised GPU access         │
└──────┬───────────┘        │  CUDA-synced timers  │      └───────────────┬─────────────────┘
       │ update()           └──────────────────────┘                      │ ask()
       ▼                                                                  ▼
┌──────────────────┐                                            {answer, timings, frame_id…}
│ LatestFrameBuffer│
│  single slot     │──── get_latest_frame(copy=True) ──▶ GUI preview / snapshot for a question
└──────────────────┘
```

## 3. Threads

| Thread | Owner | Work | Communication |
| --- | --- | --- | --- |
| Qt main | `QApplication` | widgets, preview repaint, user input | signals/slots only |
| `camera-<i>` | `CameraSource` | `cap.read()`, FPS estimation, reconnect | `on_event(CameraEvent)` → bridge |
| `moondream-loader` | `VQAPipeline.load_model_async` | one-time model load | `model_status` event |
| `vqa-inference` | `InferenceWorker` | preprocessing, encoding, generation | `inference_state` / `inference_result` events |
| `camera-discovery` | `MainWindow.refresh_cameras` | probing device indices | `cameras_found` event |

**Rule:** no thread except the Qt main thread ever touches a widget. Worker
threads emit data-only events; `PipelineBridge` (`QObject` + `Signal(str,
object)`) converts them into *queued* connections, so the slot runs on the GUI
thread.

## 4. Data flow of one question

```text
1. user presses Enter                       (GUI thread)
2. VQAPipeline.ask(question)                (GUI thread)
     ├─ validate: non-empty, ≤4000 chars, model loaded
     ├─ LIVE   → snapshot = camera.get_latest_frame(copy=True)
     │  FROZEN → snapshot = self._frozen
     ├─ cache_key = f"frame:{snapshot.frame_id}"
     └─ InferenceJob(job_id, question, snapshot, cache_key, mode)
3. worker.submit(job)  → bounded queue      (rejects with a reason if full)
4. event "submitted"  → user bubble added, status "running"
5. worker thread:
     engine.ask(question, frame=snapshot.frame, cache_key=…)
       ├─ preprocess  BGR → RGB → PIL        (StageTimer)
       ├─ encode      cached if cache_key matches → encode_time = 0
       ├─ generate    model.query(..., settings={max_tokens, temperature})
       └─ CUDA synchronize before reading the clock
6. result dict → worker → "inference_result" event → chat entry + status bar
```

`FrameSnapshot` owns its `np.ndarray`; the camera keeps writing into a
different array, so the pixels an answer was computed from can never change
mid-flight.

## 5. Live vs frozen mode

| | Live snapshot (default) | Frozen frame |
| --- | --- | --- |
| Snapshot source | newest buffer entry at submit time | the frozen `FrameSnapshot` |
| `cache_key` | `frame:<new id>` each question | `frame:<same id>` → **encoding reused** |
| Camera | keeps streaming | keeps streaming (preview stays live) |
| UI badge | `LIVE` (green) | `FROZEN` (amber) |
| Use case | "what am I holding now?" | "what objects? what colour? any text?" |

Switching to frozen copies the frame out of the buffer, so later camera frames
cannot overwrite it. `unfreeze()` clears the engine's encoding cache.

## 6. Concurrency invariants

1. **Single-slot frame buffer.** `LatestFrameBuffer.update()` replaces the one
   stored snapshot; readers always receive a fresh copy (`copy=True`).
2. **Serialised GPU access.** `MoondreamEngine.ask()` holds `_inference_lock`
   for the whole request, and only the inference worker calls it — the model's
   KV cache is a shared, mutable resource.
3. **Bounded queue.** `InferenceWorker(queue_size=1)` accepts at most one
   pending job; `queue_size=0` rejects while busy. Overflow returns
   `(False, reason)` and emits a `rejected` event — nothing accumulates.
4. **Idempotent lifecycle.** `pipeline.stop()` may be called twice; the worker
   drops results that arrive after shutdown (`_accept_results = False`), the
   camera clears its buffer, and the engine releases its image cache.
5. **Late results are discarded, not delivered to a dead window.**

## 7. Error handling layers

| Layer | Failure | Behaviour |
| --- | --- | --- |
| `frame_buffer` | empty/invalid frame | `ValueError`, counted in `rejected` |
| `CameraSource` | cannot open | status `error`/`reconnecting`, event emitted |
| `CameraSource` | consecutive read failures | `disconnected` → auto-reconnect with backoff |
| `MoondreamEngine.load` | missing files, no CUDA, API mismatch | `ModelNotFoundError` / `ModelLoadError` / `ModelApiError` |
| `MoondreamEngine.ask` | any runtime failure incl. `OutOfMemoryError` | returns `status="error"` payload (never raises) |
| `InferenceWorker` | callback or engine crash | error payload emitted, worker survives |
| `VQAPipeline` | bad question / no frame / model not ready | `(False, reason)` before queueing |
| `MainWindow` | any handler exception | logged, GUI keeps running; top-level `sys.excepthook` shows a dialog |

Everything ends up in the bottom status bar (`Last error: …`) and in
`logs/app.log`.

## 8. Why these choices

- **PySide6** — native, responsive, first-class threads/signals; the
  environment already ships 6.11.1.
- **Qt-free core** (`camera`, `vlm`, `pipeline`) — the pipeline is testable
  headlessly with pytest and reusable from the CLI front-end.
- **Single worker instead of a thread pool** — GPU inference is serialised by
  nature here; a pool would only add VRAM pressure and races on the KV cache.
- **Copy on read** — 0.07 ms for a 640×480 frame (measured) buys complete
  isolation from the capture loop.
- **No frame queue** — a 30 FPS camera against a 0.6 s model would accumulate
  18 stale frames per answer; only the newest matters.

## 9. Extension points

- `MoondreamEngine` already normalises two Moondream APIs; a third (e.g. the
  standalone `moondream` package) only needs a new `_detect_api` branch plus a
  `_generate` implementation.
- `VQAPipeline.on_event` is a single callback: a web/REST front-end could be
  added next to the GUI and CLI without touching the core.
- `InferenceConfig.encode_max_side` enables a measured, opt-in downscale.
