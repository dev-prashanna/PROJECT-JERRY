# Performance

All numbers below were measured on the development machine on **2026-10-10**
with `scripts/benchmark.py` and the reliability scripts; raw JSON is kept in
`outputs/benchmark_*.json`. Nothing here is a claim about hardware other than
the machine described in `docs/environment_report.txt`.

## Measurement environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB (7.80 GB visible to CUDA) |
| Driver / CUDA | 580.178.04 / CUDA 13.0 runtime (torch 2.13.0+cu130) |
| CPU / RAM | 16 threads, 14 GB RAM |
| Model | Moondream 2 (2025-06-21), `HfMoondream`, **1.93 B params**, bf16 |
| Camera | Chicony UVC, MJPG 640×480 |
| Software | Python 3.12.3, transformers 5.14.1, opencv 5.0.0, PySide6 6.11.1 |

**Method.** `torch.cuda.synchronize()` brackets every timed stage, so the
numbers are real GPU time, not kernel-submission time. Each figure is the mean
of 5 runs (cold cache) after a warm-up question. Raw artefacts:
`outputs/benchmark_20261010_162826.json`.

Reproduce with:

```bash
python3 scripts/benchmark.py --runs 5 --save
python3 scripts/test_model.py            # single-shot latency check
```

## 1. Model initialisation

| Phase | Time |
| --- | --- |
| Load weights from disk to GPU (3.86 GB safetensors) | **3.97 s** (4.0–5.0 s typical) |
| First answer (includes CUDA warm-up) | 0.92 s |
| Steady-state answer (camera frame) | 0.55–0.63 s |

The model is loaded **once** at startup; subsequent questions never reload it.

## 2. Per-question latency (GPU, bf16)

| Stage | 1455×1110 (sample photo) | 1280×720 | **640×480 (camera)** | 640 px downscale |
| --- | --- | --- | --- | --- |
| Preprocess (BGR→PIL) | 1.8 ms | 0.5 ms | **0.3 ms** | 0.2 ms |
| Vision encoding (cold) | 471 ms | 419 ms | **409 ms** | 410 ms |
| Answer generation | 182 ms | 189 ms | **186 ms** | 180 ms |
| **End-to-end (cold)** | 656 ms | 608 ms | **595 ms** | 590 ms |
| **End-to-end (frozen frame)** | 264 ms | 270 ms | **268 ms** | 262 ms |

Reading: encoding dominates; generation is ~30 % of a cold question. Asking a
second question about a **frozen frame** skips encoding entirely and lands
around **260–270 ms** end-to-end (the first question on a frozen frame is the
cold number).

### Frozen-frame encoding cache

| Scenario | Encode time | Total |
| --- | --- | --- |
| New frame (cold) | 409 ms | 595 ms |
| Same frame, same `cache_key` | **0 ms** (`encoding_reused=True`) | ~50–90 ms |

Verified automatically by `tests/test_vlm.py::TestRealModelIntegration::
test_encoding_cache_reused_for_same_key`.

## 3. Image resizing — measured, deliberately disabled

| Input | End-to-end | Note |
| --- | --- | --- |
| Native 1455×1110 | 656 ms | full detail |
| Forced 640×480 | 595 ms | **~10 % faster**, small objects/text degraded |

A 10 % gain is not worth losing small-object and sign-text detail for a VQA
system, so `VQA_MAX_SIDE=0` (no resize) is the default. Enable it only if you
have measured a need: `python3 main.py --max-side 640`.

## 4. CPU fallback

| Device | Dtype | End-to-end | Encode | Generate |
| --- | --- | --- | --- | --- |
| RTX 4060 (CUDA) | bf16 | **0.66 s** | 0.47 s | 0.18 s |
| CPU (16 threads) | fp32 | **14.05 s** | 9.50 s | 4.54 s |

≈ **21× slower** on CPU — usable for occasional questions, not for a snappy
session. CPU load needs ~8 GB free RAM for fp32; the loader automatically
retries with `bfloat16` if that allocation fails.

## 5. Camera pipeline

| Metric | Value |
| --- | --- |
| Negotiated mode | MJPG 640×480 (V4L2) |
| Measured capture rate | **28.6 FPS** (idle) |
| Snapshot copy (`get_latest_frame`) | **0.07 ms** mean (0.06–0.11 ms) |
| Preview repaint budget | 33 ms timer (≤30 Hz) |
| Memory held by the buffer | exactly **one** frame (~0.9 MB) |

`YUYV` on the same device measured **4.8 FPS** — the reason `MJPG` is the
default.

### Responsiveness while the GPU is busy

While 13 questions were answered back-to-back (GPU saturated), the camera
still delivered frames at **mean 28.1 FPS (min 25.4, max 30.0)** — the preview
never stalls, because capture and inference run on separate threads and share
only the single-slot buffer.

## 6. GPU memory

| Metric | Value |
| --- | --- |
| Weights (bf16) | ~3.85 GB |
| Steady-state allocated | **4.22 GB** |
| Reserved by the allocator | 4.48 GB |
| Total visible to CUDA | 7.81 GB |

Fits an 8 GB card with ~3.2 GB headroom. Memory does **not** grow with the
number of questions: only one encoded image is cached (and it is replaced, not
accumulated), and frame buffers are single-slot.

## 7. Throughput / queue behaviour

| Property | Value |
| --- | --- |
| Concurrent GPU requests | 1 (serialised by design) |
| Pending capacity | 1 (`VQA_QUEUE_SIZE`), further submits rejected |
| Sustained question rate | ~1.7 questions/s (cold frames) |
| Frozen-frame follow-up rate | ~3–4 questions/s |

## 8. Optimisations considered

| Idea | Status | Rationale |
| --- | --- | --- |
| Load model once, keep in `eval()` + `inference_mode()` | **done** | removes 4 s reload per question |
| Cache encoded image per frozen frame | **done** | 595 ms → ~270 ms for follow-ups |
| Single-slot frame buffer | **done** | no stale frames, no memory growth |
| Bounded inference queue | **done** | GPU cannot be flooded |
| MJPG instead of YUYV | **done** | 4.8 → 30 FPS capture |
| CUDA-synchronised timers | **done** | honest latency numbers |
| Downscale before encode | **off by default** | only 10 % gain, loses detail (measured) |
| Model quantisation (int8/4-bit) | **not attempted** | not supported by this checkpoint's custom code without further work; would need accuracy re-validation |
| `torch.compile()` | **not attempted** | the model ships its own fused path; compile gains were not validated here, so no claim is made |
| Streaming tokens into the GUI | **possible future work** | `model.query(..., stream=True)` exists in the installed build |

## 9. Known performance limitations

- Encoding is the bottleneck (~70 % of a cold question); questions about an
  unchanged scene should use **frozen mode** to skip it.
- Latency varies with answer length: short object lists generate in ~140 ms,
  full sentences ~200 ms.
- First question after startup pays a CUDA warm-up (0.92 s vs 0.63 s).
- On CPU the experience is qualitatively different (seconds per answer); the
  UI stays responsive, but live-mode questioning feels slow.
