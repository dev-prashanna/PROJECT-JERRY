# Performance

All numbers below were measured on the development machine:

* CPU/RAM: Linux x86_64, ~15 GB RAM
* GPU: NVIDIA RTX 4060 Laptop, 8 GB VRAM, CUDA 13 torch + CUDA-12 CT2 preload
* Python 3.12.3, models: Whisper `small.en`, Groq `openai/gpt-oss-120b`,
  Moondream-2 (HF, CUDA), Kokoro-82M (CUDA)

Reproduce any time with:

```bash
python3 scripts/benchmark_pipeline.py --runs 5
# -> reports/benchmark.json + reports/benchmark.txt
```

## Per-stage latency

| Stage | p50 | p95 | notes |
|---|---|---|---|
| Whisper small.en (GPU) | ~100 ms | ~500 ms | 4.5 s test utterance; CPU is ~1.6 s |
| Groq routing (one request) | ~640 ms | ~730 ms | `openai/gpt-oss-120b`, 64-token reply |
| Groq response generation | ~800 ms | — | full answer composition |
| Moondream-2 encode + query (warm) | ~530 ms | ~880 ms | 4.2 GB VRAM resident |
| Moondream-2 cold load | ~4–10 s | — | once per process |
| Kokoro synthesis | ~65–90 ms | — | ~5 s of audio, RTF 0.01–0.31 on GPU |

## End-to-end turns (text input, no speech)

| Turn | p50 | p95 | breakdown |
|---|---|---|---|
| General question (no camera) | ~0.8 s | ~1.6 s | route ~0.8 s + generate ~0.8 s |
| Visual question (camera + VLM) | ~1.4 s | ~5.6 s* | route ~0.8 s + snapshot ~0 ms + VLM ~0.5–0.9 s |

\* The first visual turn includes Moondream warm-up when the process just
started; subsequent warm turns land ~1.5–2.5 s.

Voice turns add: VAD segmentation + Whisper (~0.1–0.5 s) before, and Kokoro
synthesis + playback after. Time-to-first-audio is tracked as
`time_to_first_audio` in `RequestMetrics`.

## Bottlenecks and what was done about them

1. **Cloud routing dominates the general path (~0.6–0.8 s).** Mitigations:
   deterministic router for obvious intents (`ROUTER_MODE=auto`), direct
   response strategy for simple visual confirmations, and a strict
   `max_completion_tokens` budget so reasoning models don't pad replies.
2. **Whisper on CPU was 4× slower.** CTranslate2 now loads CUDA-12 libs at
   startup (`device_manager.ensure_ct2_cuda_libs`), giving ~0.3–0.5 s GPU
   transcription.
3. **Model load times on the critical path.** All three local models load in
   background warm-up threads (`AIAgent._warm_models`); the first request may
   wait, later ones don't.
4. **VLM answer quality vs latency.** `max_new_tokens` is capped; short
   object-listing questions return in ~500 ms. Longer scene descriptions cost
   proportionally more.

## Throughput / queueing

`PipelineScheduler` uses a bounded queue (size 2, drop-oldest): rapid-fire
commands collapse to the newest request rather than building latency debt.
Barge-in (`stop_speaking`) cancels the in-flight job and drains the audio
buffer immediately (half-duplex speech, full-duplex interruption).

## Memory footprint (resident)

| Component | VRAM |
|---|---|
| Moondream-2 | ~4.2 GB |
| Whisper small.en (faster-whisper/CT2) | ~0.5 GB |
| Kokoro-82M | ~0.3 GB |
| CUDA context + overhead | ~0.6 GB |
| **Total idle after warm-up** | **~5.6 GB** |

Fits an 8 GB card with headroom for the desktop compositor.

## Accuracy sanity checks

* Whisper on the Kokoro-synthesised test sentence: perfect transcript
  (`word_overlap` ≈ 1.0 in `verify_whisper.py`); silence suppressed.
* Routing on 4/4 scripted samples: correct intent + tool
  (`scripts/verify_groq.py`).
* Moondream on a synthetic test card (square/circle/triangle):
  "A square, a circle, and a triangle".
* Response audit: dark/empty frames produce honest "no objects visible"
  answers instead of hallucinated scenes (covered by
  `tests/test_end_to_end.py::TestHallucinationAudit`).
