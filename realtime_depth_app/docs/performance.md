# Performance

All numbers in this document were **measured on this machine** during development
(2026-10-10). They are not targets or claims about other hardware.

Machine: Ryzen 7 8845HS class laptop, NVIDIA RTX 4060 Laptop GPU (8 GB), torch 2.13.0+cu130,
transformers 5.14.1, OpenCV 5.0.0, Python 3.12.3.

Raw report: [`../benchmarks/results_20261010_140616.md`](../benchmarks/results_20261010_140616.md)

## Reproduce

```bash
python3 main.py --benchmark                      # resolution x precision table
python3 main.py --headless 90                    # end-to-end, writes outputs/headless_metrics.json
python3 main.py --headless 90 --precision autocast_bf16
```

## Model-only inference

40 timed iterations after 5 warmup, frame 1455x1110, CUDA:

| input | precision | mean ms | p95 ms | fps | VRAM MB |
|---|---|---|---|---|---|
| 252 | fp32 | 21.76 | 24.78 | 46.0 | 136 |
| 252 | autocast_bf16 | 21.43 | 23.81 | 46.7 | 144 |
| 378 | fp32 | 30.86 | 33.89 | 32.4 | 178 |
| 378 | autocast_bf16 | 24.11 | 26.19 | 41.5 | 191 |
| 518 | fp32 | 53.72 | 57.42 | 18.6 | 242 |
| 518 | autocast_bf16 | 29.79 | 32.76 | 33.6 | 266 |

Observations:

* at 518 fp32 the model is the bottleneck (18.6 fps) - below real-time for a 30 fps camera
* `autocast_bf16` buys ~1.8x at 518 with output correlation r = 0.99993 and 1.4 % mean
  relative difference vs fp32
* fp16 and bf16 are effectively identical here (within 0.5 %); bf16 is preferred for
  numerically safer accumulation
* below ~378 the GPU launch overhead dominates, so precision matters much less

## End-to-end with the live camera

`python3 main.py --headless 90 --precision autocast_bf16`, camera at 640x480:

| metric | value |
|---|---|
| camera fps | 27.9 |
| depth fps | **27.5** |
| preprocess | 7.1 ms |
| inference | 13.7 ms |
| postprocess | 0.5 ms |
| VRAM peak | 276 MB |
| dropped frames | < 1 per 90 |

Same run with the default `fp32` at 518:

| metric | value |
|---|---|
| camera fps | ~24 |
| depth fps | **14.9** |
| inference | ~37 ms |

The live numbers are better than the benchmark table because the camera feeds 640x480 (cheap
upsample/preprocess), while the benchmark upsamples to 1455x1110.

The camera itself is often the limiting factor: the integrated sensor at 640x480 measured
15 fps in low light (long exposure) to 30.1 fps when lit. Exposure is fixed at
`exposure_time_absolute=156` by the app where the driver permits; on this machine
`v4l2-ctl --set-ctrl` requires root, so the app's `capture.exposure` setting is best-effort.

## Pipeline behaviour

* `LatestSlot` drops stale frames, so display lag stays at ~1 inference period instead of
  growing
* `frame_skip: n` runs inference on every (n+1)th frame - halves GPU load, halves depth fps
* measured drop rate in the 90-frame run: ~1 % of frames (only under startup jitter)

## Tuning levers (in priority order)

| lever | effect | cost |
|---|---|---|
| `model.precision: autocast_bf16` | ~1.8x faster at 518, 1.4 % output change | re-verify calibration if extreme precision needed |
| `model.input_size: 378` | 30.9 → 24.1 ms (fp32/bf16) | coarser depth (smaller object detail) |
| `model.input_size: 252` | ~21 ms regardless of precision | much coarser |
| `runtime.frame_skip: 1` | halves GPU work | depth updates at half camera rate |
| `measurement.region_size` | smaller = less postprocess per click | noisier median |
| `depth.smoothing.enabled: false` | removes median + EMA cost (~1-3 ms) | flicker |
| CPU fallback (`model.device: cpu`) | works, tested | ~4-6x slower; not real-time |

## What was *not* measured

* ONNX / TensorRT export (not implemented - the app runs PyTorch directly)
* multi-camera throughput
* sustained thermally throttled runs longer than ~3 minutes (the 90-frame runs are ~4 s)
* power draw

Numbers for other GPUs cannot be derived from these measurements.
