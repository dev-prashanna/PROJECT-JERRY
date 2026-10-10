# Architecture

## Data flow

```
camera / video / image (src/camera.py)
        │  Frame{id, image, t_capture}
        ▼
LatestSlot (single-slot, bounded: stale frames are dropped, never queued)
        │
        ▼
InferenceWorker thread (src/pipeline.py)
        │  preprocess → CUDA forward → bicubic upsample to frame size
        │  → optional temporal smoothing (EMA + temporal median)
        ▼
LatestSlot  →  DepthResult{frame_id, t_capture, depth, timings}
        │
        ▼
main loop (single UI thread)
        │  undistort (optional) → measurements → render → imshow
        ▼
window: [RGB + markers | depth + legend]
```

## Frame identity

Every `Frame` carries a monotonically increasing `id` and a `t_capture` timestamp. A
`DepthResult` carries the `frame_id` of the frame it was computed from. The UI compares them:

* `depth.frame_id == frame.id` → fresh, normal display
* different ids → the depth panel is stamped `DEPTH FROM FRAME n (now m) - x ms old`

An old prediction is never silently shown as if it belonged to the current image.

## Bounded buffers

`LatestSlot` is a single-slot condition-variable buffer. Writers always replace; readers get
the newest item. Consequences:

* no unbounded queue can grow, so latency cannot accumulate
* if inference is slower than the camera, intermediate frames are *dropped by design*
  (`slot.dropped` counts them, visible in the metrics JSON)
* `runtime.frame_skip` additionally limits how often inference runs

## Threads and shutdown

| thread | role | stop path |
|---|---|---|
| `capture` (daemon) | reads the source into the frame slot | `CaptureWorker.stop()` sets an event; slot is closed |
| `inference` (daemon) | waits for newer frames, runs the model | stops on its own event or when the frame slot closes |
| main | UI, measurements, rendering | `Pipeline.stop()` joins both with a 2 s timeout, releases the source |

Workers never touch OpenCV windows; only the main thread calls `imshow`/`waitKey`.
Worker exceptions are captured (`inference.error`) and surfaced instead of killing the process
silently.

## Component responsibilities

| module | responsibility | depends on |
|---|---|---|
| `config.py` | typed config, validation, meaningful errors | PyYAML |
| `camera.py` | device probing, capture props, reconnect, `LatestSlot`, sources | OpenCV |
| `depth_engine.py` | checkpoint loading (offline), precision/device selection, timed inference | transformers, torch |
| `temporal_filter.py` | EMA, temporal median, range EMA (pure numpy state) | numpy |
| `calibration.py` | candidate models, robust fitting, held-out split, JSON profiles | numpy, scipy |
| `distance_estimator.py` | region sampling, mode conversion, quality heuristics, point store | calibration |
| `geometry.py` | intrinsics, checkerboard calibration, scaling, undistort, back-projection | OpenCV |
| `anchor.py` | ArUco detection + solvePnP of a known-size marker | OpenCV aruco, geometry |
| `visualization.py` | colorizing, legend, markers, text overlay, window→frame mapping | OpenCV |
| `performance.py` | FPS meters, per-stage rolling latency, VRAM | torch |
| `pipeline.py` | worker orchestration | camera, depth_engine, temporal_filter |
| `benchmark.py` | resolution × precision measurement | depth_engine |

Dependency direction is one-way (no cycles): `main → pipeline → {camera, depth_engine}`,
`main → {distance_estimator, visualization, performance}`, and low-level modules never import
`main` or OpenCV windows.

## Timing model

* `preprocess_ms` - RGB conversion + processor resize/normalize + host→device copy
* `inference_ms` - CUDA forward. When `model.sync_cuda_timing` is true, `torch.cuda.synchronize()`
  is called around each stage, so the number reflects completed GPU work rather than launch
  time
* `postprocess_ms` - bicubic upsample + device→host copy
* `capture_to_infer_ms` - queue wait, i.e. how stale the frame was when inference started
* `capture_to_display_ms` - end-to-end latency measured in the GUI loop
* `depth_age_ms` - age of the depth map currently on screen

Throughput (fps) and latency (ms) are tracked separately: `FpsMeter` counts events in a 1 s
window, `RollingStats` keeps a120-sample window for mean/p95.

## Why not a queue

A typical producer/consumer queue would keep every frame. At 30 fps camera + 20 fps inference
the backlog grows by 10 frames/s, i.e. display lag increases without bound. The latest-frame
slot keeps display lag equal to one inference period, permanently.

## Extension points

`DepthEngine.infer()` returns a plain `DepthResult` with the depth array, timestamps and
metadata; `DistanceEstimator` and `geometry` are independent of the UI. That surface is
intentionally small so the same objects can later feed obstacle avoidance, point clouds or a
ROS 2 node without touching rendering code.
