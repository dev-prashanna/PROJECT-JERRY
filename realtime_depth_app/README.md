# Real-Time Monocular Depth Perception System

Live RGB + Depth Anything V2 depth from a normal webcam, with point-and-click distance
inspection, optional empirical metric calibration, camera-intrinsic geometry utilities and
measured performance overlays.

```
LEFT: live camera + measurement points     RIGHT: aligned depth map + legend
```

---

## 1. Requirements and installation

Already available on this machine (no extra installs needed):

| component | version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.0+cu130 (CUDA 13.0, RTX 4060 Laptop, 8 GB) |
| transformers | 5.14.1 |
| OpenCV | 5.0.0 (with `aruco`) |
| numpy / scipy / PyYAML / pytest | 2.5.3 / 1.18.1 / 6.0.1 / 7.4.4 |

If you ever need to reinstall:

```bash
pip install -r requirements.txt
```

The checkpoint is **not** bundled with this repository. It already lives in the Hugging Face
cache (`~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf`) and is
loaded offline (`model.local_files_only: true`). Nothing is downloaded at runtime.

---

## 2. Running the application

From the project root:

```bash
cd realtime_depth_app
python3 main.py
```

That is the whole command. The window opens with the camera on the left and the depth map on
the right.

Useful variants:

```bash
python3 main.py --list-cameras                 # find a usable camera index
python3 main.py --camera 1                     # switch camera
python3 main.py --video /path/to/clip.mp4      # run on a video file
python3 main.py --image ../assets/sample_frame.jpeg
python3 main.py --device cpu                   # force CPU fallback
python3 main.py --precision autocast_bf16      # measured ~1.8x faster inference
python3 main.py --input-size 378               # smaller model input = faster, coarser
python3 main.py --frame-skip 2                 # depth on every 2nd frame
python3 main.py --no-smoothing                 # disable temporal filtering
python3 main.py --profile calibration_data/profile.json
python3 main.py --headless 60                  # no GUI: 60 frames + metrics JSON
python3 main.py --benchmark                    # resolution x precision benchmark
```

Every option can also be set permanently in `config.yaml`.

---

## 3. Controls

| key | action |
|---|---|
| `Q` / `ESC` | quit |
| `P` | pause / resume the display |
| `C` | clear all measurement points |
| `R` | reset: points, smoothing state, depth color range |
| `M` | toggle center crosshair measurement |
| `F` | toggle FPS / latency overlay |
| `D` | cycle color map |
| `G` | toggle grayscale depth view |
| `T` | toggle temporal smoothing |
| `S` | save a screenshot to `outputs/` |
| `H` | toggle the help overlay |
| `B` | print the measured performance snapshot to the terminal |

Mouse:

* **left click** on either panel - add a measurement point (up to `measurement.max_points`)
* **right click** - remove the nearest point

Each point shows `id: value [quality]` where quality is a **heuristic** (`good`/`fair`/`poor`)
derived from local depth variation, valid-pixel fraction and calibration range. It is not a
confidence probability.

---

## 4. What the numbers mean (read this before trusting any value)

The checkpoint is `depth-anything/Depth-Anything-V2-Small-hf`. Its configuration says
`depth_estimation_type: relative`, and the Depth Anything V2 paper states the model predicts
**affine-invariant inverse depth**:

* **higher value = closer**
* the scale and shift are arbitrary and **can change per image**
* there are **no meters** in the raw output

So the app has three explicit modes:

| mode | when | display |
|---|---|---|
| `relative` (default) | no calibration profile | `R = 3.42 a.u.` - arbitrary units |
| `calibrated` | `calibration_data/profile.json` exists | `1.42 m` - approximate, scene dependent |
| `native metric` | only if the checkpoint config says `metric` | `1.42 m` - **not active for this checkpoint** |

The status line always shows the active mode, and out-of-range predictions are flagged
`~extrapolated`. The app never invents meters from uncalibrated output.

---

## 5. Calibration workflow (optional, needed for meters)

```bash
# 1) collect reference distances with a tape measure (6 distances x 5 samples)
python3 tools/calibrate.py --distances 0.5,1.0,1.5,2.0,3.0,4.0 --samples 5 --scene room1

# 2) the tool fits both candidate models, prints held-out errors and saves the profile
#    -> calibration_data/profile.json

# 3) validate on distances that were NOT used for fitting
python3 tools/calibrate.py --check --distances 0.75,2.5,3.5

# 4) start the app - it picks up the profile automatically
python3 main.py
```

Full protocol, fitting math and limitations: [docs/depth_calibration.md](docs/depth_calibration.md).

### Camera intrinsics (optional, geometry only)

```bash
python3 tools/calibrate_intrinsics.py --pattern 9x6 --square 0.025
# -> calibration_data/intrinsics.json
```

Intrinsics enable back-projection `X = (u-cx)·Z/fx`, `Y = (v-cy)·Z/fy` and ArUco pose
estimation. They do **not** create metric scale.

---

## 6. Measured performance

All numbers below were measured on this machine (Ryzen 7 8845HS / RTX 4060 Laptop 8 GB /
torch 2.13.0+cu130). Reproduce with `python3 main.py --benchmark`.

Model-only inference (40 iterations after warmup):

| input | precision | mean ms | fps | VRAM |
|---|---|---|---|---|
| 252 | fp32 | 21.8 | 46.0 | 136 MB |
| 252 | autocast_bf16 | 21.4 | 46.7 | 144 MB |
| 378 | fp32 | 30.9 | 32.4 | 178 MB |
| 378 | autocast_bf16 | 24.1 | 41.5 | 191 MB |
| 518 | fp32 | 53.7 | 18.6 | 242 MB |
| 518 | autocast_bf16 | 29.6 | 33.7 | 266 MB |

End-to-end with the live camera (`python3 main.py --headless 90 --precision autocast_bf16`):

* camera: 27.9 fps, depth: **27.5 fps**
* preprocessing 7.1 ms, inference 13.7 ms, postprocess 0.5 ms
* VRAM peak 276 MB

See [docs/performance.md](docs/performance.md) for details and tuning.

---

## 7. Tests

```bash
cd realtime_depth_app
python3 -m pytest tests -q
```

92 tests cover: model loading and output conventions, coordinate mapping between panels,
calibration fitting/persistence/extrapolation, region measurement, geometry and checkerboard
calibration, ArUco pose recovery on synthetic markers, temporal filters, visualization and
pipeline threading/shutdown.

---

## 8. Repository layout

```
realtime_depth_app/
├── main.py                  entry point, UI loop, keyboard/mouse handling
├── config.yaml              all runtime settings (validated at startup)
├── requirements.txt
├── src/
│   ├── config.py            config schema + validation
│   ├── camera.py            camera/video/image sources, latest-frame slot, capture thread
│   ├── depth_engine.py      model loading, preprocessing, CUDA inference, timing
│   ├── pipeline.py          capture + inference workers, bounded buffers, shutdown
│   ├── distance_estimator.py point/region measurement, mode handling, quality heuristics
│   ├── calibration.py       candidate fits, held-out evaluation, profile persistence
│   ├── geometry.py          intrinsics, checkerboard calibration, back-projection, undistort
│   ├── anchor.py            optional ArUco known-size validation (Approach C)
│   ├── temporal_filter.py   EMA / temporal median state
│   ├── visualization.py     panels, color maps, legend, markers, window mapping
│   ├── performance.py       FPS meters, stage latency, VRAM
│   └── benchmark.py         resolution x precision benchmarks
├── tools/
│   ├── calibrate.py         metric calibration collection / fitting / checking
│   └── calibrate_intrinsics.py  checkerboard intrinsic calibration
├── tests/                   pytest suite
├── benchmarks/              generated benchmark reports
├── calibration_data/        generated profiles + intrinsics (not committed)
├── outputs/                 screenshots + headless reports (not committed)
└── docs/                    architecture, model notes, calibration, performance, limitations
```

---

## 9. Documentation

| file | contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | modules, threading, data flow, frame identity |
| [docs/model_setup.md](docs/model_setup.md) | which checkpoint, relative vs metric, offline loading |
| [docs/depth_calibration.md](docs/depth_calibration.md) | calibration protocol, math, validation, limits |
| [docs/performance.md](docs/performance.md) | measured benchmarks and tuning levers |
| [docs/limitations.md](docs/limitations.md) | what this system does **not** do |

---

## 10. Changing common settings

| I want to... | edit |
|---|---|
| change camera | `source.index` or `--camera N` |
| change checkpoint | `model.repo_id` (or `model.local_dir` for a local folder) |
| change input resolution | `model.input_size` (multiple of 14) |
| change precision | `model.precision`: `fp32`, `autocast_bf16`, `autocast_fp16`, `fp16` |
| change CPU/GPU | `model.device`: `auto`, `cuda`, `cpu` |
| change smoothing | `depth.smoothing.alpha`, `median_window`, `enabled` |
| change color map | `depth.visualization.colormap` or press `D` |
| change neighborhood for measurements | `measurement.region_size` (odd) |
| use another calibration profile | `calibration.profile` or `--profile PATH` |
| run on a video file | `source.type: video` + `source.path`, or `--video PATH` |
