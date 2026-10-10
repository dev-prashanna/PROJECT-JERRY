# Installation

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| Ubuntu 22.04 / 24.04 (or similar Linux) | developed and tested on Ubuntu with X11 |
| Python **3.10+** | tested with 3.12.3 |
| A camera | USB or built-in UVC webcam exposed as `/dev/video*` |
| PyTorch with CUDA (optional) | CPU works, but ~20× slower (measured) |
| PySide6 6.6+ | GUI toolkit |
| An existing Moondream 2 checkpoint | **never downloaded by this app** |

Verified package set: `docs/environment_report.txt`.

> **Do not upgrade a working stack.** If your machine already runs a CUDA /
> PyTorch / transformers combination that loads the checkpoint, install
> nothing — this project only needs the packages listed in
> `requirements.txt` and uses whatever is present.

## 2. Option A — use the environment that already works (recommended)

On the development machine everything is already installed, so the only step
is to enter the project folder:

```bash
cd "moondream-live-vqa"
python3 scripts/inspect_environment.py     # confirm model, CUDA, camera, Qt
```

If all sections report OK, skip to §4.

## 3. Option B — fresh virtual environment

```bash
cd "moondream-live-vqa"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# GPU build (pick the index URL that matches your driver/CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

CPU-only machine:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Check the result:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. Existing Moondream 2 model configuration

The application **reuses the checkpoint already on disk** and never downloads
weights.

Auto-detected locations (first hit wins):

1. `MODEL_PATH` environment variable / CLI flag `--model-path`
2. `MODEL_PATH` in `moondream-live-vqa/.env`
3. `../models/moondream2` (this workspace — the detected location)
4. `./models/moondream2`, `../models/moondream`, `~/models/moondream2`

A directory counts as a checkpoint when it contains `config.json` **and** one
of `model.safetensors` / `pytorch_model.bin` / `model.pt`.

```bash
# explicit is always safe
export MODEL_PATH="/home/you/Documents/PROJECT VBMD/models/moondream2"
python3 main.py --model-path "$MODEL_PATH"
```

If nothing is found the program stops with a `ModelNotFoundError` listing every
path it searched — it does **not** fetch anything.

### Offline behaviour

`config.json` uses `trust_remote_code` (`hf_moondream.HfMoondream`), and the
model code fetches one small **tokenizer** (`moondream/starmie-v1`, ~2 MB) from
the Hugging Face Hub the very first time it runs. It is already cached at
`~/.cache/huggingface/hub/models--moondream--starmie-v1/`.

By default the app sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
(`VQA_OFFLINE=1`), so that lookup is served from cache and **no network access
happens at all**. On a machine with an empty cache, run once with internet:

```bash
VQA_OFFLINE=0 python3 scripts/test_model.py     # warms the tokenizer cache
VQA_OFFLINE=1 python3 main.py                   # back to offline
```

## 5. Camera setup

```bash
ls -l /dev/video*                  # device nodes
cat /sys/class/video4linux/video0/name
v4l2-ctl -d /dev/video0 --list-formats-ext     # supported modes (optional)
```

Notes for this class of webcam:

- **Use `MJPG`** (default): uncompressed `YUYV` measured **4.8 FPS**, `MJPG`
  measured **30 FPS** at 640×480.
- Permission: on Ubuntu the user must be in the `video` group
  (`sudo usermod -aG video $USER`, then re-login).
- Only one process can own the device — close Zoom/browser previews first.

```bash
python3 scripts/test_camera.py --list        # probe without capturing
python3 scripts/test_camera.py --save-snapshot
```

## 6. Verify before you run the GUI

```bash
python3 scripts/inspect_environment.py       # environment report
python3 scripts/test_model.py                # Phase 2: static-image VQA
python3 scripts/test_camera.py               # Phase 3: capture, FPS, restart
python3 main.py --cli --ask "What do you see?"   # Phase 4: end-to-end, no GUI
python3 -m pytest tests/ -q                  # automated suite (88 tests)
```

## 7. Run

```bash
python3 main.py                              # desktop application
python3 main.py --cli                        # terminal front-end
python3 scripts/gui_selftest.py              # automated GUI check + screenshot
```

## 8. Files created at runtime

| Path | Purpose |
| --- | --- |
| `logs/app.log` | rotating application log (2 MB × 3) |
| `outputs/*.png` | snapshots and GUI screenshots |
| `outputs/benchmark_*.json` | raw benchmark results |

Nothing outside the project folder is written, and no existing model file,
environment or project asset is ever deleted.
