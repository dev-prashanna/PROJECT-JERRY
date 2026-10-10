# Troubleshooting

Everything the application reports appears in two places: the **bottom status
bar** (`Last error: …`), and **`logs/app.log`**. Start there.

---

## Camera

### `No camera found` / the dropdown says *No camera found*

```bash
ls -l /dev/video*
cat /sys/class/video4linux/video0/name
```

- No `/dev/video*` → the webcam is disabled (BIOS/privacy shutter) or not
  plugged in. Re-plug, then press **Refresh**.
- Node exists but probing fails → another program holds the device:
  ```bash
  fuser -v /dev/video0        # shows the PID that is using it
  ```
  Close Zoom/Chrome/browser previews and press **Refresh**.

### `camera 0 could not be opened (not connected, busy or permission denied)`

| Cause | Fix |
| --- | --- |
| permissions | `sudo usermod -aG video $USER` (re-login), or `ls -l /dev/video0` |
| busy | `fuser -k /dev/video0` only if you are sure nothing else needs it |
| wrong index | try `python3 main.py --camera 1` or the dropdown + `Refresh` |
| 4K/720p unsupported by the mode | `python3 main.py --width 640 --height 480` |

### `camera disconnected — attempting to reconnect`

The driver stopped delivering frames (USB brown-out, suspend, another app
taking the device). The worker keeps retrying every second and the status
badge shows `reconnecting…`. If it never recovers, press **Stop** then
**Start Camera**.

To see the detection logic exercised without unplugging anything:

```bash
python3 -m pytest tests/test_camera.py -q -k "disconnect or reconnect"
```

### Very low preview FPS (~5 instead of 30)

The camera negotiated `YUYV` (uncompressed). On this machine `YUYV` measured
4.8 FPS at 640×480 while `MJPG` reached 30 FPS.

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext     # what the hardware offers
python3 main.py --fourcc MJPG
# or in .env:  VBMD_CAMERA_FOURCC=MJPG
```

Also expect a genuine drop in dark rooms: the sensor lengthens its exposure,
so the *measured* FPS in the status bar can fall below the negotiated rate.

---

## Model

### `ModelNotFoundError … Searched: …`

The checkpoint folder was not located. Nothing was downloaded.

```bash
export MODEL_PATH="/path/to/models/moondream2"   # folder with config.json + model.safetensors
python3 scripts/inspect_environment.py | sed -n '/Local Moondream/,/transformers/p'
python3 main.py --model-path "$MODEL_PATH"
```

`config.json` and `model.safetensors` (or `pytorch_model.bin`) must both be in
that folder.

### Network requests during startup / `LocalEntryNotFoundError … starmie-v1`

The Moondream code loads a tiny tokenizer from `moondream/starmie-v1`. The app
forces `HF_HUB_OFFLINE=1`, so it is read from the local Hugging Face cache.

- Error appears only on a machine whose cache is empty → warm it once with
  internet: `VQA_OFFLINE=0 python3 scripts/test_model.py`, then leave
  `VQA_OFFLINE=1`.
- You *want* to allow network access → `VQA_OFFLINE=0` in `.env`.

### `ModelApiError: Unsupported Moondream interface …`

The loaded object exposes neither `query()` nor `answer_question()`. Verify the
checkpoint really is Moondream 2 with transformers remote code:

```bash
python3 -c "import json;print(json.load(open('/path/to/models/moondream2/config.json')))"
```

Expected: `"architectures": ["HfMoondream"]` and an `auto_map` entry pointing
at `hf_moondream.HfMoondream`. A converted/GGUF-only export will not work with
this application.

### `CUDA out of memory …`

Measured steady-state usage is **4.2 GB of an 8 GB card**. OOM means something
else is on the GPU (another model, browser WebGL, a previous run that leaked).

```bash
nvidia-smi
python3 main.py --device cpu            # fallback, slower
python3 main.py --width 640 --height 480
```

The engine also releases its cached image encoding on OOM and returns an error
payload instead of crashing.

### `CUDA requested but unavailable — falling back to CPU`

Driver/CUDA mismatch or a CPU-only install. Check
`python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"`.
CPU mode is correct but slow: **14.1 s** per question vs **0.66 s** on the
RTX 4060 (measured on the sample image). On a machine with <8 GB free RAM the
loader retries with `bfloat16` automatically.

### The answer is wrong / invented

See *Known limitations* in the README. Practical mitigations:

- ask about objects that are clearly visible and reasonably large;
- freeze the frame and rephrase — a frozen frame removes "which instant did
  it look at?" ambiguity;
- improve lighting / focus / framing;
- `--no-grounding` turns off the appended "answer only from visible evidence"
  instruction if you want raw model behaviour for comparison.

There is no confidence score: Moondream does not expose a calibrated one, and
inventing a percentage would be misleading.

---

## GUI

### `Could not load the Qt platform plugin "xcb"` / `xcb` abort

```bash
echo $DISPLAY                 # must be non-empty on X11
python3 -c "from PySide6.QtWidgets import QApplication; import sys; a=QApplication(sys.argv); print(a.platformName())"
# if DISPLAY is empty but you are on Wayland:
QT_QPA_PLATFORM=wayland python3 main.py
# headless / over SSH:
QT_QPA_PLATFORM=offscreen python3 scripts/gui_selftest.py
```

Missing system libraries (rare when PySide6 is installed from pip):

```bash
sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxkbcommon-x11-0
```

### The window freezes when I close it while an answer is generating

Expected for up to `VQA_SHUTDOWN_TIMEOUT` (default 10 s): the app waits for the
in-flight request, then releases the worker and camera. The pending answer is
discarded. Lower the wait with `VQA_SHUTDOWN_TIMEOUT=3`.

### The preview stopped updating but the status bar says *connected*

The preview repaints from the latest-frame buffer. If `Last error` mentions
frames, check the log; otherwise press **Stop** / **Start Camera**.

---

## Questions are rejected

| Message | Meaning |
| --- | --- |
| `type a question first` | empty input |
| `question is too long (max 4000 characters)` | paste was huge |
| `the model is still loading — try again in a moment` | startup takes ~4–5 s |
| `the model failed to load — check the error log` | see *Model* above |
| `no camera frame available yet — start the camera` | start the camera first |
| `no frozen frame — freeze a frame first` | mode was switched unexpectedly |
| `a request is already queued — wait for the current answer` | queue is bounded (1 pending) by design |

---

## Getting more detail

```bash
python3 main.py --log-level DEBUG
tail -f logs/app.log
python3 scripts/inspect_environment.py --save    # refresh docs/environment_report.txt
python3 -m pytest tests/ -q                      # 88 tests, hardware-aware
python3 scripts/gui_selftest.py                  # drives the real window headlessly
```
