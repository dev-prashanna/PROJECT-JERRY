# Troubleshooting

## Camera shows black frames

**Symptom:** `CameraPanel` warns "Frames are uniformly black", or logs show
`blank frames detected`.

```bash
v4l2-ctl -d /dev/video0 --all | grep -i privacy
#   privacy 0x009a0910 (bool): value=1 flags=read-only
```

`value=1` means the **hardware privacy shutter** is engaged. It is
read-only through software — toggle the physical switch on the laptop
edge/bezel. AURA's blank-frame detector (`CameraManager._check_blank`) logs a
warning after 15 consecutive black frames; the benchmark script automatically
injects a synthetic test card so the VLM path can still be measured.

If the shutter is open but frames are still black, try the other node:

```bash
CAMERA_INDEX=1 python3 main.py --text "hello" --no-speak
```

## Whisper runs on CPU / is very slow

Check the log line `STT ready` and `verify_whisper.py` output — it should say
`device: cuda`, `compute_type: float16`, and ~0.3–0.5 s per utterance.

If it falls back to CPU you will see ~1.6 s+. Usual causes:

* `libcublas.so.12` missing → fixed automatically by
  `ensure_ct2_cuda_libs()`; verify with:
  `python3 -c "from src.utils.device_manager import ensure_ct2_cuda_libs; print(ensure_ct2_cuda_libs())"`
* VRAM exhausted by another process → `nvidia-smi`, close other GPU apps.
* `STT_DEVICE=cpu` set in `.env` — remove it.

## Groq errors

| Error | Cause | Fix |
|---|---|---|
| `AuthenticationError` / 401 | bad or revoked key | new key in `.env`, `chmod 600` |
| `RateLimitError` / 429 | free-tier limits | wait; the client already backs off and falls back to the deterministic router |
| `model_not_found` / 404 | model not on your account | use `openai/gpt-oss-120b` (default) — `llama-3.3-70b-versatile` is commonly unavailable |
| timeouts | network | increase `LLM_TIMEOUT_S`, or run `--offline` |

`python3 scripts/verify_groq.py` prints exactly which models your key can
reach.

## No microphone / no speakers

```bash
python3 main.py --demo --no-listen --no-speak    # fully headless
python3 main.py --cli --no-listen                # text in, speech out
```

Device selection issues (pipewire vs ALSA): set `MICROPHONE_DEVICE` /
`TTS_OUTPUT_DEVICE` in `.env` to indexes from
`python3 -c "import sounddevice as sd; print(sd.query_devices())"`.

## TTS produces no audio

* Check the player status in the GUI monitor ("pending" should drain).
* `verify_kokoro.py` should report RTF < 1.0 on GPU.
* If `voices/` is missing, Kokoro cannot load a voice — re-download the
  `kokoro-82M` weights including `voices/`.

## VRAM out of memory

Moondream-2 takes ~4.2 GB; Whisper + Kokoro add ~1 GB. On an 8 GB card this
leaves little headroom. Close other GPU processes, or reduce
`VLM_MAX_NEW_TOKENS`. If it still OOMs, the pipeline catches it and speaks a
friendly message instead of crashing.

## Routing seems wrong / too chatty

* Force the local router to compare: `python3 main.py --cli --router
  deterministic`
* `--router compare` logs both decisions side by side.
* Ambiguous references ("what about the other one?") are *intentionally*
  answered with a clarifying question by the deterministic router — this is
  safety behaviour, not a bug.

## Tests fail after pulling changes

```bash
python3 -m pytest tests/ -q
```

The suite is fully offline (all cloud/hardware is faked in
`tests/conftest.py`). If collection fails, check that you are in the `AURA/`
directory and that `config/` and `src/` are importable (they are on `sys.path`
via `conftest.py`).
