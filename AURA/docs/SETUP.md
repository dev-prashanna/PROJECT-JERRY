# Setup

## Requirements

* Linux (tested on kernel 7.x / Ubuntu-family), Python 3.12
* NVIDIA GPU with ≥ 6 GB VRAM (tested: RTX 4060 Laptop, 8 GB)
* Microphone + speakers (or `--no-listen` / `--no-speak` for headless use)
* A Groq API key (free tier is fine)

## Install

```bash
cd AURA
python3 -m pip install --user --break-system-packages -r requirements.txt
```

Key packages: `torch` (CUDA build), `faster-whisper`, `transformers`,
`hf-moondream`/`ctranslate2`, `kokoro`, `groq`, `PySide6`, `opencv-python`,
`sounddevice`, `pytest`.

> On this machine the system Python is PEP-668 managed, hence
> `--user --break-system-packages`. Inside a plain venv, plain
> `pip install -r requirements.txt` works.

## Weights

Shared model weights live next to the project:

```
../models/moondream2/     # ~3.7 GB, HF transformers checkpoint
../models/kokoro-82M/     # ~340 MB + voices/*.bin
```

`config/model_registry.py` resolves these paths; Whisper downloads its
`small.en` model to the HF cache on first use. Override any location with the
`*_MODEL_PATH` variables in `.env`.

## API key

AURA picks its chat model from `LLM_PROVIDER`. Groq is the default (free
tier); set the key for whichever provider you choose.

```bash
cp .env.example .env
# edit .env — pick ONE provider:
#   LLM_PROVIDER=groq       && GROQ_API_KEY=...
#   LLM_PROVIDER=openai     && OPENAI_API_KEY=...
#   LLM_PROVIDER=anthropic  && ANTHROPIC_API_KEY=...
#   LLM_PROVIDER=moonshot   && MOONSHOT_API_KEY=...   (Kimi)
#   LLM_PROVIDER=deepseek   && DEEPSEEK_API_KEY=...
#   LLM_PROVIDER=mimo       && MIMO_API_KEY=...
#   LLM_PROVIDER=openai_compat && LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...
#     (vLLM, Ollama, LM Studio, Together, Mistral — any OpenAI-API server)
chmod 600 .env
```

Override per run instead of editing `.env`:

```bash
python3 main.py --text "hi" --llm-provider anthropic --llm-model claude-haiku-4-5
```

The vision model is chosen the same way via `VLM_PROVIDER` (`moondream` is
the local default; `openai`/`groq`/`openai_compat` reach cloud vision).

`llama-3.3-70b-versatile` is **not** available on all accounts — the default
Groq model is `openai/gpt-oss-120b`, which is. Verify with:

```bash
python3 scripts/verify_groq.py
```

## CUDA / CTranslate2 note

System torch is a CUDA **13** build, but CTranslate2 (used by
faster-whisper) links against `libcublas.so.12`. AURA installs
`nvidia-cublas-cu12` + `nvidia-cuda-runtime-cu12` and preloads them at
startup in `src/utils/device_manager.py::ensure_ct2_cuda_libs()`
(ctypes, `RTLD_GLOBAL`, cudart before cublas). You should not need to set
`LD_LIBRARY_PATH` manually.

## Audio devices

List devices:

```bash
python3 -c "import sounddevice as sd; [print(i, d['name'], d['max_input_channels'], d['max_output_channels']) for i, d in enumerate(sd.query_devices())]"
```

Pin a specific device in `.env` (`STT_INPUT_DEVICE`, `TTS_OUTPUT_DEVICE`,
indexes from the listing above). On pipewire systems the `default` device
usually works.

## Camera

AURA opens `/dev/video0` by default (OpenCV). Check the device:

```bash
v4l2-ctl -d /dev/video0 --all | grep -i privacy
```

If `privacy` reports `1`, the laptop's hardware shutter is engaged — frames
will be black until you toggle the physical switch. AURA logs a warning after
15 uniformly black frames and shows it in the GUI.

## Verify the install

```bash
python3 scripts/inspect_environment.py --probe-groq   # hardware + key check
python3 scripts/verify_whisper.py                     # STT on GPU
python3 scripts/verify_groq.py                        # routing + latency
python3 scripts/verify_moondream.py                   # VLM inference
python3 scripts/verify_kokoro.py                      # TTS + RTF
python3 -m pytest tests/ -q                           # full offline test suite
```

## Run

```bash
python3 main.py --demo --no-speak        # acceptance demo (headless-safe)
python3 main.py --text "What do you see in front of you?"
python3 main.py --cli
python3 main.py --gui
python3 scripts/benchmark_pipeline.py    # writes reports/benchmark.{json,txt}
```
