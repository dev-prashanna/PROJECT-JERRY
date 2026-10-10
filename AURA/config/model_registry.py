"""Registry of the models AURA is allowed to use.

Two responsibilities:

1. **Discovery** — locate already-installed local weights without ever
   silently re-downloading them.
2. **Capability declaration** — describe what each model is allowed to do so
   the orchestrator can refuse requests that a model cannot satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import (
    KOKORO_SEARCH,
    MOONDREAM_SEARCH,
    WORKSPACE_ROOT,
    _find_model_dir,
    get_settings,
)


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    kind: str                     # stt | llm | vlm | tts | vad
    backend: str                  # faster-whisper | groq | transformers | kokoro
    location: str                 # path, model id or "cloud"
    installed: bool
    detail: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


def whisper_model_dir(model_id: str, download_root: Optional[str] = None) -> Path:
    root = Path(download_root or (WORKSPACE_ROOT / "models" / "whisper"))
    # faster-whisper stores snapshots under <root>/<model-id-with-dashes>/
    return root / model_id.replace("/", "--")


def describe_moondream() -> ModelDescriptor:
    path = _find_model_dir("moondream2", list(MOONDREAM_SEARCH))
    ok = bool(path) and (path / "model.safetensors").exists()
    return ModelDescriptor(
        key="moondream2",
        kind="vlm",
        backend="transformers",
        location=str(path) if path else "",
        installed=ok,
        detail="Moondream 2 vision-language model (local weights)",
        capabilities=("caption", "query", "detect"),
        notes="trust_remote_code checkpoint loaded with bfloat16 on CUDA",
    )


def describe_kokoro() -> ModelDescriptor:
    path = _find_model_dir("kokoro-82M", list(KOKORO_SEARCH))
    ok = bool(path) and (path / "kokoro-v1_0.pth").exists()
    voices = 0
    if path and (path / "voices").is_dir():
        voices = len(list((path / "voices").glob("*.pt")))
    return ModelDescriptor(
        key="kokoro-82m",
        kind="tts",
        backend="kokoro",
        location=str(path) if path else "",
        installed=ok,
        detail=f"Kokoro-82M local TTS ({voices} voices)",
        capabilities=("synthesize",),
        notes="24 kHz mono waveform, voice pack .pt files loaded from local dir",
    )


def describe_whisper(model_id: str, download_root: Optional[str] = None) -> ModelDescriptor:
    try:
        import faster_whisper  # noqa: F401
        backend_ok = True
    except Exception:
        backend_ok = False
    local = whisper_model_dir(model_id, download_root)
    installed = backend_ok and local.exists()
    return ModelDescriptor(
        key="whisper",
        kind="stt",
        backend="faster-whisper",
        location=str(local),
        installed=installed,
        detail=f"CTranslate2 whisper '{model_id}'",
        capabilities=("transcribe",),
        notes="" if installed else "weights will be fetched on first use",
    )


def describe_groq() -> ModelDescriptor:
    from config.settings import get_settings

    s = get_settings().llm
    has_key = s.available
    return ModelDescriptor(
        key="groq",
        kind="llm",
        backend="groq",
        location=s.base_url,
        installed=has_key,
        detail=f"primary={s.model} fallback={s.fallback_model}",
        capabilities=("chat", "route", "respond"),
        notes="cloud LLM; no images or audio are ever uploaded",
    )


def all_models() -> list[ModelDescriptor]:
    s = get_settings()
    return [
        describe_whisper(s.stt.model, s.stt.download_root),
        describe_groq(),
        describe_moondream(),
        describe_kokoro(),
    ]


def missing_models() -> list[ModelDescriptor]:
    return [m for m in all_models() if not m.installed]
