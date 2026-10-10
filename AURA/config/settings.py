"""AURA application configuration.

Resolution order (later wins):

1. Built-in defaults (documented, safe).
2. ``AURA/.env`` file.
3. Real process environment variables.
4. Explicit overrides passed by the caller (CLI/GUI settings panel).

Design rules:

* Secrets (``GROQ_API_KEY``) are never returned by :func:`describe` and are
  never written to logs.
* Local model directories are auto-detected; nothing is ever downloaded
  unless :class:`STTSettings` explicitly allows it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: ``AURA/``
APP_ROOT = Path(__file__).resolve().parents[1]
#: Parent workspace that owns ``models/``
WORKSPACE_ROOT = APP_ROOT.parent

ENV_FILE = APP_ROOT / ".env"
LOG_DIR = APP_ROOT / "logs"
OUTPUT_DIR = APP_ROOT / "outputs"
METRICS_DIR = APP_ROOT / "benchmarks" / "results"

for _d in (LOG_DIR, OUTPUT_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _find_model_dir(name: str, candidates: list[Path]) -> Optional[Path]:
    for c in candidates:
        if c.is_dir():
            return c
    return None


MOONDREAM_SEARCH: tuple[Path, ...] = (
    WORKSPACE_ROOT / "models" / "moondream2",
    APP_ROOT / "models" / "moondream2",
    Path.home() / "models" / "moondream2",
)
KOKORO_SEARCH: tuple[Path, ...] = (
    WORKSPACE_ROOT / "models" / "kokoro-82M",
    APP_ROOT / "models" / "kokoro-82M",
    Path.home() / "models" / "kokoro-82M",
)


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` file into ``os.environ``.

    Existing environment variables are never overwritten.  Values are never
    returned to callers that log them, only the list of newly-set keys.
    """
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return loaded
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


load_env_file()


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if raw == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Settings blocks
# ---------------------------------------------------------------------------

def _resolve_llm_provider() -> str:
    from config.provider_presets import LLM_ALIASES

    name = (_get("LLM_PROVIDER", "groq") or "groq").strip().lower()
    return LLM_ALIASES.get(name, name)


def _llm_preset() -> dict[str, Any]:
    from config.provider_presets import resolve_llm_preset

    return resolve_llm_preset(_resolve_llm_provider())


def _first_env(names: list[str]) -> str:
    for n in names:
        v = _get(n)
        if v:
            return v
    return ""


def _resolve_llm_api_key() -> str:
    """LLM_API_KEY → provider-specific env → (groq: GROQ_API_KEY)."""
    explicit = _get("LLM_API_KEY")
    if explicit:
        return explicit
    preset = _llm_preset()
    key = _first_env(list(preset.get("key_env", [])))
    if key:
        return key
    if preset.get("protocol") == "groq":
        return _get("GROQ_API_KEY")
    return ""


def _resolve_llm_base_url() -> str:
    explicit = _get("LLM_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    if _resolve_llm_provider() == "groq":
        return _get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    preset_url = (_llm_preset().get("base_url") or "").rstrip("/")
    return preset_url


def _resolve_llm_model() -> str:
    explicit = _get("LLM_MODEL")
    if explicit:
        return explicit
    if _resolve_llm_provider() == "groq":
        return _get("GROQ_MODEL", "openai/gpt-oss-120b")
    preset_model = _llm_preset().get("model") or ""
    if preset_model:
        return preset_model
    return "gpt-4o-mini"


def _resolve_llm_fallback_model() -> str:
    explicit = _get("LLM_FALLBACK_MODEL")
    if explicit:
        return explicit
    if _resolve_llm_provider() == "groq":
        return _get("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")
    return _llm_preset().get("fallback") or ""


@dataclass(frozen=True)
class LLMSettings:
    """Main chat-model configuration (routing + response composition).

    ``provider`` selects the backend: ``groq`` (default, native SDK),
    ``openai`` / ``moonshot``(Kimi) / ``deepseek`` / ``mimo`` /
    ``openai_compat`` (any OpenAI-compatible server), or ``anthropic``
    (Claude).  Legacy ``GROQ_*`` variables keep working when the provider
    is groq.
    """

    provider: str = field(default_factory=_resolve_llm_provider)
    api_key: str = field(default_factory=_resolve_llm_api_key)
    model: str = field(default_factory=_resolve_llm_model)
    fallback_model: str = field(default_factory=_resolve_llm_fallback_model)
    base_url: str = field(default_factory=_resolve_llm_base_url)
    timeout_s: float = field(default_factory=lambda: _get_float("LLM_TIMEOUT_S", _get_float("GROQ_TIMEOUT_S", 25.0)))
    max_retries: int = field(default_factory=lambda: _get_int("LLM_MAX_RETRIES", _get_int("GROQ_MAX_RETRIES", 2)))
    max_history_turns: int = field(default_factory=lambda: _get_int("GROQ_MAX_HISTORY_TURNS", 6))
    max_completion_tokens: int = field(
        default_factory=lambda: _get_int(
            "LLM_MAX_COMPLETION_TOKENS", _get_int("GROQ_MAX_COMPLETION_TOKENS", 1024)
        )
    )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def protocol(self) -> str:
        from config.provider_presets import resolve_llm_preset

        return str(resolve_llm_preset(self.provider).get("protocol", "openai"))


@dataclass(frozen=True)
class STTSettings:
    model: str = field(default_factory=lambda: _get("STT_MODEL", "small.en"))
    device: str = field(default_factory=lambda: _get("STT_DEVICE", "auto"))
    compute_type: str = field(default_factory=lambda: _get("STT_COMPUTE_TYPE", "int8"))
    download_root: str = field(
        default_factory=lambda: _get("STT_DOWNLOAD_ROOT", str(WORKSPACE_ROOT / "models" / "whisper"))
    )
    allow_download: bool = field(default_factory=lambda: _get_bool("STT_ALLOW_DOWNLOAD", True))
    sample_rate: int = field(default_factory=lambda: _get_int("STT_SAMPLE_RATE", 16000))
    microphone_device: str = field(default_factory=lambda: _get("MICROPHONE_DEVICE", "default"))
    vad_enabled: bool = field(default_factory=lambda: _get_bool("VAD_ENABLED", True))
    vad_threshold: float = field(default_factory=lambda: _get_float("VAD_THRESHOLD", 0.5))
    silence_timeout_s: float = field(default_factory=lambda: _get_float("SILENCE_TIMEOUT", 1.2))
    max_utterance_s: float = field(default_factory=lambda: _get_float("MAX_UTTERANCE_SECONDS", 15.0))
    min_utterance_s: float = field(default_factory=lambda: _get_float("MIN_UTTERANCE_SECONDS", 0.35))
    language: str = field(default_factory=lambda: _get("STT_LANGUAGE", "en"))


@dataclass(frozen=True)
class CameraSettings:
    device_index: int = field(default_factory=lambda: _get_int("CAMERA_INDEX", 0))
    width: int = field(default_factory=lambda: _get_int("CAMERA_WIDTH", 640))
    height: int = field(default_factory=lambda: _get_int("CAMERA_HEIGHT", 480))
    fps: int = field(default_factory=lambda: _get_int("CAMERA_FPS", 30))
    fourcc: str = field(default_factory=lambda: _get("CAMERA_FOURCC", "MJPG"))
    backend: str = field(default_factory=lambda: _get("CAMERA_BACKEND", "auto"))
    auto_reconnect: bool = field(default_factory=lambda: _get_bool("CAMERA_AUTO_RECONNECT", True))
    reconnect_interval_s: float = field(
        default_factory=lambda: _get_float("CAMERA_RECONNECT_INTERVAL_S", 2.0)
    )
    max_frame_age_s: float = field(default_factory=lambda: _get_float("CAMERA_MAX_FRAME_AGE_S", 1.5))
    enabled: bool = field(default_factory=lambda: _get_bool("CAMERA_ENABLED", True))


@dataclass(frozen=True)
class VLMSettings:
    """Vision-model configuration.

    ``provider`` selects the engine: ``moondream`` (local default) or any
    OpenAI-compatible vision backend (``openai``, ``groq``,
    ``openai_compat``, …) which is reached through ``base_url``/``api_key``/
    ``model``.  Local weights still come from ``model_path``.
    """

    provider: str = field(
        default_factory=lambda: (_get("VLM_PROVIDER", "moondream") or "moondream").strip().lower()
    )
    model_path: str = field(
        default_factory=lambda: _get("VLM_MODEL_PATH", str(_find_model_dir("moondream2", list(MOONDREAM_SEARCH)) or MOONDREAM_SEARCH[0]))
    )
    model: str = field(default_factory=lambda: _get("VLM_MODEL", ""))
    api_key: str = field(default_factory=lambda: _get("VLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: _get("VLM_BASE_URL", ""))
    device: str = field(default_factory=lambda: _get("VLM_DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: _get("VLM_DTYPE", "auto"))
    max_new_tokens: int = field(default_factory=lambda: _get_int("VLM_MAX_NEW_TOKENS", 320))
    temperature: float = field(default_factory=lambda: _get_float("VLM_TEMPERATURE", 0.0))
    queue_size: int = field(default_factory=lambda: _get_int("VLM_QUEUE_SIZE", 1))
    shutdown_timeout_s: float = field(default_factory=lambda: _get_float("VLM_SHUTDOWN_TIMEOUT", 10.0))


@dataclass(frozen=True)
class DepthSettings:
    """Monocular depth estimation (Depth Anything V2) for the GUI + VLM.

    Runs on a background thread fed by AURA's existing camera buffer — it
    never opens a second camera.  The colourized depth map is shown beside
    the live preview; the raw map is exposed to the VLM so answers can talk
    about near/far.
    """

    enabled: bool = field(default_factory=lambda: _get_bool("DEPTH_ENABLED", True))
    model_id: str = field(
        default_factory=lambda: _get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
    )
    device: str = field(default_factory=lambda: _get("DEPTH_DEVICE", "auto"))
    precision: str = field(
        default_factory=lambda: _get("DEPTH_PRECISION", "autocast_bf16")
    )
    input_size: int = field(default_factory=lambda: _get_int("DEPTH_INPUT_SIZE", 518))
    colormap: str = field(default_factory=lambda: _get("DEPTH_COLORMAP", "TURBO"))
    max_fps: float = field(default_factory=lambda: _get_float("DEPTH_MAX_FPS", 12.0))
    smoothing: bool = field(default_factory=lambda: _get_bool("DEPTH_SMOOTHING", True))
    smoothing_alpha: float = field(default_factory=lambda: _get_float("DEPTH_SMOOTH_ALPHA", 0.45))
    warmup_iters: int = field(default_factory=lambda: _get_int("DEPTH_WARMUP_ITERS", 2))


@dataclass(frozen=True)
class TTSSettings:
    model_path: str = field(
        default_factory=lambda: _get("TTS_MODEL_PATH", str(_find_model_dir("kokoro-82M", list(KOKORO_SEARCH)) or KOKORO_SEARCH[0]))
    )
    voice: str = field(default_factory=lambda: _get("TTS_VOICE", "af_heart"))
    speed: float = field(default_factory=lambda: _get_float("TTS_SPEED", 1.0))
    lang_code: str = field(default_factory=lambda: _get("TTS_LANG_CODE", "a"))
    device: str = field(default_factory=lambda: _get("TTS_DEVICE", "auto"))
    output_device: int = field(default_factory=lambda: _get_int("TTS_OUTPUT_DEVICE", -1))
    sample_rate: int = field(default_factory=lambda: _get_int("TTS_SAMPLE_RATE", 24000))
    max_sentence_chars: int = field(default_factory=lambda: _get_int("TTS_MAX_SENTENCE_CHARS", 300))


@dataclass(frozen=True)
class OrchestratorSettings:
    interaction_mode: str = field(default_factory=lambda: _get("INTERACTION_MODE", "ptt"))
    router_mode: str = field(default_factory=lambda: _get("ROUTER_MODE", "llm"))
    response_strategy: str = field(default_factory=lambda: _get("RESPONSE_STRATEGY", "auto"))
    offline: bool = field(default_factory=lambda: _get_bool("AURA_OFFLINE", False))
    queue_size: int = field(default_factory=lambda: _get_int("ORCHESTRATOR_QUEUE_SIZE", 2))
    tool_timeout_s: float = field(default_factory=lambda: _get_float("TOOL_TIMEOUT_S", 30.0))
    deterministic_router_confidence: float = field(
        default_factory=lambda: _get_float("DETERMINISTIC_ROUTER_CONFIDENCE", 0.92)
    )
    memory_turns: int = field(default_factory=lambda: _get_int("MEMORY_TURNS", 8))
    persist_history: bool = field(default_factory=lambda: _get_bool("PERSIST_HISTORY", False))
    wake_word: str = field(default_factory=lambda: _get("WAKE_WORD", "hey robot"))


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    stt: STTSettings = field(default_factory=STTSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    depth: DepthSettings = field(default_factory=DepthSettings)
    vlm: VLMSettings = field(default_factory=VLMSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)
    log_level: str = field(default_factory=lambda: _get("AURA_LOG_LEVEL", "INFO"))

    # -- serialisation ----------------------------------------------------
    def as_dict(self, redact_secrets: bool = True) -> dict[str, Any]:
        """Flat ``SECTION.KEY`` mapping, optionally redacting secrets."""
        out: dict[str, Any] = {}
        for sec_name in self.__dataclass_fields__:
            sec = getattr(self, sec_name)
            if not hasattr(sec, "__dataclass_fields__"):
                out[sec_name] = sec
                continue
            for f in fields(sec):
                value = getattr(sec, f.name)
                if redact_secrets and f.name in {"api_key"} and value:
                    value = f"<redacted len={len(value)}>"
                out[f"{sec_name}.{f.name}"] = value
        if redact_secrets:
            out.pop("api_key", None)
        return out

    def redacted_env(self) -> dict[str, str]:
        """Environment-style view safe for logging/GUI display."""
        out = {}
        for k, v in self.as_dict().items():
            out[k.upper().replace(".", "_")] = str(v)
        return out

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a new Settings with dotted-key overrides, e.g.

        ``s.with_overrides(**{"stt.model": "base.en"})``
        """
        updates: dict[str, dict[str, Any]] = {}
        for key, value in kwargs.items():
            if "." in key:
                sec, _, attr = key.partition(".")
            else:  # bare key: try to resolve against every section
                matches = [
                    name
                    for name in self.__dataclass_fields__
                    if hasattr(getattr(self, name), "__dataclass_fields__")
                    and hasattr(getattr(self, name), key)
                ]
                if not matches:
                    raise KeyError(f"Unknown setting: {key}")
                sec, attr = matches[0], key
            updates.setdefault(sec, {})[attr] = value
        new: dict[str, Any] = {}
        for sec, patch in updates.items():
            current = getattr(self, sec)
            known = {f.name for f in fields(current)}
            unknown = set(patch) - known
            if unknown:
                raise KeyError(f"Unknown {sec} setting(s): {sorted(unknown)}")
            new[sec] = replace(current, **patch)
        return replace(self, **new)


_settings: Optional[Settings] = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide cached settings object."""
    global _settings
    if _settings is None or refresh:
        load_env_file()
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Testing hook."""
    global _settings
    _settings = None
