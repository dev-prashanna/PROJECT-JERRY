"""Application configuration for the Moondream live VQA application.

Configuration is resolved in this order (later wins):

1. Built-in defaults (safe, documented values).
2. A ``.env`` file located next to this repository (``moondream-live-vqa/.env``).
3. Real process environment variables.
4. Command line arguments parsed by ``main.py``.

The model location is intentionally configurable: the Moondream 2 weights are
already installed on this machine and must never be downloaded twice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: ``moondream-live-vqa/``  (this repository)
APP_ROOT = Path(__file__).resolve().parents[1]
#: Parent folder (the larger VBMD workspace that already owns ``models/``)
WORKSPACE_ROOT = APP_ROOT.parent

LOG_DIR = APP_ROOT / "logs"
ENV_FILE = APP_ROOT / ".env"


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------
# The Moondream code loads a small tokenizer via ``tokenizers`` from the
# Hugging Face Hub (``moondream/starmie-v1``).  The checkpoint itself is read
# from disk, but that one lookup would still touch the network on a cold cache.
# These variables must be set *before* ``huggingface_hub`` is imported, so
# they are applied at module import time.  Set ``VQA_OFFLINE=0`` to allow
# hub access (e.g. on a machine whose cache is empty).
_OFFLINE = os.environ.get("VQA_OFFLINE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
if _OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
OFFLINE_MODE = _OFFLINE

#: Locations checked (in order) when ``MODEL_PATH`` is not supplied.
MODEL_SEARCH_CANDIDATES: tuple[Path, ...] = (
    WORKSPACE_ROOT / "models" / "moondream2",
    APP_ROOT / "models" / "moondream2",
    WORKSPACE_ROOT / "models" / "moondream",
    Path.home() / "models" / "moondream2",
)


# ---------------------------------------------------------------------------
# .env loading (tiny parser, avoids adding a dependency)
# ---------------------------------------------------------------------------

def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` file and export entries into ``os.environ``.

    Existing environment variables are never overwritten.  Returns the
    variables that were newly set so callers can inspect them.
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


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """OpenCV camera acquisition settings."""

    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    #: FourCC code.  ``MJPG`` is important: uncompressed ``YUYV`` on this
    #: machine's integrated webcam collapses to ~5 FPS, while ``MJPG`` runs at
    #: the advertised 30 FPS.  Set to ``""`` to let the driver decide.
    fourcc: str = "MJPG"
    #: ``auto`` | ``v4l2`` | ``any`` (Linux).  ``auto`` prefers V4L2.
    backend: str = "auto"
    #: Consecutive failed reads tolerated before the device is declared lost.
    max_read_failures: int = 10
    #: Seconds between reconnection attempts after a disconnect.
    reconnect_interval: float = 1.0
    #: Automatically try to reopen a lost camera (user can still press Start).
    auto_reconnect: bool = True
    #: Upper bound of camera indices probed by :func:`discover_cameras`.
    probe_max_index: int = 6

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class ModelConfig:
    """Moondream 2 loading and inference settings."""

    #: ``None`` -> resolved through :func:`resolve_model_path`.
    model_path: Optional[str] = None
    #: ``auto`` | ``cuda`` | ``cpu`` (``mps`` is accepted for completeness).
    device: str = "auto"
    #: ``auto`` | ``bf16`` | ``fp16`` | ``fp32``.
    dtype: str = "auto"
    #: Never hit the network: the weights live on disk already.
    local_files_only: bool = True
    trust_remote_code: bool = True
    #: ``auto`` lets the engine pick the best supported API for this build.
    api: str = "auto"
    #: Sampling settings forwarded to ``model.query(..., settings=...)``.
    max_new_tokens: int = 256
    temperature: float = 0.0

    @property
    def sampling_settings(self) -> dict:
        return {
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
        }


@dataclass
class InferenceConfig:
    """Pipeline behaviour around the model."""

    #: Capacity of the request queue *excluding* the job being processed.
    #: ``0`` means "reject while busy".  Never unbounded.
    queue_size: int = 1
    #: Append an image-grounding instruction to every question.
    grounding_prompt: bool = True
    grounding_text: str = (
        "Answer using only visible evidence in the provided image. "
        "If the answer cannot be determined visually, say that it is unclear. "
        "Keep the answer short."
    )
    #: Optional longest-side resize applied *before* encoding.  ``None``
    #: disables resizing so fine detail (small objects, text) is preserved.
    encode_max_side: Optional[int] = None
    #: Seconds to wait for an in-flight request during shutdown.
    shutdown_timeout: float = 10.0

    def __post_init__(self) -> None:
        # Guard against tuple/str values coming from bad env entries.
        try:
            self.queue_size = max(0, int(self.queue_size))
        except (TypeError, ValueError):
            self.queue_size = 1
        if self.encode_max_side is not None:
            try:
                side = int(self.encode_max_side)
            except (TypeError, ValueError):
                side = 0
            self.encode_max_side = side if side > 0 else None
        self.grounding_prompt = bool(self.grounding_prompt)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    log_level: str = "INFO"
    log_file: Optional[str] = None
    resolved_model_path: Optional[str] = None

    # -- convenience --------------------------------------------------------
    @property
    def model_path(self) -> Path:
        if self.resolved_model_path:
            return Path(self.resolved_model_path)
        return resolve_model_path(self.model.model_path)

    def with_overrides(self, **kwargs) -> "AppConfig":
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

class ModelNotFoundError(FileNotFoundError):
    """Raised when no usable Moondream installation can be located."""


def resolve_model_path(explicit: Optional[str] = None) -> Path:
    """Locate the *already installed* Moondream 2 checkpoint.

    Order: explicit argument -> ``MODEL_PATH`` env var -> well known folders.
    Raises :class:`ModelNotFoundError` with actionable instructions instead of
    downloading anything.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(MODEL_SEARCH_CANDIDATES)

    for candidate in candidates:
        if _is_model_dir(candidate):
            return candidate.resolve()

    searched = "\n".join(f"  - {c}" for c in candidates)
    raise ModelNotFoundError(
        "Moondream 2 weights could not be found. No download was attempted.\n"
        f"Searched:\n{searched}\n"
        "Set MODEL_PATH to the folder containing config.json and "
        "model.safetensors, e.g.\n"
        '  export MODEL_PATH="/path/to/models/moondream2"\n'
        "or add MODEL_PATH=... to moondream-live-vqa/.env"
    )


def _is_model_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    if not (path / "config.json").is_file():
        return False
    return any(
        (path / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin", "model.pt")
    )


def discover_model() -> Optional[Path]:
    """Best-effort model lookup that never raises (used by status displays)."""
    try:
        return resolve_model_path()
    except ModelNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_config(
    *,
    model_path: Optional[str] = None,
    camera_index: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    fourcc: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    grounding: Optional[bool] = None,
    max_side: Optional[int] = None,
    queue_size: Optional[int] = None,
    log_level: Optional[str] = None,
    load_env: bool = True,
) -> AppConfig:
    """Build an :class:`AppConfig` from defaults, ``.env`` and CLI overrides."""
    if load_env:
        load_env_file()

    camera = CameraConfig(
        device_index=(
            camera_index
            if camera_index is not None
            else _env_int("VBMD_CAMERA_INDEX", 0)
        ),
        width=width if width is not None else _env_int("VBMD_CAMERA_WIDTH", 640),
        height=height if height is not None else _env_int("VBMD_CAMERA_HEIGHT", 480),
        fps=fps if fps is not None else _env_int("VBMD_CAMERA_FPS", 30),
        fourcc=(
            fourcc
            if fourcc is not None
            else _env("VBMD_CAMERA_FOURCC", "MJPG")
        ),
        backend=_env("VBMD_CAMERA_BACKEND", "auto"),
        auto_reconnect=_env_bool("VBMD_CAMERA_AUTO_RECONNECT", True),
    )

    model = ModelConfig(
        model_path=model_path if model_path is not None else os.environ.get("MODEL_PATH"),
        device=device if device is not None else _env("VQA_DEVICE", "auto"),
        dtype=dtype if dtype is not None else _env("VQA_DTYPE", "auto"),
        max_new_tokens=_env_int("VQA_MAX_NEW_TOKENS", 256),
        temperature=_env_float("VQA_TEMPERATURE", 0.0),
    )

    inference = InferenceConfig(
        queue_size=(
            queue_size if queue_size is not None else _env_int("VQA_QUEUE_SIZE", 1)
        ),
        grounding_prompt=(
            grounding if grounding is not None else _env_bool("VQA_GROUNDING", True)
        ),
        encode_max_side=(
            max_side
            if max_side is not None
            else (_env_int("VQA_MAX_SIDE", 0) or None)  # 0 -> disabled
        ),
        shutdown_timeout=_env_float("VQA_SHUTDOWN_TIMEOUT", 10.0),
    )

    log_file_env = os.environ.get("VQA_LOG_FILE")
    return AppConfig(
        camera=camera,
        model=model,
        inference=inference,
        log_level=(log_level or _env("VQA_LOG_LEVEL", "INFO")).upper(),
        log_file=log_file_env,
    )


def supported_resolutions() -> Sequence[tuple[int, int]]:
    """Common resolutions offered by the camera selector."""
    return ((320, 240), (640, 480), (960, 540), (1280, 720))
