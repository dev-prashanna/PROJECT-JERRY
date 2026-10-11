"""Hardware / device discovery and GPU memory reporting."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.logger import get_logger

log = get_logger("devices")


@dataclass(frozen=True)
class DeviceInfo:
    kind: str
    index: int
    name: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Torch / CUDA
# ---------------------------------------------------------------------------

def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        info["error"] = str(exc)
        return info
    info["available"] = True
    info["version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
        try:
            free, total = torch.cuda.mem_get_info(0)
            info["vram_total_mb"] = round(total / 1024**2, 1)
            info["vram_free_mb"] = round(free / 1024**2, 1)
        except Exception:
            pass
        try:
            info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2, 1)
            info["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1024**2, 1)
        except Exception:
            pass
    return info


def gpu_snapshot() -> dict[str, Any]:
    """Small, cheap GPU status suitable for a GUI timer."""
    return {
        "cuda": torch_info(),
        "nvsmi": _nvsmi(),
    }


def _nvsmi() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return {}
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) >= 4:
            return {
                "util_pct": int(parts[0]),
                "mem_used_mb": int(parts[1]),
                "mem_total_mb": int(parts[2]),
                "temp_c": int(parts[3]),
            }
    except Exception:
        return {}
    return {}


def resolve_torch_device(preference: str = "auto") -> str:
    """Map a settings value to a concrete torch device string."""
    pref = (preference or "auto").lower()
    if pref in {"cuda", "gpu"}:
        return "cuda"
    if pref == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


#: Libraries CTranslate2's CUDA backend ``dlopen()``s at inference time.
#: Without *both* of these, ``WhisperModel(device="cuda")`` loads fine but the
#: very first encode() raises ``Library libcublas.so.12 is not found``.
CT2_REQUIRED_LIBS: tuple[str, ...] = ("libcudart.so.12", "libcublas.so.12")
#: Nice to have; a failure here is not fatal for transcription.
CT2_OPTIONAL_LIBS: tuple[str, ...] = ("libcufft.so.11", "libcurand.so.10", "libcusparse.so.12")


def _ct2_library_dirs() -> list:
    """Directories that plausibly contain the CUDA 12 runtime libraries."""
    import site
    import sys
    from pathlib import Path

    dirs: list[Path] = []

    # pip wheels ship them under site-packages/nvidia/<component>/lib
    for root in [site.getusersitepackages()] + list(site.getsitepackages()):
        base = Path(root) / "nvidia"
        if base.is_dir():
            dirs.extend(p for p in base.glob("*/lib") if p.is_dir())

    prefix = Path(sys.prefix)
    dirs.append(prefix / "lib")
    # conda envs live at <root>/envs/<name>, so the root env's lib is two levels up.
    dirs.append(prefix.parent.parent / "lib")

    for var in ("LD_LIBRARY_PATH", "CUDA_PATH", "CUDA_HOME"):
        for entry in (os.environ.get(var) or "").split(":"):
            if entry:
                dirs.append(Path(entry))

    dirs.extend(
        Path(p)
        for p in ("/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib")
    )

    seen: set[str] = set()
    ordered: list[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen and d.is_dir():
            seen.add(key)
            ordered.append(d)
    return ordered


def _load_soname(soname: str, dirs: list) -> bool:
    """dlopen ``soname`` (by name, then by absolute path) with RTLD_GLOBAL."""
    import ctypes
    from pathlib import Path

    try:
        ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        pass
    for d in dirs:
        for match in sorted(d.glob(soname + "*")):
            if not match.is_file() and not match.is_symlink():
                continue
            try:
                ctypes.CDLL(str(match), mode=ctypes.RTLD_GLOBAL)
                log.debug("preloaded %s from %s", soname, match)
                return True
            except OSError:
                continue
    return False


def ensure_ct2_cuda_libs() -> bool:
    """Preload the CUDA 12 runtime libraries CTranslate2 (faster-whisper) needs.

    This machine ships a CUDA 13 PyTorch build, while ctranslate2 4.x links
    against ``libcublas.so.12`` / ``libcudart.so.12``.  Those libraries are
    usually on disk but not on the dynamic loader's search path, so we
    ``dlopen`` them with an absolute path (``RTLD_GLOBAL``) here; later
    ``dlopen("libcublas.so.12")`` calls then resolve to the already-loaded
    object instead of failing.

    Returns ``True`` only when **every required** library is loaded — a
    partial preload (say, ``libcurand`` only) would make the model load on
    CUDA and then crash on the first transcription.
    """
    dirs = _ct2_library_dirs()

    # Order matters: cudart must come before cublas (dependency order).
    missing = [s for s in CT2_REQUIRED_LIBS if not _load_soname(s, dirs)]
    if missing:
        log.warning(
            "CUDA 12 libraries required by CTranslate2 not found (%s); "
            "whisper will run on CPU",
            ", ".join(missing),
        )
        return False

    for soname in CT2_OPTIONAL_LIBS:
        _load_soname(soname, dirs)
    log.debug("CUDA 12 libraries for CTranslate2 preloaded")
    return True


# ---------------------------------------------------------------------------
# Audio devices
# ---------------------------------------------------------------------------

def audio_devices() -> tuple[list[DeviceInfo], list[DeviceInfo]]:
    """Return (inputs, outputs)."""
    inputs: list[DeviceInfo] = []
    outputs: list[DeviceInfo] = []
    try:
        import sounddevice as sd
    except Exception as exc:
        log.warning("sounddevice unavailable: %s", exc)
        return inputs, outputs
    try:
        devices = sd.query_devices()
        defaults = sd.default.device
    except Exception as exc:
        log.warning("cannot query audio devices: %s", exc)
        return inputs, outputs
    for i, d in enumerate(devices):
        info = DeviceInfo(
            kind="audio",
            index=i,
            name=str(d.get("name", f"device {i}")),
            detail=f"{int(d.get('default_samplerate', 0))} Hz, "
            f"in={d.get('max_input_channels', 0)} out={d.get('max_output_channels', 0)}",
        )
        if d.get("max_input_channels", 0) > 0:
            inputs.append(info)
        if d.get("max_output_channels", 0) > 0:
            outputs.append(info)
        if isinstance(defaults, (list, tuple)) and len(defaults) == 2:
            if i == defaults[0]:
                inputs = [x for x in inputs if x.index != i] + [info]
            if i == defaults[1]:
                outputs = [x for x in outputs if x.index != i] + [info]
    return inputs, outputs


def resolve_audio_device(spec: str, kind: str) -> Optional[int]:
    """Resolve ``default`` / ``pipewire`` / name / index to a device index."""
    try:
        import sounddevice as sd
    except Exception:
        return None
    devices = sd.query_devices()
    if kind == "input":
        candidates = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    else:
        candidates = [i for i, d in enumerate(devices) if d["max_output_channels"] > 0]
    if not candidates:
        return None
    spec = (spec or "").strip()
    if spec == "" or spec.lower() == "default":
        default = sd.default.device
        idx = default[0] if kind == "input" else default[1]
        if isinstance(idx, int) and idx >= 0 and idx in candidates:
            return idx
        return candidates[0]
    if spec.isdigit():
        idx = int(spec)
        return idx if idx in candidates else None
    lowered = spec.lower()
    for i in candidates:
        if lowered in str(devices[i]["name"]).lower():
            return i
    return None


def list_cameras(probe: bool = True) -> list[DeviceInfo]:
    """Discover candidate camera indices; optionally probe them."""
    found: list[DeviceInfo] = []
    try:
        import cv2
    except Exception as exc:
        log.warning("OpenCV unavailable: %s", exc)
        return found
    for index in range(4):
        path = f"/dev/video{index}"
        exists = os.path.exists(path)
        if not probe and not exists:
            continue
        ok, detail = False, path if exists else "no device node"
        if probe:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ok = True
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                detail = f"{w}x{h} @ {fps:.0f} fps"
            cap.release()
        if ok or exists:
            found.append(DeviceInfo(kind="camera", index=index, name=path, detail=detail))
    return found


def speaker_available() -> bool:
    return bool(shutil.which("aplay") or shutil.which("paplay") or shutil.which("ffplay"))


def summarize() -> dict[str, Any]:
    """Full environment summary used by scripts/inspect_environment.py."""
    inputs, outputs = audio_devices()
    return {
        "python": {
            "version": _python_version(),
            "executable": os.path.realpath(os.sys.executable),
        },
        "gpu": gpu_snapshot(),
        "ram": _ram(),
        "audio": {
            "inputs": [d.__dict__ for d in inputs],
            "outputs": [d.__dict__ for d in outputs],
            "player_cli": [c for c in ("aplay", "paplay", "ffplay") if shutil.which(c)],
        },
        "cameras": [d.__dict__ for d in list_cameras()],
        "tools": {c: shutil.which(c) for c in ("ffmpeg", "arecord", "aplay", "nvidia-smi")},
    }


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _ram() -> dict[str, Any]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info = {}
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    info[key.strip()] = int(parts[0])  # kB
        return {
            "total_mb": round(info.get("MemTotal", 0) / 1024),
            "available_mb": round(info.get("MemAvailable", 0) / 1024),
        }
    except Exception:
        return {}
