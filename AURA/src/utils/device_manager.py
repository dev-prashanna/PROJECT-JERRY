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


def ensure_ct2_cuda_libs() -> bool:
    """Preload the CUDA 12 runtime libraries CTranslate2 (faster-whisper) needs.

    This machine ships a CUDA 13 PyTorch build, while ctranslate2 links
    against ``libcublas.so.12``.  The cu12 libraries are installed by pip
    (``nvidia-cublas-cu12``) but are not on the dynamic loader's search path.
    Loading them here with an absolute path makes later ``dlopen`` calls
    succeed without requiring ``LD_LIBRARY_PATH`` to be set by hand.
    """
    import ctypes
    import glob
    import site
    from pathlib import Path

    try:
        ctypes.CDLL("libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        pass

    roots = [site.getusersitepackages()] + list(site.getsitepackages())
    # cudart must come before cublas (dependency order).
    wanted = ("libcudart.so.12", "libcublas.so.12", "libcufft.so.11", "libcurand.so.10")
    loaded_any = False
    for soname in wanted:
        for root in roots:
            base = Path(root) / "nvidia"
            if not base.is_dir():
                continue
            for match in sorted(glob.glob(str(base / "**" / soname), recursive=True)):
                try:
                    ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)
                    loaded_any = True
                    break
                except OSError:
                    continue
    if loaded_any:
        log.debug("preloaded CUDA 12 libraries for CTranslate2")
    return loaded_any


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
