#!/usr/bin/env python3
"""Phase 1 — inspect the runtime environment before touching anything.

Reports Python, GPU/CUDA, installed packages, the local Moondream 2
checkpoint, camera devices and GUI toolkit availability.

Usage::

    python scripts/inspect_environment.py           # print report
    python scripts/inspect_environment.py --save    # also write docs/environment_report.txt
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pkg_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return str(getattr(mod, "__version__", "installed"))
    except Exception as exc:  # noqa: BLE001
        return f"NOT AVAILABLE ({type(exc).__name__})"


def _section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def inspect_python() -> None:
    _section("1. Python environment")
    print(f"executable   : {sys.executable}")
    print(f"version      : {sys.version.split()[0]} ({sys.version.splitlines()[0][-40:]})")
    print(f"platform     : {platform.platform()}")
    print(f"machine      : {platform.machine()}  cpus={os.cpu_count()}")
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    print(f"active venv  : {venv or 'none (system Python)'}")
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    print(f"in a venv    : {in_venv}")


def inspect_packages() -> None:
    _section("2. Key packages")
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
        "tokenizers",
        "cv2",
        "numpy",
        "PIL",
        "PySide6",
        "pytest",
        "dotenv",
        "einops",
        "pyvips",
    ):
        label = {"cv2": "opencv-python", "PIL": "Pillow"}.get(name, name)
        print(f"{label:<20}: {_pkg_version(name)}")


def inspect_torch_cuda() -> None:
    _section("3. PyTorch / CUDA")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"torch import failed: {exc}")
        return
    print(f"torch        : {torch.__version__}")
    print(f"cuda runtime : {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device count : {torch.cuda.device_count()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(
                f"gpu[{i}]       : {props.name} | {props.total_memory / 1e9:.1f} GB "
                f"| free {free / 1e9:.1f} GB | cc {props.major}.{props.minor}"
            )
        print(f"cudnn        : {torch.backends.cudnn.version()}")
    nvcc = shutil.which("nvcc")
    print(f"nvcc         : {nvcc or 'not on PATH (not required at runtime)'}")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            print(f"driver       : {out.stdout.strip()}")
        except Exception:  # noqa: BLE001
            pass


def inspect_model() -> None:
    _section("4. Local Moondream 2 installation")
    from config.settings import ModelNotFoundError, discover_model, resolve_model_path

    try:
        path = resolve_model_path(os.environ.get("MODEL_PATH"))
    except ModelNotFoundError as exc:
        print(str(exc))
        return
    print(f"model path   : {path}")
    total = 0
    for name in sorted(p.name for p in path.iterdir() if p.is_file()):
        size = (path / name).stat().st_size
        total += size
        if size > 100_000 or name in {"config.json", "versions.txt"}:
            print(f"  {name:<28} {size / 1e6:9.2f} MB")
    print(f"total size   : {total / 1e9:.2f} GB")

    import json

    cfg = json.loads((path / "config.json").read_text())
    print(f"architectures: {cfg.get('architectures')}")
    print(f"auto_map     : {cfg.get('auto_map')}")
    print(f"torch_dtype  : {cfg.get('torch_dtype')}")
    versions = path / "versions.txt"
    if versions.is_file():
        revs = [line.strip() for line in versions.read_text().splitlines() if line.strip()]
        print(f"revisions    : {len(revs)} listed, latest = {revs[-1] if revs else '?'}")
    print(f"discoverable : {discover_model() is not None}")

    # Does the installed transformers build accept this checkpoint structure?
    try:
        import transformers

        print(f"transformers : {transformers.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"transformers : FAILED ({exc})")


def inspect_cameras() -> None:
    _section("5. Camera devices")
    from src.camera.camera import discover_cameras

    nodes = sorted(Path("/dev").glob("video*"))
    print(f"/dev/video*  : {[n.name for n in nodes] or 'none'}")
    for node in nodes:
        name = Path(f"/sys/class/video4linux/{node.name}/name")
        if name.is_file():
            try:
                print(f"  {node.name}: {name.read_text().strip()}")
            except OSError:
                pass
    try:
        found = discover_cameras(max_index=6)
    except Exception as exc:  # noqa: BLE001
        print(f"discovery failed: {exc}")
        return
    if not found:
        print("No camera yielded a frame (device busy or permission denied).")
    for info in found:
        print(f"  FOUND {info.label()} path={info.path}")


def inspect_gui() -> None:
    _section("6. GUI / display")
    print(f"DISPLAY      : {os.environ.get('DISPLAY', '(unset)')}")
    print(f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY', '(unset)')}")
    print(f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', '(unset)')}")
    print(f"PySide6      : {_pkg_version('PySide6')}")
    try:
        from PySide6.QtWidgets import QApplication

        # Creates (and immediately destroys) a QApplication to verify a
        # platform plugin can be instantiated.
        import tempfile

        args = [sys.argv[0], "-platform", "offscreen"]
        app = QApplication(args)
        print(f"qt platform  : {app.platformName()} (offscreen probe OK)")
        del app
    except Exception as exc:  # noqa: BLE001
        print(f"qt probe     : FAILED ({type(exc).__name__}: {exc})")


def _run_report() -> None:
    print("Moondream live VQA — environment inspection")
    print(f"project root : {ROOT}")
    inspect_python()
    inspect_packages()
    inspect_torch_cuda()
    inspect_model()
    inspect_cameras()
    inspect_gui()


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:  # noqa: BLE001
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", action="store_true", help="write docs/environment_report.txt"
    )
    args = parser.parse_args()

    if args.save:
        import contextlib

        target = ROOT / "docs" / "environment_report.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            with contextlib.redirect_stdout(_Tee(sys.stdout, fh)):
                _run_report()
        print(f"\nSaved report -> {target}")
    else:
        _run_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
