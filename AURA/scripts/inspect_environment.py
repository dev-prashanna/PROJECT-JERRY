#!/usr/bin/env python3
"""AURA environment inspector.

Prints a human-readable report of hardware, models, devices and cloud
configuration.  Secrets are redacted: only a masked key prefix/length is
ever shown, never the value.

Usage:
    python3 scripts/inspect_environment.py [--json] [--probe-groq]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import ENV_FILE, get_settings  # noqa: E402
from src.utils import device_manager  # noqa: E402


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _ok(flag: bool) -> str:
    return "OK  " if flag else "MISS"


def collect(probe_groq: bool = False) -> dict:
    from config import model_registry

    settings = get_settings()
    report: dict = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": os.path.realpath(sys.executable),
        },
        "env_file": {"path": str(ENV_FILE), "exists": ENV_FILE.is_file()},
        "hardware": device_manager.summarize(),
        "models": [],
        "groq": {},
        "problems": [],
        "warnings": [],
    }

    # --- models ---------------------------------------------------------
    for desc in model_registry.all_models():
        report["models"].append(
            {
                "key": desc.key,
                "kind": desc.kind,
                "backend": desc.backend,
                "installed": desc.installed,
                "location": desc.location,
                "detail": desc.detail,
                "notes": desc.notes,
            }
        )
        if not desc.installed and desc.key != "whisper":
            report["problems"].append(f"{desc.key} not found at {desc.location or '(unset)'}")

    # --- third-party imports -------------------------------------------
    imports = {}
    for mod in (
        "torch", "transformers", "faster_whisper", "kokoro", "groq",
        "cv2", "sounddevice", "soundfile", "PySide6", "pydantic", "numpy",
    ):
        try:
            __import__(mod)
            imports[mod] = True
        except Exception as exc:
            imports[mod] = f"{type(exc).__name__}: {exc}"
    report["imports"] = imports
    for mod, state in imports.items():
        if state is not True and mod in {"torch", "transformers", "faster_whisper", "kokoro", "groq", "cv2"}:
            report["problems"].append(f"python module '{mod}' unavailable: {state}")

    # --- groq -----------------------------------------------------------
    g = settings.llm
    masked = ""
    if g.api_key:
        masked = f"{g.api_key[:4]}… (len={len(g.api_key)})"
    report["groq"] = {
        "api_key_configured": bool(g.api_key),
        "api_key_masked": masked,
        "model": g.model,
        "fallback_model": g.fallback_model,
        "base_url": g.base_url,
        "reachable": None,
        "accessible_models": [],
    }
    if probe_groq:
        report["groq"].update(probe_groq_api(g))

    # --- derived warnings ----------------------------------------------
    gpu = report["hardware"]["gpu"]["cuda"]
    if gpu.get("available") and not gpu.get("cuda_available"):
        report["warnings"].append("torch built without CUDA – local models will run on CPU")
    vram_free = gpu.get("vram_free_mb")
    if vram_free is not None and vram_free < 4000:
        report["warnings"].append(
            f"only {vram_free} MB VRAM free: consider STT_DEVICE=cpu or a smaller STT model"
        )
    if not report["hardware"]["audio"]["inputs"]:
        report["problems"].append("no microphone input device detected")
    if not report["hardware"]["audio"]["outputs"]:
        report["problems"].append("no audio output device detected")
    if not report["hardware"]["cameras"]:
        report["warnings"].append("no camera detected (visual questions will be refused)")
    if not g.api_key:
        report["warnings"].append("GROQ_API_KEY not set – AURA will run in offline degraded mode")
    return report


def probe_groq_api(g) -> dict:
    result = {"reachable": False, "accessible_models": [], "error": None}
    if not g.api_key:
        result["error"] = "no API key"
        return result
    try:
        from groq import Groq

        client = Groq(api_key=g.api_key, base_url=g.base_url, timeout=g.timeout_s)
        listing = client.models.list()
        ids = sorted(m.id for m in listing.data)
        chat_ids = [i for i in ids if "whisper" not in i and "guard" not in i]
        result.update(reachable=True, accessible_models=chat_ids)
        preferred = [m for m in (g.model, g.fallback_model) if m in ids]
        result["configured_models_ok"] = preferred
        missing = [m for m in (g.model, g.fallback_model) if m not in ids]
        if missing:
            result["warning"] = f"configured model(s) not accessible: {missing}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return result


def print_report(r: dict) -> None:
    _section("AURA ENVIRONMENT REPORT")
    p = r["platform"]
    print(f"system      : {p['system']} {p['release']} ({p['machine']})")
    print(f"python      : {p['python']}  {p['python_executable']}")
    print(f".env        : {r['env_file']['path']} ({'present' if r['env_file']['exists'] else 'missing'})")

    _section("HARDWARE")
    hw = r["hardware"]
    gpu = hw["gpu"]
    cuda = gpu.get("cuda", {})
    print(f"torch       : {cuda.get('version', 'n/a')}  cuda={cuda.get('cuda_available')}")
    if cuda.get("device_name"):
        print(f"gpu         : {cuda['device_name']}")
    nv = gpu.get("nvsmi") or {}
    if nv:
        print(
            f"gpu usage   : {nv.get('util_pct')}% util, "
            f"{nv.get('mem_used_mb')}/{nv.get('mem_total_mb')} MB, {nv.get('temp_c')}C"
        )
    if "vram_free_mb" in cuda:
        print(f"vram free   : {cuda['vram_free_mb']} / {cuda.get('vram_total_mb')} MB")
    ram = hw.get("ram", {})
    print(f"ram         : {ram.get('available_mb')} MB available / {ram.get('total_mb')} MB total")

    _section("AUDIO DEVICES")
    for d in hw["audio"]["inputs"]:
        print(f"  IN  [{d['index']}] {d['name']}  {d['detail']}")
    for d in hw["audio"]["outputs"]:
        print(f"  OUT [{d['index']}] {d['name']}  {d['detail']}")
    print(f"  players   : {', '.join(hw['audio']['player_cli']) or 'none'}")

    _section("CAMERAS")
    if hw["cameras"]:
        for c in hw["cameras"]:
            print(f"  [{c['index']}] {c['name']}  {c['detail']}")
    else:
        print("  none detected")

    _section("MODELS")
    for m in r["models"]:
        print(f"  {_ok(m['installed'])} {m['kind']:<5} {m['key']:<12} {m['detail']}")
        if m["location"]:
            print(f"        location: {m['location']}")
        if m["notes"]:
            print(f"        notes   : {m['notes']}")

    _section("PYTHON PACKAGES")
    for mod, state in r["imports"].items():
        print(f"  {_ok(state is True)} {mod:<18} {'' if state is True else state}")

    _section("GROQCLOUD")
    g = r["groq"]
    print(f"  api key    : {'configured ' + g['api_key_masked'] if g['api_key_configured'] else 'NOT SET'}")
    print(f"  base url   : {g['base_url']}")
    print(f"  model      : {g['model']}")
    print(f"  fallback   : {g['fallback_model']}")
    if g.get("reachable") is True:
        print(f"  reachable  : yes ({len(g['accessible_models'])} chat models)")
        for m in g["accessible_models"]:
            mark = "*" if m in (g["model"], g["fallback_model"]) else " "
            print(f"     {mark} {m}")
    elif g.get("reachable") is False:
        print(f"  reachable  : NO – {g.get('error')}")
    else:
        print("  reachable  : not probed (use --probe-groq)")

    if r["problems"]:
        _section("PROBLEMS")
        for x in r["problems"]:
            print(f"  [!] {x}")
    if r["warnings"]:
        _section("WARNINGS")
        for x in r["warnings"]:
            print(f"  [-] {x}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--probe-groq", action="store_true", help="query Groq for reachable models")
    ap.add_argument("--save", action="store_true", help="write reports/environment_report.txt")
    args = ap.parse_args()

    report = collect(probe_groq=args.probe_groq)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    if args.save:
        out = APP_ROOT / "reports" / "environment_report.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(report)
        out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"saved {out}")

    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
