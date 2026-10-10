#!/usr/bin/env python3
"""Verify the local Moondream 2 integration.

Loads the checkpoint once, runs one question against ``assets/sample_frame.jpeg``
(or an image you pass), and prints measured latency and VRAM usage.

    python3 scripts/verify_moondream.py [--image PATH] [--question Q] [--repeat N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.perception.moondream_engine import get_engine  # noqa: E402
from src.utils.device_manager import gpu_snapshot  # noqa: E402

DEFAULT_IMAGE = APP_ROOT.parent / "assets" / "sample_frame.jpeg"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--question", default="What objects are visible in this image?")
    ap.add_argument("--repeat", type=int, default=2, help="repeat to measure warm latency")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[FAIL] image not found: {image_path}")
        return 2

    if args.cpu:
        from config.settings import VLMSettings

        engine = get_engine(VLMSettings(device="cpu"))
    else:
        engine = get_engine()

    print(f"image    : {image_path}")
    print(f"question : {args.question}")

    t0 = time.perf_counter()
    try:
        info = engine.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] load: {type(exc).__name__}: {exc}")
        return 1
    print(f"loaded   : {info.describe()} in {info.load_seconds:.1f}s")
    print(f"api      : {info.api}")

    latencies: list[float] = []
    evidence = None
    for i in range(max(1, args.repeat)):
        evidence = engine.ask(str(image_path), args.question, frame_id=f"verify_{i}")
        latencies.append(evidence.inference_time_ms or 0.0)
        status = "ok" if evidence.usable else "FAIL"
        print(f"[{status}] run {i}: {evidence.inference_time_ms:.0f} ms -> {evidence.raw_answer!r}")
        if not evidence.usable:
            print(f"        error: {evidence.error}")
            for note in evidence.uncertainty_notes:
                print(f"        note : {note}")

    if evidence and evidence.usable:
        warm = latencies[-1]
        print(f"\nmeasured latency: cold/warm runs = {[round(x) for x in latencies]} ms")
        print(f"warm inference   : {warm:.0f} ms")
        gpu = gpu_snapshot().get("cuda", {})
        if gpu.get("vram_allocated_mb"):
            print(f"vram allocated   : {gpu['vram_allocated_mb']} MB")
        print("\n[EVIDENCE RECORD]")
        import json

        print(json.dumps(evidence.model_dump(), indent=2))
        return 0
    print("\n[FAIL] no usable observation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
