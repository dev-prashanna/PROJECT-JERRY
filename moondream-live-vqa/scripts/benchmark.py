#!/usr/bin/env python3
"""Phase 6 — latency and memory benchmark for the local Moondream pipeline.

Measures (with proper CUDA synchronisation):

* model load time and device/dtype in use
* image preprocessing (BGR -> PIL)
* vision encoding (cold cache)
* answer generation (cold vs. frozen-frame reuse)
* end-to-end question latency per image size
* camera capture FPS and snapshot cost
* CUDA memory allocated / reserved

Usage::

    python scripts/benchmark.py                       # defaults, CUDA
    python scripts/benchmark.py --runs 5 --save
    python scripts/benchmark.py --device cpu --runs 1 --no-camera
    python scripts/benchmark.py --max-side 640          # resize experiment
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKSPACE_ROOT = ROOT.parent
DEFAULT_IMAGE = WORKSPACE_ROOT / "assets" / "sample_frame.jpeg"


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def _ms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    return f"{seconds * 1000:.0f} ms"


def _sec(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    return f"{seconds:.2f} s"


def build_image_variants(image, max_side: Optional[int]):
    from src.vlm.preprocessing import resize_max_side

    variants = [("native", image)]
    for label, size in (("640x480", (640, 480)), ("1280x720", (1280, 720))):
        variants.append((label, image.resize(size)))
    if max_side:
        variants.append((f"max_side={max_side}", resize_max_side(image, max_side)))
    return variants


def bench_model(
    config, image, runs: int, questions: List[str], max_side: Optional[int] = None
) -> Dict[str, Any]:
    from src.vlm.moondream_engine import MoondreamEngine
    from src.utils.performance import gpu_memory_mb

    engine = MoondreamEngine(config.model, config.inference)
    t0 = time.perf_counter()
    info = engine.load()
    load_seconds = time.perf_counter() - t0

    report: Dict[str, Any] = {
        "load_seconds": load_seconds,
        "model": {
            "path": info.model_path,
            "class": info.model_class,
            "parameters": info.parameter_count,
            "device": info.device,
            "dtype": info.dtype,
            "api": info.api,
            "gpu": info.gpu_name,
        },
        "variants": [],
    }

    for label, img in build_image_variants(image, max_side):
        cold: Dict[str, List[float]] = {
            "preprocess": [],
            "encode": [],
            "generate": [],
            "total": [],
        }
        answers: List[str] = []
        for i in range(runs):
            engine.clear_image_cache()
            result = engine.ask(
                questions[i % len(questions)],
                image=img,
                frame_id=i,
                cache_key=f"bench-{label}-{i}",
            )
            if result["status"] != "success":
                raise SystemExit(f"benchmark failed: {result.get('error')}")
            cold["preprocess"].append(result["preprocess_time_seconds"])
            cold["encode"].append(result["encode_time_seconds"])
            cold["generate"].append(result["generate_time_seconds"])
            cold["total"].append(result["inference_time_seconds"])
            answers.append(result["answer"])

        # Frozen-frame behaviour: one encode, then repeated questions.
        engine.clear_image_cache()
        warm_totals: List[float] = []
        warm_encodes: List[float] = []
        reused = 0
        for i in range(max(1, runs)):
            result = engine.ask(
                questions[i % len(questions)],
                image=img,
                frame_id=0,
                cache_key=f"frozen-{label}",
            )
            warm_totals.append(result["inference_time_seconds"])
            warm_encodes.append(result["encode_time_seconds"])
            if result["encoding_reused"]:
                reused += 1

        variant_report = {
            "label": label,
            "size": list(img.size),
            "cold": {k: _stats(v) for k, v in cold.items()},
            "frozen": {
                "total": _stats(warm_totals),
                "encode": _stats(warm_encodes),
                "reused_count": reused,
                "runs": len(warm_totals),
            },
            "sample_answer": answers[0] if answers else "",
        }
        report["variants"].append(variant_report)

        print(f"\n--- image variant: {label} ({img.size[0]}x{img.size[1]}) ---")
        print(
            f"  cold   : total={_ms(variant_report['cold']['total'].get('mean'))} "
            f"(pre={_ms(variant_report['cold']['preprocess'].get('mean'))}, "
            f"enc={_ms(variant_report['cold']['encode'].get('mean'))}, "
            f"gen={_ms(variant_report['cold']['generate'].get('mean'))}) "
            f"over {runs} runs"
        )
        print(
            f"  frozen : total={_ms(variant_report['frozen']['total'].get('mean'))} "
            f"enc={_ms(variant_report['frozen']['encode'].get('mean'))} "
            f"(reused {reused}/{len(warm_totals)})"
        )
        print(f"  answer : {answers[0][:100] if answers else ''}")

    report["gpu_memory_mb"] = gpu_memory_mb(engine.device)
    if report["gpu_memory_mb"]:
        mem = report["gpu_memory_mb"]
        print(
            f"\nCUDA memory: allocated={mem['allocated_mb']:.0f} MiB "
            f"reserved={mem['reserved_mb']:.0f} MiB "
            f"total={mem['total_mb']:.0f} MiB"
        )
    engine.unload()
    return report


def bench_camera(config, seconds: float = 4.0) -> Dict[str, Any]:
    from src.camera.camera import CameraSource

    camera = CameraSource(config.camera)
    camera.start()
    deadline = time.time() + 5
    while camera.get_latest_frame() is None and time.time() < deadline:
        time.sleep(0.05)
    if camera.get_latest_frame() is None:
        camera.stop()
        return {"available": False, "reason": "no frame"}

    time.sleep(seconds)
    snap = camera.get_latest_frame()
    fps = camera.actual_fps

    # Snapshot cost (the copy the GUI/inference performs each time).
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        camera.get_latest_frame(copy=True)
        samples.append(time.perf_counter() - t0)

    info = {
        "available": True,
        "resolution": list(camera.resolution),
        "measured_fps": round(fps, 2),
        "frames_read": camera.frames_read,
        "backend": camera.backend_name,
        "fourcc": config.camera.fourcc,
        "snapshot_copy_ms": _stats([s * 1000 for s in samples]),
        "latest_frame": snap.describe() if snap else None,
    }
    camera.stop()
    print(
        f"\nCamera: {info['resolution'][0]}x{info['resolution'][1]} @ "
        f"{fps:.1f} fps ({info['backend']}, {info['fourcc']}), "
        f"snapshot copy = {info['snapshot_copy_ms']['mean']:.2f} ms"
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default=None, choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--camera-seconds", type=float, default=4.0)
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument(
        "--save",
        action="store_true",
        help="write outputs/benchmark_<timestamp>.json",
    )
    args = parser.parse_args()

    from config.settings import build_config
    from src.utils.logging_utils import configure_logging

    configure_logging(args.log_level)
    # Image-size experiments are done by resizing the input explicitly, so the
    # engine-side cap is left disabled here.
    config = build_config(device=args.device, dtype=args.dtype, load_env=True)

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    questions = [
        "What objects are visible in this image?",
        "What is the color of the main object?",
        "Describe the scene in one sentence.",
        "Is there any readable text?",
        "What is the lighting like?",
    ]

    print("=" * 78)
    print("Moondream Live VQA — benchmark")
    print(f"time        : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"python      : {platform.python_version()} on {platform.platform()}")
    print(f"image       : {image_path} ({image.size[0]}x{image.size[1]})")
    print(f"runs        : {args.runs}")
    print(f"max_side    : {args.max_side} (variant, engine cap disabled)")
    print("=" * 78)

    import torch

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "image": str(image_path),
        "runs": args.runs,
        "max_side": args.max_side,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 2),
        }
        print(f"gpu         : {props.name} ({props.total_memory / 1e9:.1f} GB)")

    report["model"] = bench_model(config, image, args.runs, questions, args.max_side)

    if not args.no_camera:
        report["camera"] = bench_camera(config, args.camera_seconds)

    if args.save:
        out = ROOT / "outputs" / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
