#!/usr/bin/env python3
"""Phase 2 — standalone verification of the local Moondream 2 checkpoint.

Loads the already-installed weights, asks a question about a static image and
prints per-stage latency.  Run this before starting the GUI: if this fails,
the application cannot work either.

Usage::

    python scripts/test_model.py                       # default sample image
    python scripts/test_model.py --image /path/img.jpg
    python scripts/test_model.py --questions "What is this?" "Describe it"
    python scripts/test_model.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import WORKSPACE_ROOT, build_config  # noqa: E402
from src.utils.logging_utils import configure_logging, get_logger  # noqa: E402
from src.utils.performance import format_seconds  # noqa: E402
from src.vlm.moondream_engine import MoondreamEngine  # noqa: E402

log = get_logger("test_model")

DEFAULT_IMAGES = [
    WORKSPACE_ROOT / "assets" / "sample_frame.jpeg",
    WORKSPACE_ROOT / "realtime_depth_app" / "outputs" / "headless_last.png",
    ROOT / "assets" / "test_frame.png",
]

DEFAULT_QUESTIONS = [
    "What objects are visible in this image?",
    "What color is the most prominent object?",
    "Is there any readable text?",
]


def pick_image(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"image not found: {path}")
        return path
    for candidate in DEFAULT_IMAGES:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "No test image found. Pass --image /path/to/image.jpg\n"
        f"Tried: {', '.join(str(p) for p in DEFAULT_IMAGES)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="path to a test image")
    parser.add_argument(
        "--questions",
        nargs="*",
        default=None,
        help="questions to ask (defaults to three probing questions)",
    )
    parser.add_argument("--model-path", help="override MODEL_PATH")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--no-grounding", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="repeat each question N times")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = build_config(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        grounding=not args.no_grounding,
        load_env=True,
    )

    image_path = pick_image(args.image)
    questions = args.questions or DEFAULT_QUESTIONS

    print(f"image    : {image_path}")
    print(f"model    : {config.model_path}")
    print(f"grounding: {config.inference.grounding_prompt}")

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    print(f"size     : {image.size[0]}x{image.size[1]}")

    engine = MoondreamEngine(config.model, config.inference)
    t0 = time.perf_counter()
    try:
        info = engine.load()
    except Exception as exc:  # noqa: BLE001
        print(f"\nMODEL LOAD FAILED: {type(exc).__name__}: {exc}")
        return 2
    print(f"loaded   : {format_seconds(time.perf_counter() - t0)}")
    print(f"info     : {info.describe()}")
    print(f"api      : {info.api} | device={info.device} | dtype={info.dtype}")
    print("-" * 78)

    exit_code = 0
    for index in range(max(1, args.repeat)):
        for question in questions:
            result = engine.ask(
                question,
                image=image,
                frame_id=index,
                cache_key=None,  # always encode fresh: measures baseline
            )
            status = result["status"]
            if status != "success":
                exit_code = 1
            print(
                f"[{status}] Q: {question}\n"
                f"         A: {result['answer'] or result['error']}\n"
                f"         total={format_seconds(result['inference_time_seconds'])} "
                f"pre={format_seconds(result['preprocess_time_seconds'])} "
                f"enc={format_seconds(result['encode_time_seconds'])} "
                f"gen={format_seconds(result['generate_time_seconds'])} "
                f"device={result['device']}\n"
            )

    # Frozen-frame behaviour: the second question should reuse the encoding.
    if len(questions) > 1:
        print("-" * 78)
        print("Encoding-cache check (same image, cache_key='frame:test'):")
        first = engine.ask(
            questions[0], image=image, frame_id=0, cache_key="frame:test"
        )
        second = engine.ask(
            questions[-1], image=image, frame_id=0, cache_key="frame:test"
        )
        print(
            f"  first : enc={format_seconds(first['encode_time_seconds'])} "
            f"total={format_seconds(first['inference_time_seconds'])}"
        )
        print(
            f"  second: enc={format_seconds(second['encode_time_seconds'])} "
            f"total={format_seconds(second['inference_time_seconds'])} "
            f"reused={second['encoding_reused']}"
        )
        if not second["encoding_reused"]:
            print("  WARNING: encoding cache was not reused")
            exit_code = max(exit_code, 1)

    from src.utils.performance import gpu_memory_mb

    mem = gpu_memory_mb(engine.device)
    if mem:
        print(
            "\nCUDA memory: allocated={allocated_mb:.0f} MiB "
            "reserved={reserved_mb:.0f} MiB total={total_mb:.0f} MiB".format(**mem)
        )
    engine.unload()
    print("\nDone.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
