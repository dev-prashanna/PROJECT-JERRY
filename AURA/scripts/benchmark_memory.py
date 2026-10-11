#!/usr/bin/env python3
"""Benchmark the Multimodal Memory System stages in isolation.

Usage::

    python3 scripts/benchmark_memory.py                  # all stages, 20 runs each
    python3 scripts/benchmark_memory.py --runs 50
    python3 scripts/benchmark_memory.py --stages retrieve fact_store
    python3 scripts/benchmark_memory.py --out reports/memory_benchmark.json

Stages (all run against a throwaway tmp database — your real memory is never
touched):

* ``fact_extract`` – deterministic statement classification + fact extraction
* ``fact_store``   – remember_fact round trip (extract + resolve + insert)
* ``retrieve``     – text query → MemoryContext + prompt block
* ``face``         – detect + embed one synthetic frame (real backend if the
                     OpenCV models are available, otherwise a stub)
* ``object``       – grounder + segmenter + sift-color embedding of a frame

Every stage is skipped gracefully (with a recorded reason) when a dependency
is unavailable, so the script always produces a report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger, setup_logging  # noqa: E402

log = get_logger("memory_benchmark")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pct(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        if n == 1:
            return ordered[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return ordered[idx]

    return {
        "min_ms": round(ordered[0] * 1000, 3),
        "p50_ms": round(q(0.50) * 1000, 3),
        "p95_ms": round(q(0.95) * 1000, 3),
        "max_ms": round(ordered[-1] * 1000, 3),
        "mean_ms": round(statistics.fmean(ordered) * 1000, 3),
        "runs": n,
    }


def time_fn(fn: Callable[[], Any], runs: int, warmup: int = 2) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return pct(samples)


# ---------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------

def synth_frame(seed: int = 0, size=(120, 160)) -> np.ndarray:
    h, w = size
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[h // 4 : h // 2, w // 3 : 2 * w // 3] = (0, 180, 220)
    img += rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


TRANSCRIPTS = [
    "My robotics project is called NATERIDA",
    "Please remember that my favourite editor is Neovim",
    "I built a custom ESP32 motor controller last weekend",
    "My name is Alex",
    "Actually the project is now called NATERIDA-X",
    "If I were rich I would buy a bigger camera",   # hypothetical → no fact
    "What is my name?",                             # question → no fact
]


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_fact_extract(runs: int) -> dict[str, Any]:
    from src.memory.fact_extractor import extract_deterministic

    cycle = {"i": 0}

    def one():
        text = TRANSCRIPTS[cycle["i"] % len(TRANSCRIPTS)]
        cycle["i"] += 1
        return extract_deterministic(text)

    result = time_fn(one, runs)
    result["stage"] = "fact_extract"
    return result


def stage_fact_store(tmp: Path, runs: int) -> dict[str, Any]:
    from src.memory.engine import MultimodalMemoryEngine
    from src.memory.models import RememberFactRequest

    try:
        from config.settings import Settings as AuraSettings

        s = AuraSettings().with_overrides(
            **{
                "memory.enabled": True,
                "memory.db_path": str(tmp / "bench.db"),
                "memory.asset_dir": str(tmp / "assets"),
                "memory.face_recognition_enabled": False,
                "memory.object_memory_enabled": False,
                "memory.background_indexing": False,
                "memory.auto_learn_facts": False,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return {"stage": "fact_store", "skipped": str(exc)}

    engine = MultimodalMemoryEngine(s.memory)
    engine.initialize()
    try:
        cycle = {"i": 0}

        def one():
            text = TRANSCRIPTS[cycle["i"] % len(TRANSCRIPTS)]
            cycle["i"] += 1
            res = engine.remember_fact(
                RememberFactRequest(transcript=text, turn_id=f"b{cycle['i']}")
            )
            assert res.ok, res.message

        result = time_fn(one, runs, warmup=1)
        result["stage"] = "fact_store"
        return result
    finally:
        engine.shutdown()


def stage_retrieve(tmp: Path, runs: int) -> dict[str, Any]:
    from config.settings import Settings as AuraSettings
    from src.memory.engine import MultimodalMemoryEngine
    from src.memory.models import RememberFactRequest

    s = AuraSettings().with_overrides(
        **{
            "memory.enabled": True,
            "memory.db_path": str(tmp / "bench_retrieve.db"),
            "memory.asset_dir": str(tmp / "assets"),
            "memory.face_recognition_enabled": False,
            "memory.object_memory_enabled": False,
            "memory.background_indexing": False,
            "memory.auto_learn_facts": False,
        }
    )
    engine = MultimodalMemoryEngine(s.memory)
    engine.initialize()
    try:
        for i, text in enumerate(TRANSCRIPTS * 5):
            engine.remember_fact(RememberFactRequest(transcript=text, turn_id=f"r{i}"))

        queries = [
            "What did I tell you my robotics project was called?",
            "What is my name?",
            "What editor do I use?",
        ]
        cycle = {"i": 0}

        def one():
            q = queries[cycle["i"] % len(queries)]
            cycle["i"] += 1
            ctx = engine.retrieve_context(q)
            return engine.build_context_block(ctx, request_text=q)

        result = time_fn(one, runs, warmup=1)
        result["stage"] = "retrieve"
        result["facts_in_db"] = len(TRANSCRIPTS) * 5
        return result
    finally:
        engine.shutdown()


def stage_face(runs: int) -> dict[str, Any]:
    from src.memory.face_memory import create_face_backend

    frame = synth_frame()
    try:
        backend = create_face_backend("opencv", allow_download=True)
        if not backend.available():
            raise RuntimeError("opencv face backend unavailable")
        real = True
    except Exception as exc:  # noqa: BLE001
        log.info("real face backend unavailable (%s); using null stub", exc)

        class _Stub:
            name = "stub"

            def available(self) -> bool:
                return True

            def detect(self, image):
                from src.memory.face_memory import DetectedFace

                h, w = image.shape[:2]
                return [DetectedFace(bbox_xyxy=(w // 3, h // 4, 2 * w // 3, h // 2), confidence=1.0)]

            def embed(self, image, face):
                return np.ones(128, dtype=np.float32) / 128.0

        backend = _Stub()
        real = False

    def one():
        faces = backend.detect(frame)
        if faces:
            backend.embed(frame, faces[0])

    result = time_fn(one, runs)
    result["stage"] = "face"
    result["backend"] = getattr(backend, "name", "?")
    result["real_models"] = real
    return result


def stage_object(runs: int) -> dict[str, Any]:
    from src.memory.object_isolation import (
        BoundingBoxFallbackSegmenter,
        CenterPriorGrounder,
    )
    from src.memory.visual_embeddings import SiftColorEmbedder

    frame = synth_frame(seed=2)
    grounder = CenterPriorGrounder()
    segmenter = BoundingBoxFallbackSegmenter()
    embedder = SiftColorEmbedder()

    def one():
        regions = grounder.locate(frame, "motor controller")
        crop = segmenter.segment(frame, regions[0].bbox_xyxy)
        return embedder.encode(crop)

    result = time_fn(one, runs)
    result["stage"] = "object"
    result["embedder"] = embedder.model_id
    result["dimension"] = embedder.embedding_dimension
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

STAGES = ("fact_extract", "fact_store", "retrieve", "face", "object")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging("DEBUG" if args.verbose else "WARNING")

    report: dict[str, Any] = {"runs": args.runs, "stages": []}
    with tempfile.TemporaryDirectory(prefix="aura_mem_bench_") as tmp_str:
        tmp = Path(tmp_str)
        for name in args.stages:
            print(f"… {name}")
            try:
                if name == "fact_extract":
                    r = stage_fact_extract(args.runs)
                elif name == "fact_store":
                    r = stage_fact_store(tmp, args.runs)
                elif name == "retrieve":
                    r = stage_retrieve(tmp, args.runs)
                elif name == "face":
                    r = stage_face(args.runs)
                elif name == "object":
                    r = stage_object(args.runs)
                else:  # pragma: no cover - argparse guards this
                    continue
            except Exception as exc:  # noqa: BLE001
                r = {"stage": name, "skipped": str(exc)}
            report["stages"].append(r)
            if "skipped" in r:
                print(f"  skipped: {r['skipped']}")
            else:
                print(
                    f"  p50={r['p50_ms']}ms  p95={r['p95_ms']}ms  "
                    f"max={r['max_ms']}ms  ({r['runs']} runs)"
                )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
