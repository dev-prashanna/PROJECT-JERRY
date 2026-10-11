#!/usr/bin/env python3
"""Benchmark PECM stages: durable write, keyword recall, hybrid search, RAG.

Usage::

    python3 scripts/benchmark_pecm.py                       # all scales
    python3 scripts/benchmark_pecm.py --scales 100 1000
    python3 scripts/benchmark_pecm.py --out reports/pecm_benchmark.json

Every scale (100 / 1k / 10k / 100k events) runs against a throwaway tmp
database.  Reports p50/p95/max per stage so regressions are easy to spot.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OBJECTS = ["curtain", "bottle", "cup", "laptop", "book", "chair", "lamp", "phone"]
COLOURS = ["white", "red", "blue", "green", "black", "yellow", "purple", "orange"]
PLACES = ["table", "desk", "shelf", "floor", "counter", "bed", "sofa", "chair"]


def _sample_turn(i: int) -> dict:
    obj = OBJECTS[i % len(OBJECTS)]
    colour = COLOURS[(i // 3) % len(COLOURS)]
    place = PLACES[(i // 7) % len(PLACES)]
    return {
        "transcript": f"What color is the {obj} on the {place}?",
        "observation": f"The {obj} appears {colour}.",
        "response": f"The {obj} appears {colour}.",
        "frame_id": f"frame_{i:06d}",
    }


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
        "n": n,
        "mean": statistics.fmean(ordered),
        "p50": q(0.50),
        "p95": q(0.95),
        "max": ordered[-1],
        "min": ordered[0],
    }


def bench(fn: Callable[[], object], runs: int) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)  # ms
    return pct(samples)


def run_scale(n_events: int, runs: int) -> dict:
    from config.settings import Settings, reset_settings_cache
    from src.memory.engine import MultimodalMemoryEngine
    from src.memory.indexing_worker import process_pending_jobs

    tmp = Path(tempfile.mkdtemp(prefix=f"pecm_bench_{n_events}_"))
    reset_settings_cache()
    s = Settings().with_overrides(
        **{
            "memory.enabled": True,
            "memory.db_path": str(tmp / "bench.db"),
            "memory.asset_dir": str(tmp / "assets"),
            "memory.face_recognition_enabled": False,
            "memory.object_memory_enabled": False,
            "memory.background_indexing": False,
            "memory.auto_learn_facts": False,
            "memory.conversation_memory_enabled": True,
            "memory.episodic_memory_enabled": True,
            "memory.memory_semantic_search_enabled": True,
            "memory.memory_text_embedder": "hash",
        }
    )
    eng = MultimodalMemoryEngine(s.memory)
    eng.initialize()

    # --- populate -------------------------------------------------------
    t0 = time.perf_counter()
    for i in range(n_events):
        t = _sample_turn(i)
        eng.record_turn(
            transcript=t["transcript"],
            observation=t["observation"],
            response=t["response"],
            frame_id=t["frame_id"],
            vlm_model="moondream2",
        )
    populate_s = time.perf_counter() - t0
    n_recorded = eng.store.count_events()

    # drain index jobs once so search benchmarks see a warm index
    process_pending_jobs(eng)

    # --- stages ---------------------------------------------------------
    results: dict = {
        "n_events": n_recorded,
        "populate_s": round(populate_s, 3),
        "populate_events_per_s": round(n_recorded / max(populate_s, 1e-9), 1),
    }

    last_turn = _sample_turn(n_events)
    results["record_turn"] = bench(
        lambda: eng.record_turn(
            transcript=last_turn["transcript"],
            observation=last_turn["observation"],
            response=last_turn["response"],
            frame_id=last_turn["frame_id"],
        ),
        runs,
    )

    results["recall_visual"] = bench(
        lambda: eng.recall_last_visual_observation(
            "what color was the curtain earlier?"
        ),
        runs,
    )

    results["search_fts"] = bench(
        lambda: eng.search_conversations("curtain color"),
        runs,
    )

    results["search_hybrid"] = bench(
        lambda: eng.search_conversations("what color was the blue bottle?"),
        runs,
    )

    results["rag_context"] = bench(
        lambda: eng.build_rag_context("what did you see on the table?"),
        runs,
    )

    results["recent_events"] = bench(
        lambda: eng.store.recent_events(limit=12),
        runs,
    )

    results["pending_jobs_after"] = eng.store.count_pending_index_jobs()
    results["database_bytes"] = eng.store.database_bytes()
    eng.shutdown()
    reset_settings_cache()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", type=int, nargs="+", default=[100, 1000, 10_000])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    report: dict = {"runs_per_stage": args.runs, "scales": {}}
    for n in args.scales:
        print(f"\n=== scale: {n:,} turns ({n * 3:,} events expected) ===")
        r = run_scale(n, args.runs)
        report["scales"][str(n)] = r
        print(f"  populate : {r['populate_events_per_s']:,} events/s "
              f"({r['populate_s']:.2f}s for {r['n_events']:,} events)")
        for stage in ("record_turn", "recall_visual", "search_fts",
                      "search_hybrid", "rag_context", "recent_events"):
            m = r.get(stage) or {}
            if m:
                print(f"  {stage:<16} p50={m['p50']:.3f}ms  p95={m['p95']:.3f}ms  max={m['max']:.3f}ms")
        print(f"  db size   : {r['database_bytes'] / 1024:.1f} KiB   "
              f"pending jobs: {r['pending_jobs_after']}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
