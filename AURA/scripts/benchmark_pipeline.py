#!/usr/bin/env python3
"""Benchmark every stage of the AURA pipeline and the end-to-end turns.

Usage::

    python3 scripts/benchmark_pipeline.py                 # all stages, 5 runs each
    python3 scripts/benchmark_pipeline.py --runs 10
    python3 scripts/benchmark_pipeline.py --stages vlm tts
    python3 scripts/benchmark_pipeline.py --out reports/benchmark.json

Stages:

* ``whisper``  – Kokoro-synthesised speech transcribed by Whisper (no mic needed)
* ``groq``     – cloud routing latency (one small request per run)
* ``vlm``      – Moondream-2 on a synthetic scene (cold + warm)
* ``tts``      – Kokoro synthesis + real-time-factor
* ``e2e_text`` – full text turn, general question (no camera)
* ``e2e_vision`` – full text turn, visual question (camera or synthetic frame)

Every stage is skipped gracefully (with a recorded reason) when its hardware
or model is unavailable, so the script always produces a report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from src.utils.logger import get_logger, setup_logging  # noqa: E402

log = get_logger("benchmark")


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
        "runs": n,
        "mean_ms": round(statistics.fmean(ordered) * 1000, 1),
        "p50_ms": round(q(0.50) * 1000, 1),
        "p95_ms": round(q(0.95) * 1000, 1),
        "min_ms": round(ordered[0] * 1000, 1),
        "max_ms": round(ordered[-1] * 1000, 1),
    }


def synthetic_scene(width: int = 640, height: int = 480):
    """A deterministic, VLM-describable test card (BGR, camera order)."""
    import cv2  # noqa: F401 - optional dependency check
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (30, 34, 44))
    draw = ImageDraw.Draw(img)
    # red square, blue circle, yellow triangle - easy for Moondream to name
    draw.rectangle((60, 60, 220, 220), fill=(200, 40, 40))
    draw.ellipse((380, 80, 560, 260), fill=(40, 80, 200))
    draw.polygon([(320, 400), (220, 260), (420, 260)], fill=(210, 190, 40))
    draw.text((30, 430), "AURA BENCH", fill=(220, 220, 220))
    arr = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(arr[..., ::-1])  # RGB -> BGR


# ---------------------------------------------------------------------------
# stage runners
# ---------------------------------------------------------------------------

def bench_whisper(runs: int) -> dict[str, Any]:
    """Synthesise speech with Kokoro, transcribe with Whisper, time both."""
    from src.speech.kokoro_engine import get_tts_engine
    from src.speech.whisper_engine import get_stt_engine

    settings = get_settings()
    tts = get_tts_engine(settings.tts)
    tts.load()
    stt = get_stt_engine(settings.stt)
    stt.load()

    sentence = "Hello there Aura can you hear me testing the speech pipeline one two three."
    audio = tts.synthesize(sentence)
    sr = tts.sample_rate
    log.info("benchmark audio: %.1fs of speech", audio.size / sr)

    synth_times: list[float] = []
    for _ in range(max(1, runs // 2)):
        t0 = time.perf_counter()
        tts.synthesize("The quick brown fox jumps over the lazy dog.")
        synth_times.append(time.perf_counter() - t0)

    stt_times: list[float] = []
    similarity_best = 0.0
    for _ in range(runs):
        t0 = time.perf_counter()
        result = stt.transcribe(audio, sr, metrics=None)
        stt_times.append(time.perf_counter() - t0)
        if result.ok:
            a, b = sentence.lower().split(), result.text.lower().split()
            overlap = len(set(a) & set(b)) / max(1, len(set(a)))
            similarity_best = max(similarity_best, overlap)

    audio_s = audio.size / sr
    tts_stats = pct(synth_times)
    rtf = [t / audio_s for t in synth_times] if audio_s else []
    return {
        "audio_s": round(audio_s, 2),
        "stt": pct(stt_times),
        "tts_repeat": tts_stats,
        "tts_rtf_mean": round(statistics.fmean(rtf), 3) if rtf else None,
        "word_overlap": round(similarity_best, 2),
        "transcript_sample": result.text if result.ok else result.error,
    }


def bench_groq(runs: int) -> dict[str, Any]:
    from src.intelligence.groq_client import GroqClient

    client = GroqClient(get_settings().llm)
    samples = [
        "What do you see in front of the camera?",
        "Explain Newton's second law briefly.",
        "Hey robot, hello there!",
        "How far away is that red object?",
    ]
    latencies: list[float] = []
    errors = 0
    for i in range(runs):
        text = samples[i % len(samples)]
        t0 = time.perf_counter()
        try:
            client.chat([{"role": "user", "content": text}], max_tokens=64)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.warning("groq run %d failed: %s", i, exc)
        latencies.append(time.perf_counter() - t0)
    return {"total": pct(latencies), "errors": errors, "model": client.model_chain[0] if client.model_chain else "?"}


def bench_vlm(runs: int) -> dict[str, Any]:
    from src.perception.moondream_engine import get_engine
    from src.perception.preprocessing import frame_to_pil

    settings = get_settings()
    engine = get_engine(settings.vlm)
    image = frame_to_pil(synthetic_scene())

    cold: dict[str, float] = {}
    if not engine.is_loaded:
        t0 = time.perf_counter()
        engine.load()
        cold["load_s"] = round(time.perf_counter() - t0, 2)

    times: list[float] = []
    evidence = None
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        evidence = engine.ask(image, "What objects are visible in this image?")
        times.append(time.perf_counter() - t0)
    answer = getattr(evidence, "raw_answer", "") or str(evidence)
    return {
        **cold,
        "warm": pct(times),
        "sample_answer": answer[:160],
        "inference_ms_last": getattr(evidence, "inference_time_ms", None),
        "device": engine.status().get("device"),
    }


def bench_tts(runs: int) -> dict[str, Any]:
    from src.speech.kokoro_engine import get_tts_engine

    tts = get_tts_engine(get_settings().tts)
    tts.load()
    text = "AURA is a multimodal voice assistant for robotics demonstrations."
    audio = tts.synthesize(text)
    audio_s = audio.size / tts.sample_rate

    times: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        tts.synthesize(text)
        times.append(time.perf_counter() - t0)
    stats = pct(times)
    stats["audio_s"] = round(audio_s, 2)
    stats["rtf_mean"] = round(statistics.fmean([t / audio_s for t in times]), 3)
    stats["device"] = tts.status().get("device")
    return stats


def _inject_frame(agent) -> bool:
    """Put a synthetic scene in the camera buffer when no real camera works."""
    cam = agent.camera
    if cam is None:
        return False
    try:
        frame = cam.buffer.get()
        if frame is not None and frame.image.size and frame.image.max() > 10:
            return True  # real, non-black frames are already flowing
    except Exception:  # noqa: BLE001
        pass
    try:
        cam.buffer.put(synthetic_scene(), source="benchmark-synthetic")
        return True
    except Exception:  # noqa: BLE001
        return False


def bench_e2e(agent, text: str, runs: int, label: str) -> dict[str, Any]:
    times: list[float] = []
    stage_totals: dict[str, list[float]] = {}
    last_metrics = None
    for i in range(runs):
        # Reset memory each iteration so the router does not short-circuit
        # a repeated prompt to the "repeat last response" fast path.
        agent.memory.clear()
        t0 = time.perf_counter()
        record = agent.handle_text(text, speak=False)
        times.append(time.perf_counter() - t0)
        last_metrics = agent.last_metrics
        if last_metrics:
            for stage, dur in last_metrics.stages.items():
                stage_totals.setdefault(stage, []).append(dur)
        if record.error:
            log.warning("%s run %d error: %s", label, i, record.error)

    out: dict[str, Any] = {"turn": pct(times), "text": text}
    if last_metrics:
        out["stages"] = {k: pct(v) for k, v in stage_totals.items()}
        out["intent"] = last_metrics.flags.get("intent")
        out["tool"] = last_metrics.flags.get("tool")
        out["response_sample"] = (agent.last_record.response[:120] if agent.last_record else "")
    return out


def bench_pipeline(runs: int) -> dict[str, Any]:
    from src.orchestrator.bootstrap import build_agent

    agent = build_agent(with_camera=True, with_stt=False, with_tts=False, warm=False)
    agent.start(warm_models=True)
    try:
        has_camera = _inject_frame(agent)
        general = bench_e2e(agent, "Explain what machine learning is in two sentences.", runs, "e2e_text")
        visual = bench_e2e(agent, "What objects are visible in front of the camera?", runs, "e2e_vision")
        visual["frame_source"] = "camera" if has_camera else "synthetic-injected"
        return {"general": general, "visual": visual}
    finally:
        agent.shutdown()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

STAGES = ("whisper", "groq", "vlm", "tts", "pipeline")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark the AURA pipeline")
    ap.add_argument("--runs", type=int, default=5, help="iterations per stage")
    ap.add_argument("--stages", default="", help=f"comma list of {STAGES} (default: all)")
    ap.add_argument("--out", default="", help="JSON output path")
    ap.add_argument("--skip-e2e", action="store_true", help="skip the full agent loop")
    args = ap.parse_args(argv)

    setup_logging("INFO")
    wanted = [s.strip() for s in args.stages.split(",") if s.strip()] or list(STAGES)
    unknown = set(wanted) - set(STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": args.runs,
        "python": sys.version.split()[0],
        "stages": {},
    }

    runners: dict[str, Callable[[], Any]] = {
        "whisper": lambda: bench_whisper(args.runs),
        "groq": lambda: bench_groq(max(3, args.runs)),
        "vlm": lambda: bench_vlm(args.runs),
        "tts": lambda: bench_tts(args.runs),
    }

    for name in wanted:
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        try:
            if name == "pipeline":
                if args.skip_e2e:
                    report["stages"][name] = {"skipped": "--skip-e2e"}
                    continue
                report["stages"][name] = bench_pipeline(max(2, args.runs // 2 or 1))
            else:
                report["stages"][name] = runners[name]()
            report["stages"][name]["wall_s"] = round(time.perf_counter() - t0, 2)
            print(json.dumps(report["stages"][name], indent=2))
        except Exception as exc:  # noqa: BLE001 - report and keep going
            report["stages"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            log.exception("stage %s failed", name)
            print(report["stages"][name])

    out_path = Path(args.out or ROOT / "reports" / "benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {out_path}")

    txt_path = out_path.with_suffix(".txt")
    lines = [f"AURA benchmark {report['generated_at']}  runs={args.runs}", "=" * 60]
    for name, data in report["stages"].items():
        lines.append(f"\n[{name}]")
        if "error" in data or "skipped" in data:
            lines.append(f"  {data}")
            continue
        for key, val in data.items():
            if isinstance(val, dict) and "p50_ms" in val:
                lines.append(
                    f"  {key}: mean={val['mean_ms']}ms p50={val['p50_ms']}ms "
                    f"p95={val['p95_ms']}ms (n={val['runs']})"
                )
            elif isinstance(val, dict):
                lines.append(f"  {key}: {val}")
            else:
                lines.append(f"  {key}: {val}")
    txt_path.write_text("\n".join(lines))
    print(f"summary written to {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
