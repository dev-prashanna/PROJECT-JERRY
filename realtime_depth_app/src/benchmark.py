"""Inference benchmarks: input resolution x precision, measured on the local GPU/CPU."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import AppConfig
from .depth_engine import DepthEngine

RESOLUTIONS = (252, 378, 518)   # multiples of 14 (nearest to the common 256/384/518)
PRECISIONS = ("fp32", "autocast_fp16", "autocast_bf16")
ITERS = 40
WARMUP = 5


def _load_frame(cfg: AppConfig) -> np.ndarray:
    for candidate in (
        cfg.base_dir.parent / "assets" / "sample_frame.jpeg",
        cfg.base_dir / "assets" / "sample_frame.jpeg",
    ):
        if candidate.exists():
            img = cv2.imread(str(candidate))
            if img is not None:
                return img
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (cfg.source.height, cfg.source.width, 3), dtype=np.uint8)


def _bench_one(cfg: AppConfig, frame: np.ndarray, input_size: int, precision: str, device: str) -> dict:
    from dataclasses import replace

    model_cfg = replace(cfg.model, input_size=input_size, precision=precision, device=device, warmup_iters=0)
    bench_cfg = replace(cfg, model=model_cfg)
    try:
        engine = DepthEngine(bench_cfg)
    except Exception as exc:
        return {"input_size": input_size, "precision": precision, "device": device, "error": str(exc)}

    engine.reset_peak_memory()
    for _ in range(WARMUP):
        engine.infer(frame)

    times: list[float] = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        engine.infer(frame)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times)
    depth = engine.infer(frame).depth
    return {
        "input_size": input_size,
        "precision": precision,
        "device": str(engine.device),
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "fps": round(1000.0 / float(arr.mean()), 1),
        "vram_mb": round(engine.gpu_memory_mb() or 0.0, 1),
        "output_shape": list(depth.shape),
        "finite": bool(np.isfinite(depth).all()),
    }


def _env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
    return info


def run_benchmarks(cfg: AppConfig, out_dir: Path | None = None) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() and cfg.model.device in ("auto", "cuda") else "cpu"
    frame = _load_frame(cfg)
    rows: list[dict] = []
    precisions = PRECISIONS if device == "cuda" else ("fp32",)
    for size in RESOLUTIONS:
        if size % 14 != 0:
            continue
        for precision in precisions:
            row = _bench_one(cfg, frame, size, precision, device)
            rows.append(row)
            label = row.get("error") or f"{row['mean_ms']:.1f} ms ({row['fps']:.1f} fps, {row['vram_mb']:.0f} MB)"
            print(f"input {size:>4} | {precision:<14} | {device:<5} | {label}")

    env = _env_info()
    out_dir = out_dir or (cfg.base_dir / "benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"results_{stamp}.json").write_text(json.dumps({"env": env, "results": rows}, indent=2))

    lines = [
        "# Inference benchmarks",
        "",
        f"Generated: {env['timestamp']}",
        "",
        f"- GPU: {env.get('gpu', 'n/a')} ({env.get('vram_total_mb', 0):.0f} MB VRAM)" if "gpu" in env else "- CPU only",
        f"- torch {env['torch']} | python {env['python']}",
        f"- frame: {frame.shape[1]}x{frame.shape[0]}, {ITERS} timed iterations after {WARMUP} warmup",
        "- input sizes are multiples of 14 (ViT patch size); 252/378 ~= 256/384",
        "",
        "| input | precision | device | mean ms | p95 ms | fps | VRAM MB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['input_size']} | {r['precision']} | {r['device']} | error: {r['error']} | | | |")
        else:
            lines.append(
                f"| {r['input_size']} | {r['precision']} | {r['device']} | {r['mean_ms']} | {r['p95_ms']} "
                f"| {r['fps']} | {r['vram_mb']} |"
            )
    lines += [
        "",
        "Notes:",
        "- Model-only latency (preprocess + forward + bicubic upsample to frame size).",
        "- End-to-end application latency additionally includes capture, rendering and display;",
        "  measure it with `python3 main.py --headless 60` (see outputs/headless_metrics.json).",
        "- These are measured values on this machine, not targets.",
    ]
    md_path = out_dir / f"results_{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {md_path}")
    return rows
