#!/usr/bin/env python3
"""Verify the local Whisper (faster-whisper) speech-to-text integration.

Three modes:

* ``--synth``  – synthesise a sentence with Kokoro and transcribe it
  (no microphone needed; measures end-to-end STT accuracy locally).
* ``--record N`` – record N seconds from the microphone and transcribe.
* default      – synthesise, transcribe, and additionally sanity-check that
  silence produces no transcript.

    python3 scripts/verify_whisper.py [--synth TEXT] [--record 4] [--model small.en]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

DEFAULT_SENTENCE = "Hey robot, can you tell me what is on this table right now?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth", default=DEFAULT_SENTENCE, help="text to synthesise then transcribe")
    ap.add_argument("--record", type=float, default=0.0, help="record N seconds from the microphone")
    ap.add_argument("--model", default=None, help="override STT_MODEL")
    ap.add_argument("--no-vad", action="store_true")
    args = ap.parse_args()

    from config.settings import get_settings

    s = get_settings()
    if args.model:
        s = s.with_overrides(**{"stt.model": args.model})
    if args.no_vad:
        s = s.with_overrides(**{"stt.vad_enabled": False})

    from src.speech.whisper_engine import WhisperEngine

    engine = WhisperEngine(s.stt)
    t0 = time.perf_counter()
    try:
        info = engine.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] load: {type(exc).__name__}: {exc}")
        return 1
    print(f"model  : {info['model']} on {info['device']} ({info['compute_type']})")
    print(f"load   : {info['load_seconds']:.1f}s (weights {info['local_path']})")
    warm = engine.warmup()
    print(f"warmup : {warm * 1000:.0f} ms")

    ok = True

    if args.record > 0:
        from src.speech.audio_capture import AudioCapture

        print(f"\nrecording {args.record:.1f}s from the microphone…")
        with AudioCapture(s.stt) as cap:
            rec = cap.record_for(args.record)
        print(f"captured {rec.duration_s:.2f}s  peak={rec.peak:.3f} rms={rec.rms:.4f}")
        result = engine.transcribe(rec.samples, rec.sample_rate, metrics=None)
        print(f"transcript ({result.latency_s * 1000:.0f} ms): {result.text!r}")
        return 0 if result.ok else 1

    # --- synthesised round trip -----------------------------------------
    print("\n[1] synthesising a test sentence with Kokoro…")
    from src.speech.kokoro_engine import get_tts_engine

    tts = get_tts_engine(s.tts)
    tts.load()
    audio = tts.synthesize(args.synth)
    duration = tts.duration_s(audio)
    print(f"    {duration:.2f}s of audio generated")

    print("[2] transcribing with Whisper…")
    # Kokoro outputs 24 kHz; the engine resamples internally.
    result = engine.transcribe(audio, tts.sample_rate)
    print(f"    latency : {result.latency_s * 1000:.0f} ms")
    print(f"    language: {result.language} (p={result.language_probability:.2f})")
    print(f"    text    : {result.text!r}")

    if not result.ok:
        print("[FAIL] no transcript produced")
        ok = False
    else:
        score = _similarity(args.synth.lower(), result.text.lower())
        print(f"    similarity vs source: {score:.2f}")
        if score < 0.5:
            print("[WARN] transcript differs substantially from the source text")
        else:
            print("[OK] transcript looks correct")

    print("[3] silence should transcribe to nothing…")
    silent = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    if silent.text:
        print(f"[FAIL] silence produced a transcript: {silent.text!r}")
        ok = False
    else:
        print("    empty transcript suppressed as expected")

    print(f"\ntotal script time: {time.perf_counter() - t0:.1f}s")
    print("[OK] whisper verified" if ok else "[FAIL] whisper verification failed")
    return 0 if ok else 1


def _similarity(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()


if __name__ == "__main__":
    raise SystemExit(main())
