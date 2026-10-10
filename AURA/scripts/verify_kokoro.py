#!/usr/bin/env python3
"""Verify the local Kokoro-82M text-to-speech integration.

    python3 scripts/verify_kokoro.py [--text "..."] [--voice af_heart] [--play]

Without ``--play`` the script only synthesises and reports timing/length.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

DEFAULT_TEXT = (
    "Hello. I am AURA, the robot assistant. I can look through my camera "
    "and answer questions about what I see."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--play", action="store_true", help="play through the default speaker")
    ap.add_argument("--save", default="", help="optional wav output path")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from config.settings import get_settings

    s = get_settings()
    voice = args.voice or s.tts.voice
    speed = args.speed if args.speed is not None else s.tts.speed

    from src.speech.kokoro_engine import get_tts_engine

    print(f"voice : {voice}")
    print(f"speed : {speed}")
    try:
        engine = get_tts_engine()
        info = engine.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] load: {type(exc).__name__}: {exc}")
        return 1
    print(f"loaded: {info}")

    try:
        t0 = time.perf_counter()
        audio = engine.synthesize(args.text, voice=voice, speed=speed)
        synth_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] synth: {type(exc).__name__}: {exc}")
        return 1

    duration_s = engine.duration_s(audio)
    if duration_s <= 0:
        print("[FAIL] synthesised audio is empty")
        return 1

    rtf = (synth_ms / 1000.0) / duration_s
    print(f"audio : {duration_s:.2f}s at {engine.sample_rate} Hz ({len(audio)} samples)")
    print(f"synth : {synth_ms:.0f} ms  (real-time factor {rtf:.2f}; <1.0 is faster than real time)")

    if args.save:
        path = engine.save_wav(audio, args.save)
        print(f"saved : {path}")

    if args.play:
        t1 = time.perf_counter()
        ok = engine.play(audio)
        print(f"play  : {'ok' if ok else 'FAILED'} ({(time.perf_counter() - t1) * 1000:.0f} ms overhead)")
        if not ok:
            return 1

    print("[OK] kokoro verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
