#!/usr/bin/env python3
"""Verify GroqCloud connectivity, model access and routing behaviour.

    python3 scripts/verify_groq.py [--list] [--route "some utterance"] [--chat "hi"]

The API key is never printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

SAMPLES = [
    "Hello robot!",
    "What do you see in front of you?",
    "Explain Newton's second law.",
    "What color is the bottle on the left?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list accessible models")
    ap.add_argument("--route", default="", help="route one utterance")
    ap.add_argument("--chat", default="", help="plain chat completion")
    ap.add_argument("--all-routes", action="store_true", help="route the built-in samples")
    args = ap.parse_args()

    from config.settings import get_settings

    s = get_settings().llm
    masked = f"{s.api_key[:4]}…(len={len(s.api_key)})" if s.api_key else "NOT SET"
    print(f"api key : {masked}")
    print(f"model   : {s.model}")
    print(f"fallback: {s.fallback_model}")
    print(f"base url: {s.base_url}")
    if not s.available:
        print("[FAIL] no API key configured")
        return 1

    from src.intelligence.groq_client import GroqClient, sdk_base_url

    client = GroqClient(s)
    failures = 0

    if args.list or not (args.route or args.chat or args.all_routes):
        try:
            from groq import Groq

            raw = Groq(
                api_key=s.api_key,
                base_url=sdk_base_url(s.base_url),
                timeout=s.timeout_s,
                max_retries=0,
            ).models.list()
            ids = sorted(m.id for m in raw.data)
            print(f"\naccessible models ({len(ids)}):")
            for i in ids:
                mark = "*" if i in (s.model, s.fallback_model) else " "
                print(f"  {mark} {i}")
            if s.model not in ids:
                print(f"\n[FAIL] configured GROQ_MODEL '{s.model}' is not accessible")
                failures += 1
            if s.fallback_model not in ids:
                print(f"[WARN] fallback '{s.fallback_model}' is not accessible")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] models.list: {type(exc).__name__}: {exc}")
            failures += 1
        if args.list:
            return 1 if failures else 0

    if args.chat:
        try:
            res = client.chat([{"role": "user", "content": args.chat}], max_tokens=200)
            print(f"\nchat ({res.model}, {res.latency_s * 1000:.0f} ms, "
                  f"{res.prompt_tokens}+{res.completion_tokens} tok):")
            print(res.text.strip())
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] chat: {type(exc).__name__}: {exc}")
            failures += 1

    if args.route or args.all_routes:
        from src.intelligence.intent_classifier import IntentRouter

        router = IntentRouter(client)
        texts = [args.route] if args.route else SAMPLES
        for text in texts:
            decision = router.route(text)
            plan = decision.plan
            print(f"\n  {text!r}")
            print(
                f"    -> {plan.intent.value} via {decision.router}"
                f" (conf={plan.confidence:.2f}, tool={plan.tool},"
                f" camera={plan.requires_camera})"
            )
            if decision.error:
                print(f"    error: {decision.error}")
                failures += 1
            if plan.is_visual:
                print(f"    visual question: {plan.normalized_question!r}")

    # timing probe
    try:
        res = client.chat([{"role": "user", "content": "Say OK."}], max_tokens=300)
        print(f"\nlatency probe: {res.latency_s * 1000:.0f} ms using {res.model}")
        if not res.text.strip():
            print("[WARN] empty content (reasoning models may spend tokens on reasoning; "
                  "raise GROQ_MAX_COMPLETION_TOKENS if this repeats)")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] probe: {type(exc).__name__}: {exc}")
        failures += 1

    print("\n[OK] groq verified" if not failures else f"\n[FAIL] {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
