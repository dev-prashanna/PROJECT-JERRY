#!/usr/bin/env python3
"""AURA — hybrid multimodal voice intelligence system for robotics.

Launch commands
---------------
Text pipeline (no audio hardware needed)::

    python3 main.py --text "What do you see in front of you?"

Interactive terminal front-end (push-to-talk + typed commands)::

    python3 main.py --cli

Scripted acceptance demo (the three required live interactions)::

    python3 main.py --demo

Desktop dashboard::

    python3 main.py --gui

Run ``python3 main.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    from config.settings import APP_ROOT as _root

    ap = argparse.ArgumentParser(
        prog="aura",
        description="AURA: listen → interpret → observe → reason → speak",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--text", default="", help="process one text command and exit")
    ap.add_argument("--cli", action="store_true", help="interactive terminal front-end")
    ap.add_argument("--gui", action="store_true", help="launch the PySide6 dashboard")
    ap.add_argument("--demo", action="store_true", help="run the scripted acceptance demo")
    ap.add_argument("--no-speak", action="store_true", help="do not synthesise/play speech")
    ap.add_argument("--no-listen", action="store_true", help="skip microphone usage")
    ap.add_argument("--offline", action="store_true", help="force offline degraded mode")
    ap.add_argument("--router", default="", choices=["", "llm", "deterministic", "auto", "compare"])
    ap.add_argument("--strategy", default="", choices=["", "auto", "full", "direct"])
    ap.add_argument("--llm-provider", default="", help="override LLM_PROVIDER (e.g. groq, openai, anthropic)")
    ap.add_argument("--llm-model", default="", help="override LLM_MODEL for the selected provider")
    ap.add_argument("--vlm-provider", default="", help="override VLM_PROVIDER (e.g. moondream, openai, groq)")
    ap.add_argument("--vlm-model", default="", help="override VLM_MODEL for the selected vision provider")
    ap.add_argument("--quiet", action="store_true", help="less console output")
    ap.add_argument("--version", action="store_true")
    return ap


def apply_overrides(args: argparse.Namespace):
    overrides = {}
    if args.offline:
        overrides["orchestrator.offline"] = True
    if args.router:
        overrides["orchestrator.router_mode"] = args.router
    if args.strategy:
        overrides["orchestrator.response_strategy"] = args.strategy
    if args.llm_provider:
        overrides["llm.provider"] = args.llm_provider
    if args.llm_model:
        overrides["llm.model"] = args.llm_model
    if args.vlm_provider:
        overrides["vlm.provider"] = args.vlm_provider
    if args.vlm_model:
        overrides["vlm.model"] = args.vlm_model
    if args.quiet:
        overrides["log_level"] = "WARNING"
    if not overrides:
        return get_settings()
    settings = get_settings().with_overrides(**overrides)
    # refresh the process-wide cache so helpers see the overrides
    import config.settings as settings_module

    settings_module._settings = settings
    return settings


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_text(settings, text: str, speak: bool = True) -> int:
    from src.orchestrator.bootstrap import build_agent
    from src.utils.logger import setup_logging

    setup_logging(settings.log_level)
    agent = build_agent(settings, with_camera=True, with_stt=False, with_tts=speak, warm=False)
    agent.start(warm_models=False)
    try:
        record = agent.handle_text(text, speak=speak)
        _print_turn(agent, record, verbose=not settings.log_level == "WARNING")
        return 1 if record.error and not record.response else 0
    finally:
        agent.shutdown()


def run_cli(settings, speak: bool = True, listen: bool = True) -> int:
    from src.orchestrator.bootstrap import build_agent
    from src.utils.logger import setup_logging

    setup_logging(settings.log_level)
    print("AURA terminal front-end — type a command, or press Enter on an empty line to talk.")
    print("Commands: /status  /state  /stop  /clear  /mic <seconds>  /quit")
    agent = build_agent(
        settings,
        with_camera=True,
        with_stt=listen,
        with_tts=speak,
        warm=True,
    )
    agent.start(warm_models=True)
    _wire_events(agent, quiet=False)

    try:
        while True:
            try:
                line = input("\naura> ")
            except EOFError:
                break
            cmd = line.strip()
            if not cmd:
                if not listen:
                    print("microphone disabled (--no-listen)")
                    continue
                _push_to_talk(agent, speak=speak)
                continue
            if cmd in {"/quit", "/exit", "quit", "exit"}:
                break
            if cmd == "/status":
                import json

                print(json.dumps(agent.status(), indent=2, default=str))
                continue
            if cmd == "/state":
                print(agent.sm.snapshot())
                continue
            if cmd == "/stop":
                agent.stop_speaking()
                print("speech stopped")
                continue
            if cmd == "/clear":
                agent.memory.clear()
                print("conversation memory cleared")
                continue
            if cmd.startswith("/mic"):
                parts = cmd.split()
                seconds = float(parts[1]) if len(parts) > 1 else 4.0
                _push_to_talk(agent, seconds=seconds, speak=speak)
                continue
            if cmd.startswith("/"):
                print(f"unknown command: {cmd}")
                continue

            record = agent.handle_text(cmd, speak=speak)
            _print_turn(agent, record)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        agent.shutdown()
    return 0


def _push_to_talk(agent, seconds: float | None = None, speak: bool = True) -> None:
    print("listening… (Ctrl+C to abort)")
    try:
        recording = agent.listen_once(seconds)
    except KeyboardInterrupt:
        agent.stop_speaking()
        print("aborted")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"microphone error: {exc}")
        return
    if recording.duration_s <= 0 or recording.is_effectively_empty:
        print("no speech detected")
        return
    print(f"recorded {recording.duration_s:.2f}s (peak {recording.peak:.2f})")
    record = agent.handle_recording(recording, metrics=None)
    _print_turn(agent, record)


def run_demo(settings, speak: bool = True, listen: bool = True) -> int:
    """The three required acceptance interactions (plus a bypass check)."""
    from src.orchestrator.bootstrap import build_agent
    from src.utils.logger import setup_logging

    setup_logging(settings.log_level)
    print("=" * 70)
    print("AURA ACCEPTANCE DEMO")
    print("=" * 70)

    agent = build_agent(
        settings, with_camera=True, with_stt=listen, with_tts=speak, warm=True
    )
    agent.start(warm_models=True)
    _wire_events(agent, quiet=False)

    interactions = [
        ("Hey robot, hello!", "Interaction 1 — greeting (no camera)"),
        ("What do you see in front of you?", "Interaction 2 — visual question (camera + Moondream)"),
        ("Explain what machine learning is.", "Interaction 3 — general knowledge (no VLM)"),
    ]
    failures = 0
    try:
        for text, title in interactions:
            print("\n" + "-" * 70)
            print(title)
            print(f"USER: {text}")
            record = agent.handle_text(text, speak=speak)
            _print_turn(agent, record)
            checks = _check_interaction(text, record)
            for name, ok, detail in checks:
                print(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")
                if not ok:
                    failures += 1

        # extra: proof that non-visual requests never touch the VLM
        print("\n" + "-" * 70)
        print("Extra — offline degradation probe (no camera involvement expected)")
        settings_off = settings.with_overrides(**{"orchestrator.offline": True})
        import config.settings as settings_module

        settings_module._settings = settings_off
        agent.settings = settings_off
        agent.router.mode = "deterministic"
        record = agent.handle_text("What is the capital of France?", speak=speak)
        _print_turn(agent, record)
        if record.plan and record.plan.is_visual:
            print("  [FAIL] offline general question unexpectedly required the camera")
            failures += 1
        else:
            print("  [OK] offline general question stayed local")
        settings_module._settings = settings
        agent.settings = settings
        agent.router.mode = settings.orchestrator.router_mode
    finally:
        print("\n" + "=" * 70)
        metrics = agent.last_metrics
        if metrics:
            print("last request metrics:", metrics.to_dict())
        agent.shutdown()

    print(f"\nDEMO {'PASSED' if failures == 0 else f'FAILED ({failures} checks)'}")
    return 0 if failures == 0 else 1


def _check_interaction(text: str, record) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    wants_visual = "see" in text.lower() or "what do you see" in text.lower()
    is_general = "machine learning" in text.lower()

    has_response = bool(record.response.strip())
    checks.append(("response produced", has_response, repr(record.response[:80])))

    used_vlm = bool(record.evidence) or any(t.tool == "analyze_camera" for t in record.tool_results)
    if wants_visual:
        checks.append(("camera snapshot used", used_vlm, f"frame={record.evidence[0].frame_id if record.evidence else None}"))
        checks.append(("evidence present", bool(record.evidence_text), record.evidence_text[:80]))
    if is_general:
        checks.append(("VLM bypassed", not used_vlm, f"tools={[t.tool for t in record.tool_results]}"))
        checks.append(("intent correct", record.plan and record.plan.intent.value == "general_question",
                       record.plan.intent.value if record.plan else "none"))
    return checks


def _wire_events(agent, quiet: bool = False) -> None:
    agent.on("transcript", lambda text, res: print(f"  STT   : {text!r}"))
    agent.on(
        "route",
        lambda d: print(
            f"  ROUTE : {d.plan.intent.value} via {d.router} "
            f"(conf={d.plan.confidence:.2f}, tool={d.plan.tool}, camera={d.plan.requires_camera})"
        ),
    )
    agent.on("error", lambda msg: print(f"  ERROR : {msg}"))


def _print_turn(agent, record, verbose: bool = True) -> None:
    print(f"  INTENT: {record.plan.intent.value if record.plan else 'n/a'}"
          f"  tool={record.plan.tool if record.plan else 'n/a'}"
          f"  strategy={record.response_strategy}")
    if record.evidence:
        for ev in record.evidence:
            print(f"  VLM   : [{ev.frame_id}] {ev.raw_answer}"
                  f"  ({ev.inference_time_ms} ms, age={ev.image_age_s}s)")
    for t in record.tool_results:
        if t.tool != "analyze_camera":
            print(f"  TOOL  : {t.tool} ok={t.ok} {t.error or t.text[:60]}")
    print(f"  AURA  : {record.response}")
    if record.error:
        print(f"  NOTE  : {record.error}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print("AURA 1.0.0")
        return 0

    settings = apply_overrides(args)
    speak = not args.no_speak

    if args.text:
        return run_text(settings, args.text, speak=speak)
    if args.demo:
        return run_demo(settings, speak=speak, listen=not args.no_listen)
    if args.gui:
        from src.gui.main_window import run_gui

        return run_gui(settings, speak=speak, listen=not args.no_listen)
    if args.cli or True:
        return run_cli(settings, speak=speak, listen=not args.no_listen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
