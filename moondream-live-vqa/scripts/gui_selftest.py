#!/usr/bin/env python3
"""GUI self-test: launch the real window, ask a question, save a screenshot.

Used to verify the desktop application end-to-end without manual interaction
(works with the regular X display or with ``QT_QPA_PLATFORM=offscreen``).

Usage::

    python scripts/gui_selftest.py
    python scripts/gui_selftest.py --ask "What do you see?" --seconds 25
    QT_QPA_PLATFORM=offscreen python scripts/gui_selftest.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import build_config  # noqa: E402
from src.utils.logging_utils import configure_logging, get_logger  # noqa: E402

log = get_logger("gui_selftest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask", default="What objects are visible?")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--screenshot",
        default=str(ROOT / "outputs" / "gui_screenshot.png"),
    )
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument(
        "--close-during-inference",
        action="store_true",
        help="close the window while the first answer is still generating",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from src.gui.main_window import STYLESHEET, MainWindow

    config = build_config(
        camera_index=args.camera,
        model_path=args.model_path,
        device=args.device,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow(config)
    window.show()

    events: list[tuple[str, dict]] = []
    original_emit = window.pipeline._on_event

    def recorder(kind: str, payload: dict) -> None:
        events.append((kind, dict(payload)))
        original_emit(kind, payload)

    window.pipeline._on_event = recorder

    state = {
        "asked": False,
        "answered": False,
        "screenshot_taken": False,
        "frames": 0,
        "asked_at": 0.0,
    }
    started = time.time()

    def tick() -> None:
        elapsed = time.time() - started
        snapshot = window.pipeline.get_latest_frame()
        if snapshot is not None:
            state["frames"] = max(state["frames"], snapshot.frame_id)

        if (
            not state["asked"]
            and window.pipeline.model_loaded
            and snapshot is not None
            and elapsed > 1.0
        ):
            log.info("Submitting scripted question: %r", args.ask)
            window._submit_question(args.ask)
            state["asked"] = True
            state["asked_at"] = time.time()

        if (
            args.close_during_inference
            and state["asked"]
            and not state.get("closing")
            and time.time() - float(state.get("asked_at", 0)) > 0.3
        ):
            state["closing"] = True
            log.info("Closing the window while inference is still running…")
            window.close()
            app.quit()

        if state["asked"] and not state["answered"]:
            for kind, payload in events:
                if kind == "inference_result":
                    state["answered"] = True
                    log.info(
                        "Answer [%s] frame=%s in %.3fs: %s",
                        payload.get("status"),
                        payload.get("frame_id"),
                        float(payload.get("inference_time_seconds") or 0),
                        payload.get("answer") or payload.get("error"),
                    )
                    break

        if state["answered"] and not state["screenshot_taken"]:
            time.sleep(0.4)  # let the bubble render
            app.processEvents()
            pixmap = window.grab()
            target = Path(args.screenshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            saved = pixmap.save(str(target))
            state["screenshot_taken"] = saved
            log.info("Screenshot -> %s (%s)", target, "saved" if saved else "FAILED")

        if elapsed > args.seconds or (
            state["screenshot_taken"] and state["frames"] > 0
        ):
            log.info(
                "Self-test complete: frames=%s asked=%s answered=%s shot=%s",
                state["frames"],
                state["asked"],
                state["answered"],
                state["screenshot_taken"],
            )
            window.close()
            app.quit()

    timer = QTimer()
    timer.setInterval(250)
    timer.timeout.connect(tick)
    timer.start()

    exit_code = app.exec()
    timer.stop()

    camera_ok = state["frames"] > 0
    model_ok = window.pipeline.model_loaded
    answered = state["answered"]
    shot = state["screenshot_taken"]
    print("\n--- GUI self-test summary ---")
    print(f"camera frames captured : {state['frames']} ({'OK' if camera_ok else 'FAIL'})")
    print(f"model loaded           : {model_ok} ({'OK' if model_ok else 'FAIL'})")
    print(f"question answered      : {answered} ({'OK' if answered else 'FAIL'})")
    print(f"screenshot written     : {shot} ({'OK' if shot else 'FAIL'})")
    print(f"last error             : {window.pipeline.last_error or 'none'}")
    if args.close_during_inference:
        closed = state.get("closing", False) and not window.pipeline.camera.is_running
        print(f"closed during inference: {closed} ({'OK' if closed else 'FAIL'})")
        ok = camera_ok and model_ok and state["asked"] and closed
    else:
        ok = camera_ok and model_ok and answered and shot
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1 if exit_code == 0 else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
