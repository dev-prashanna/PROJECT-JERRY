#!/usr/bin/env python3
"""Moondream Live VQA — real-time visual question answering on a local camera.

Entry point with two front-ends sharing the same pipeline:

* ``python main.py``            → PySide6 desktop application (default)
* ``python main.py --cli``      → terminal front-end (useful over SSH)
* ``python main.py --cli --ask "What do you see?"`` → one-shot scripted run

The Moondream 2 checkpoint is loaded from ``MODEL_PATH`` (or an auto-detected
local folder); nothing is downloaded.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import AppConfig, build_config  # noqa: E402
from src.utils.logging_utils import configure_logging, get_logger  # noqa: E402

log = get_logger("main")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli(config: AppConfig, scripted: Optional[List[str]] = None) -> int:
    """Terminal front-end: same pipeline as the GUI, no Qt required."""
    from src.pipeline.vqa_pipeline import PipelineMode, VQAPipeline
    from src.utils.performance import format_seconds

    def on_event(kind: str, payload: Dict[str, Any]) -> None:
        if kind == "inference_result":
            status = payload.get("status")
            if status == "success":
                print(
                    f"\nmoondream ({payload.get('device')}, frame "
                    f"#{payload.get('frame_id')}, "
                    f"{format_seconds(payload.get('inference_time_seconds'))}):\n"
                    f"  {payload.get('answer')}\n"
                )
            else:
                print(f"\nERROR: {payload.get('error')}\n")
        elif kind == "error":
            print(f"\n[error] {payload.get('message')}", file=sys.stderr)
        elif kind == "camera_status" and payload.get("status") != "streaming":
            print(f"[camera] {payload.get('status')}: {payload.get('message')}")
        elif kind == "model_status":
            state = payload.get("state")
            if state == "loaded":
                print(f"[model] loaded: {payload.get('describe')}")
            elif state == "error":
                print(f"[model] failed: {payload.get('error')}", file=sys.stderr)

    pipeline = VQAPipeline(config, on_event=on_event)
    try:
        print(f"Starting camera {config.camera.device_index} …")
        pipeline.start_camera()
        deadline = time.time() + 10
        while pipeline.get_latest_frame() is None and time.time() < deadline:
            time.sleep(0.1)
        if pipeline.get_latest_frame() is None:
            print("WARNING: no camera frame yet (continuing; model still loads)")

        print("Loading Moondream 2 (first load takes a few seconds)…")
        try:
            pipeline.load_model_blocking()
        except Exception as exc:  # noqa: BLE001 - report and exit cleanly
            print(f"\nFailed to load the model: {exc}", file=sys.stderr)
            print(
                "Check MODEL_PATH and see docs/TROUBLESHOOTING.md.",
                file=sys.stderr,
            )
            return 2

        print(
            "\nReady. Type a question and press Enter.\n"
            "Commands: freeze | live | status | quit\n"
        )

        if scripted:
            questions = list(scripted)
            interactive = False
        else:
            questions = []
            interactive = True

        while True:
            if not interactive:
                if not questions:
                    break
                question = questions.pop(0)
                print(f"you> {question}")
            else:
                try:
                    question = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not question:
                    continue
                if question.lower() in {"quit", "exit", "q"}:
                    break
                if question.lower() == "freeze":
                    snap = pipeline.freeze()
                    print(f"frozen frame #{snap.frame_id}" if snap else "cannot freeze")
                    continue
                if question.lower() == "live":
                    pipeline.unfreeze()
                    print("live mode: every question uses a new frame")
                    continue
                if question.lower() == "status":
                    print(pipeline.status_summary)
                    continue

            accepted, message = pipeline.ask(question)
            if not accepted:
                print(f"[rejected] {message}")
                continue
            while pipeline.busy:
                time.sleep(0.05)

        return 0
    finally:
        pipeline.stop()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_gui(config: AppConfig) -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    from src.gui.main_window import STYLESHEET, MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Moondream Live VQA")
    app.setOrganizationName("VBMD")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)

    def excepthook(exc_type, exc, tb):  # pragma: no cover - crash safety net
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Unexpected error")
            box.setText(f"{exc_type.__name__}: {exc}")
            box.setDetailedText(
                "".join(
                    __import__("traceback").format_exception(exc_type, exc, tb)
                )
            )
            box.exec()
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook

    window = MainWindow(config)
    window.show()
    log.info("GUI started")
    return app.exec()


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cli", action="store_true", help="run the terminal front-end")
    parser.add_argument(
        "--ask",
        action="append",
        metavar="QUESTION",
        help="ask a question in --cli mode and exit (repeatable)",
    )
    parser.add_argument("--model-path", help="path to the local Moondream checkpoint")
    parser.add_argument("--camera", type=int, help="camera device index (default 0)")
    parser.add_argument("--width", type=int, help="capture width, e.g. 640")
    parser.add_argument("--height", type=int, help="capture height, e.g. 480")
    parser.add_argument("--fps", type=int, help="requested capture FPS")
    parser.add_argument("--fourcc", help="capture FOURCC (MJPG / YUYV / '')")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], help="inference device"
    )
    parser.add_argument(
        "--dtype", choices=["auto", "bf16", "fp16", "fp32"], help="model dtype"
    )
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="do not append the image-grounding instruction to questions",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        help="optional longest-side resize before encoding (default: no resize)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        help="pending request capacity (0 = reject while busy, default 1)",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    parser.add_argument(
        "--log-file", default=None, help="override the log file path"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = build_config(
        model_path=args.model_path,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        device=args.device,
        dtype=args.dtype,
        grounding=False if args.no_grounding else None,
        max_side=args.max_side,
        queue_size=args.queue_size,
        log_level=args.log_level,
    )
    configure_logging(config.log_level, log_file=args.log_file or config.log_file)

    log.info(
        "Starting | model=%s | camera=%s %sx%s | device=%s | offline=%s",
        config.model_path,
        config.camera.device_index,
        config.camera.width,
        config.camera.height,
        config.model.device,
        __import__("config.settings", fromlist=["OFFLINE_MODE"]).OFFLINE_MODE,
    )

    if args.cli:
        return run_cli(config, scripted=args.ask)
    return run_gui(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
