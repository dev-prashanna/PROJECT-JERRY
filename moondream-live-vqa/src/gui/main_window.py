"""Main application window: layout, dark theme, and event wiring.

Threading model
---------------
The pipeline emits ``on_event(kind, payload)`` from *its* worker threads
(camera, model loader, inference).  A tiny :class:`PipelineBridge` QObject
turns every event into a Qt queued signal, so all slots in
:class:`MainWindow` always execute on the GUI thread — no cross-thread widget
access ever happens.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config.settings import AppConfig
from src.camera.camera import CameraInfo
from src.gui.camera_widget import CameraWidget
from src.gui.chat_widget import ChatWidget
from src.pipeline.vqa_pipeline import VQAPipeline
from src.utils.logging_utils import get_logger
from src.utils.performance import describe_device, format_seconds

log = get_logger(__name__)

__all__ = ["MainWindow", "STYLESHEET"]


# ---------------------------------------------------------------------------
# Dark, professional theme
# ---------------------------------------------------------------------------

STYLESHEET = """
* { font-family: 'Inter', 'Segoe UI', 'Ubuntu', 'DejaVu Sans', sans-serif; }

QMainWindow, QWidget#root {
    background-color: #12151a;
    color: #d8dee6;
    font-size: 13px;
}
QLabel { color: #d8dee6; background: transparent; }
QLabel#panelTitle { font-size: 15px; font-weight: 600; color: #f0f3f7; }
QLabel#modelLabel { color: #9aa4b1; font-size: 12px; }
QLabel#hintLabel { color: #7d8794; font-size: 11px; }
QLabel#bubbleMeta { color: #7d8794; font-size: 11px; }
QLabel#bubbleBody { color: #e6ebf1; font-size: 13px; }
QLabel#inferenceLabel { color: #9aa4b1; }
QLabel#inferenceLabel[state="running"] { color: #f0b429; }
QLabel#inferenceLabel[state="idle"] { color: #7d8794; }

QLabel#modeBadgeLive, QLabel#modeBadgeFrozen {
    padding: 3px 10px;
    border-radius: 9px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#modeBadgeLive { background-color: #12351f; color: #4ade80; border: 1px solid #1d5c35; }
QLabel#modeBadgeFrozen { background-color: #3a2c0d; color: #fbbf24; border: 1px solid #6b4e13; }

QLabel#statusDot[state="streaming"] { color: #4ade80; }
QLabel#statusDot[state="connecting"], QLabel#statusDot[state="reconnecting"] { color: #fbbf24; }
QLabel#statusDot[state="error"], QLabel#statusDot[state="disconnected"] { color: #f87171; }
QLabel#statusDot[state="idle"], QLabel#statusDot[state="stopped"] { color: #7d8794; }

QLabel#preview {
    background-color: #0b0d11;
    border: 1px solid #262c35;
    border-radius: 8px;
    color: #6b7480;
    font-size: 13px;
}

QFrame#cameraStatusFrame, QFrame#statusBar {
    background-color: #171b21;
    border: 1px solid #262c35;
    border-radius: 6px;
}
QFrame#statusBar { border-radius: 0px; border-left: none; border-right: none; border-bottom: none; }

QPushButton {
    background-color: #232935;
    border: 1px solid #313a48;
    border-radius: 6px;
    padding: 6px 14px;
    color: #dbe2ea;
}
QPushButton:hover { background-color: #2b3342; border-color: #3d4859; }
QPushButton:pressed { background-color: #1d232d; }
QPushButton:disabled { background-color: #1b1f26; color: #5c6470; border-color: #262c35; }
QPushButton:checked { background-color: #3a2c0d; border-color: #6b4e13; color: #fbbf24; }
QPushButton#primary {
    background-color: #2563eb;
    border: 1px solid #3b82f6;
    color: white;
    font-weight: 600;
}
QPushButton#primary:hover { background-color: #2f6fed; }
QPushButton#primary:disabled { background-color: #1e2a44; border-color: #27364f; color: #7d8ba3; }

QLineEdit {
    background-color: #0f1217;
    border: 1px solid #2b3341;
    border-radius: 6px;
    padding: 8px 10px;
    color: #e6ebf1;
    selection-background-color: #2563eb;
}
QLineEdit:focus { border-color: #3b82f6; }
QLineEdit:disabled { color: #5c6470; }

QComboBox {
    background-color: #232935;
    border: 1px solid #313a48;
    border-radius: 6px;
    padding: 6px 10px;
    color: #dbe2ea;
    min-height: 20px;
}
QComboBox:hover { border-color: #3d4859; }
QComboBox QAbstractItemView {
    background-color: #1b2028;
    border: 1px solid #313a48;
    selection-background-color: #2563eb;
    color: #dbe2ea;
}

QScrollArea#history { background-color: transparent; border: none; }
QScrollArea#history > QWidget > QWidget { background: transparent; }
QFrame#bubbleUser {
    background-color: #16324f;
    border: 1px solid #1f4a75;
    border-radius: 8px;
}
QFrame#bubbleAssistant {
    background-color: #1a1f26;
    border: 1px solid #2b3341;
    border-radius: 8px;
}
QFrame#bubbleSystem {
    background-color: #2a1d1d;
    border: 1px solid #4a2b2b;
    border-radius: 8px;
}

QSplitter::handle { background-color: #12151a; width: 2px; }
QScrollBar:vertical { background: #12151a; width: 10px; margin: 0; }
QScrollBar::handle:vertical {
    background: #2b3341; border-radius: 5px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #3d4859; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #12151a; height: 10px; }
QStatusBar { background: #171b21; color: #9aa4b1; }
QToolTip { background-color: #1b2028; color: #dbe2ea; border: 1px solid #313a48; }
"""


class PipelineBridge(QObject):
    """Marshals pipeline events from worker threads into the GUI thread."""

    event = Signal(str, object)


class MainWindow(QMainWindow):
    """Application window: camera panel + AI assistant + status bar."""

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config
        self._discovering = False
        self._closing = False

        self.bridge = PipelineBridge(self)
        self.pipeline = VQAPipeline(config, on_event=self._pipeline_event)
        self.bridge.event.connect(self._on_event)

        self.setWindowTitle("Moondream Live VQA — local visual question answering")
        self.resize(1280, 780)
        self.setMinimumSize(980, 620)

        self._build_ui()
        self._connect_signals()

        # Startup sequence: camera stream first (so discovery skips the index
        # that is already being opened), then camera list, then model load.
        self.camera_widget.attach_camera(self.pipeline.camera)
        self.pipeline.start_camera()
        self.refresh_cameras()
        self.pipeline.load_model_async()
        self.chat_widget.set_mode({"mode": self.pipeline.mode})

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.camera_widget = CameraWidget()
        self.chat_widget = ChatWidget()
        splitter.addWidget(self.camera_widget)
        splitter.addWidget(self.chat_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 520])
        layout.addWidget(splitter, stretch=1)

        self.status_bar_frame = QFrame()
        self.status_bar_frame.setObjectName("statusBar")
        status_layout = QHBoxLayout(self.status_bar_frame)
        status_layout.setContentsMargins(12, 7, 12, 7)
        status_layout.setSpacing(22)

        self.status_camera = QLabel("Camera: starting")
        self.status_model = QLabel("Model: loading")
        self.status_inference = QLabel("Inference: idle")
        self.status_device = QLabel("Device: --")
        self.status_device.setObjectName("modelLabel")
        self.status_error = QLabel("Last error: none")
        self.status_error.setObjectName("modelLabel")
        self.status_error.setStyleSheet("color: #7d8794;")
        self.status_error.setMinimumWidth(180)

        for label in (
            self.status_camera,
            self.status_model,
            self.status_inference,
            self.status_device,
        ):
            status_layout.addWidget(label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.status_error)
        layout.addWidget(self.status_bar_frame)

    def _connect_signals(self) -> None:
        cam = self.camera_widget
        cam.start_requested.connect(self._start_camera)
        cam.stop_requested.connect(self.pipeline.stop_camera)
        cam.restart_requested.connect(self.pipeline.restart_camera)
        cam.refresh_requested.connect(self.refresh_cameras)
        cam.camera_selected.connect(self._select_camera)
        cam.freeze_toggled.connect(self._toggle_freeze)
        self.chat_widget.question_submitted.connect(self._submit_question)

    # ------------------------------------------------------------------
    # pipeline event -> GUI thread
    # ------------------------------------------------------------------
    def _pipeline_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Called from camera / loader / inference threads."""
        self.bridge.event.emit(kind, payload)

    @Slot(str, object)
    def _on_event(self, kind: str, payload: Any) -> None:
        if self._closing:
            return
        data = dict(payload or {})
        handler = {
            "camera_status": self._on_camera_status,
            "camera_selected": self._on_camera_selected,
            "cameras_found": self._on_cameras_found,
            "model_status": self._on_model_status,
            "submitted": self._on_submitted,
            "rejected": self._on_rejected,
            "inference_state": self._on_inference_state,
            "inference_result": self._on_inference_result,
            "mode_changed": self._on_mode_changed,
            "error": self._on_error,
            "stopped": self._on_stopped,
        }.get(kind)
        if handler is not None:
            try:
                handler(data)
            except Exception:  # pragma: no cover - GUI must never crash
                log.exception("failed handling event %s", kind)

    # -- individual handlers ------------------------------------------------
    def _on_camera_status(self, data: Dict[str, Any]) -> None:
        self.camera_widget.handle_camera_event(data)
        status = data.get("status", "")
        message = data.get("message", "")
        text = {
            "streaming": "Camera: connected",
            "connecting": "Camera: connecting",
            "reconnecting": "Camera: reconnecting",
            "disconnected": "Camera: disconnected",
            "error": "Camera: error",
            "stopped": "Camera: stopped",
            "idle": "Camera: idle",
        }.get(status, f"Camera: {status}")
        if status == "streaming":
            text += f" {data.get('width')}x{data.get('height')}"
        self.status_camera.setText(text)
        if status in {"error", "disconnected"}:
            self._set_last_error(message or status)

    def _on_camera_selected(self, data: Dict[str, Any]) -> None:
        self.camera_widget.set_selected_index(int(data.get("camera_index", 0)))

    def _on_cameras_found(self, data: Dict[str, Any]) -> None:
        cameras: List[CameraInfo] = list(data.get("cameras", []))
        # Discovery deliberately skips the index that is already streaming, so
        # re-insert it to keep the selector complete.
        camera = self.pipeline.camera
        if not any(info.index == camera.active_index for info in cameras):
            cameras.insert(
                0,
                CameraInfo(
                    index=camera.active_index,
                    name=camera.device_name or f"device {camera.active_index}",
                ),
            )
        self.camera_widget.set_camera_list(cameras)
        self.camera_widget.set_selected_index(camera.active_index)

    def _on_model_status(self, data: Dict[str, Any]) -> None:
        self.chat_widget.set_model_status(data)
        state = data.get("state", "not_loaded")
        if state == "loaded":
            self.status_model.setText(f"Model: loaded ({data.get('device', '')})")
            self.status_device.setText(
                f"Device: {describe_device(data.get('device') or '')}"
            )
        elif state == "loading":
            self.status_model.setText("Model: loading")
        elif state == "error":
            self.status_model.setText("Model: failed")
            self._set_last_error(str(data.get("error") or "model failed to load"))
        else:
            self.status_model.setText("Model: not loaded")

    def _on_submitted(self, data: Dict[str, Any]) -> None:
        self.chat_widget.add_user_message(
            str(data.get("question", "")), data.get("frame_timestamp")
        )
        frozen = bool(data.get("frozen"))
        detail = "frozen frame" if frozen else "live frame"
        self.chat_widget.set_busy(True, f"frame #{data.get('frame_id')} · {detail}")
        self.status_inference.setText("Inference: running")

    def _on_rejected(self, data: Dict[str, Any]) -> None:
        reason = str(data.get("reason", "request rejected"))
        self.chat_widget.add_system_message(reason, level="error")
        self._set_last_error(reason)

    def _on_inference_state(self, data: Dict[str, Any]) -> None:
        state = data.get("state")
        if state == "running":
            self.chat_widget.set_busy(True)
            self.status_inference.setText("Inference: running")
        elif state == "finished":
            self.chat_widget.set_busy(False)
            self.status_inference.setText("Inference: idle")

    def _on_inference_result(self, data: Dict[str, Any]) -> None:
        self.chat_widget.add_answer(data)
        self.chat_widget.set_busy(False)
        if data.get("status") == "success":
            self.status_inference.setText(
                f"Last answer: {format_seconds(data.get('inference_time_seconds'))}"
            )
        else:
            self.status_inference.setText("Inference: error")
            self._set_last_error(str(data.get("error") or "inference failed"))

    def _on_mode_changed(self, data: Dict[str, Any]) -> None:
        self.camera_widget.set_mode(data)
        self.chat_widget.set_mode(data)

    def _on_error(self, data: Dict[str, Any]) -> None:
        message = str(data.get("message", ""))
        source = str(data.get("source", ""))
        self._set_last_error(message)
        if source == "model":
            self.chat_widget.add_system_message(message, level="error")

    def _on_stopped(self, data: Dict[str, Any]) -> None:
        log.debug("Pipeline stopped after %.1fs", float(data.get("uptime_seconds", 0)))

    def _set_last_error(self, message: str) -> None:
        self.status_error.setText(f"Last error: {message[:80]}")
        self.status_error.setToolTip(message)
        self.status_error.setStyleSheet("color: #f87171;")

    # ------------------------------------------------------------------
    # user actions
    # ------------------------------------------------------------------
    @Slot()
    def refresh_cameras(self) -> None:
        """Scan for cameras off-thread so the UI never blocks."""
        if self._discovering:
            return
        self._discovering = True
        self.camera_widget.set_frame_hint("Scanning for cameras…")

        def worker() -> None:
            try:
                cameras = self.pipeline.discover_cameras()
            except Exception as exc:  # noqa: BLE001
                log.exception("camera discovery failed")
                cameras = []
                self._pipeline_event(
                    "error", {"message": f"camera discovery failed: {exc}", "source": "camera"}
                )
            self._pipeline_event("cameras_found", {"cameras": cameras})
            self._discovering = False

        threading.Thread(target=worker, name="camera-discovery", daemon=True).start()

    @Slot()
    def _start_camera(self) -> None:
        self.pipeline.start_camera()

    @Slot(int)
    def _select_camera(self, index: int) -> None:
        if index == self.pipeline.camera.active_index and self.pipeline.camera.is_running:
            return
        self.pipeline.start_camera(index=index)

    @Slot(bool)
    def _toggle_freeze(self, checked: bool) -> None:
        if checked:
            snap = self.pipeline.freeze()
            if snap is None:
                self.camera_widget.set_mode({"mode": "live"})
        else:
            self.pipeline.unfreeze()

    @Slot(str)
    def _submit_question(self, question: str) -> None:
        accepted, message = self.pipeline.ask(question)
        if not accepted:
            self.chat_widget.add_system_message(message, level="error")

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API)
        if self._closing:
            event.accept()
            return
        self._closing = True
        log.info("Close requested — stopping workers")
        self.status_camera.setText("Camera: shutting down")
        self.status_inference.setText("Inference: shutting down")
        self.camera_widget.shutdown()
        QApplication.processEvents()  # let the labels paint before blocking
        try:
            self.pipeline.stop(timeout=self.config.inference.shutdown_timeout)
        except Exception:  # pragma: no cover
            log.exception("pipeline shutdown failed")
        event.accept()
