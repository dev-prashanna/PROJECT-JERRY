"""AURA desktop dashboard (PySide6).

Layout::

    ┌────────────────────────────────────────────────┐
    │  Conversation                    Camera        │
    │                                   preview       │
    │                                   [blank warn]  │
    │  ──────────────────────────────────────────────  │
    │  Voice controls      System monitor             │
    └────────────────────────────────────────────────┘

All agent callbacks arrive through :class:`AgentBridge` signals, so every
widget update happens on the Qt main thread.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

from PySide6.QtCore import Qt, QRunnable, QThreadPool, QTimer
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.gui.bridge import AgentBridge, wire_agent
from src.gui.camera_panel import CameraPanel
from src.gui.conversation_panel import ConversationPanel
from src.gui.speech_panel import SpeechPanel
from src.gui.system_monitor import SystemMonitor

_DARK_QSS = """
QWidget { background: #0b0e14; color: #d5d9e0; font-size: 13px; }
QFrame#Panel { border: 1px solid #2c3140; border-radius: 8px; }
QStatusBar { color: #7f8798; font-size: 11px; }
"""


class _Submitter(QRunnable):
    """Runs a blocking agent call on the global thread pool."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._fn = fn

    def run(self) -> None:
        try:
            self._fn()
        except Exception:
            traceback.print_exc()


class MainWindow(QMainWindow):
    def __init__(
        self,
        agent: Any,
        bridge: AgentBridge,
        *,
        listen: bool = True,
        speak: bool = True,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.bridge = bridge
        self._listen = listen

        self.setWindowTitle("AURA — Multimodal Voice Assistant")
        self.resize(1100, 700)
        self.setStyleSheet(_DARK_QSS)
        font = QFont("Inter", 10)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        self.setFont(font)

        # --- panels ----------------------------------------------------
        self.conversation = ConversationPanel(agent)
        self.camera = CameraPanel(agent)
        self.speech = SpeechPanel(agent, speak_default=speak)
        self.monitor = SystemMonitor(agent)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.addWidget(self.camera, 3)
        right_l.addWidget(self.monitor, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.conversation)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        bottom = QWidget()
        bottom_l = QHBoxLayout(bottom)
        bottom_l.setContentsMargins(0, 0, 0, 0)
        bottom_l.addWidget(self.speech)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(splitter, 1)
        layout.addWidget(bottom)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("starting…")

        # --- wiring ----------------------------------------------------
        wire_agent(agent, bridge)
        bridge.state.connect(self._on_state)
        bridge.transcript.connect(self._on_transcript)
        bridge.route.connect(self._on_route)
        bridge.response.connect(self._on_response)
        bridge.error.connect(self._on_error)

        self.conversation.submit_requested.connect(self._submit_text)
        self.conversation.stop_speaking_requested.connect(agent.stop_speaking)
        self.speech.stop_speaking_requested.connect(agent.stop_speaking)
        self.speech.recording_finished.connect(self._on_recording)

        # --- lifecycle -------------------------------------------------
        self.agent.start(warm_models=True)
        self.statusBar().showMessage(
            f"online · router={agent.router.mode} · "
            f"{'mic ON' if listen else 'mic OFF'}"
        )

    # ------------------------------------------------------------------
    # agent events (already on the Qt main thread via queued signals)
    # ------------------------------------------------------------------
    def _on_state(self, transition: Any) -> None:
        current = getattr(transition, "current", transition)
        name = getattr(current, "value", str(current))
        self.speech.set_state(name)
        self.statusBar().showMessage(f"state → {name}")

    def _on_transcript(self, text: str, result: Any = None) -> None:
        self.conversation.add_status(f"heard: {text!r}")

    def _on_route(self, decision: Any) -> None:
        plan = getattr(decision, "plan", None)
        if plan is None:
            return
        intent = getattr(plan.intent, "value", plan.intent)
        tool = getattr(plan, "tool", "?")
        router = getattr(decision, "router", "?")
        self.conversation.add_status(
            f"route → {intent} · tool={tool} · router={router}"
        )

    def _on_response(self, record: Any) -> None:
        text = getattr(record, "response", "") or ""
        if not text:
            return
        plan = getattr(record, "plan", None)
        intent = getattr(getattr(plan, "intent", None), "value", "")
        strategy = getattr(record, "response_strategy", "")
        meta = " · ".join(x for x in (intent, strategy) if x)
        self.conversation.add_assistant(text, meta)

    def _on_error(self, message: str) -> None:
        self.conversation.add_error(message)
        self.statusBar().showMessage(f"error: {message}")

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def _submit_text(self, text: str) -> None:
        speak = self.speech.speak_enabled
        # The scheduler runs this on its own worker thread; just enqueue.
        self.agent.submit_text({"transcript": text, "speak": speak})

    def _on_recording(self, recording: Any) -> None:
        speak = self.speech.speak_enabled
        QThreadPool.globalInstance().start(
            _Submitter(lambda: self.agent.handle_recording(recording, speak=speak))
        )

    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.camera.shutdown()
        self.monitor.shutdown()
        self.speech.shutdown()
        try:
            self.agent.shutdown(timeout_s=3.0)
        except Exception:
            traceback.print_exc()
        event.accept()


def run_gui(settings: Any = None, *, speak: bool = True, listen: bool = True) -> int:
    """Build the agent + Qt app and run the event loop. Returns an exit code."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AURA")

    from src.orchestrator.bootstrap import build_agent

    agent = build_agent(
        settings=settings,
        with_camera=True,
        with_stt=True,
        with_tts=True,
        warm=False,
        speak=speak,
    )
    bridge = AgentBridge()
    window = MainWindow(agent, bridge, listen=listen, speak=speak)
    window.show()

    # Let the pipeline finish loading in the background; show a hint meanwhile.
    QTimer.singleShot(4000, lambda: window.statusBar().showMessage(
        "warmup complete · ready for voice or text"
    ))

    code = app.exec()
    try:
        agent.shutdown(timeout_s=3.0)
    except Exception:
        traceback.print_exc()
    return int(code)
