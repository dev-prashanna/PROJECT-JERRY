"""Right panel: question input, answers and conversation history."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.utils.performance import format_seconds

__all__ = ["ChatWidget"]


def _clock(ts: Optional[float] = None) -> str:
    value = ts if ts is not None else time.time()
    return time.strftime("%H:%M:%S", time.localtime(value))


class ChatWidget(QWidget):
    """Chat-style Q&A surface.

    Each answer entry records the *frame id* it was computed from, so a reply
    is never mistaken for a description of a continuous video stream.
    """

    question_submitted = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._busy = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 12, 12, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("AI Assistant")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.mode_badge = QLabel("LIVE")
        self.mode_badge.setObjectName("modeBadgeLive")
        header.addWidget(self.mode_badge)
        root.addLayout(header)

        self.model_label = QLabel("Model: loading…")
        self.model_label.setObjectName("modelLabel")
        self.model_label.setWordWrap(True)
        root.addWidget(self.model_label)

        # --- history -------------------------------------------------------
        self.history = QScrollArea()
        self.history.setWidgetResizable(True)
        self.history.setObjectName("history")
        self.history.setFrameShape(QFrame.Shape.NoFrame)
        self._history_container = QWidget()
        self._history_layout = QVBoxLayout(self._history_container)
        self._history_layout.setContentsMargins(4, 4, 4, 4)
        self._history_layout.setSpacing(8)
        self._history_layout.addStretch(1)
        self.history.setWidget(self._history_container)
        root.addWidget(self.history, stretch=1)

        # --- status --------------------------------------------------------
        status_row = QHBoxLayout()
        status_row.setSpacing(12)
        self.inference_label = QLabel("Inference: idle")
        self.inference_label.setObjectName("inferenceLabel")
        self.latency_label = QLabel("Latency: --")
        self.mode_label = QLabel("Mode: live snapshot")
        status_row.addWidget(self.inference_label)
        status_row.addWidget(self.latency_label)
        status_row.addWidget(self.mode_label)
        status_row.addStretch(1)
        root.addLayout(status_row)

        # --- input ---------------------------------------------------------
        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        self.input = QLineEdit()
        self.input.setPlaceholderText(
            "Ask about what the camera sees… (Enter to send)"
        )
        self.input.setClearButtonEnabled(True)
        self.input.returnPressed.connect(self._submit)
        self.input.setEnabled(False)
        input_row.addWidget(self.input, stretch=1)

        self.ask_btn = QPushButton("Ask AI")
        self.ask_btn.setObjectName("primary")
        self.ask_btn.setDefault(True)
        self.ask_btn.setEnabled(False)
        self.ask_btn.clicked.connect(self._submit)
        input_row.addWidget(self.ask_btn)
        root.addLayout(input_row)

        self.hint_label = QLabel("")
        self.hint_label.setObjectName("hintLabel")
        self.hint_label.setWordWrap(True)
        root.addWidget(self.hint_label)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def set_input_enabled(self, enabled: bool) -> None:
        self.input.setEnabled(enabled)
        self.ask_btn.setEnabled(enabled and not self._busy)
        if enabled:
            self.input.setFocus()

    def set_model_status(self, payload: Dict[str, Any]) -> None:
        state = payload.get("state", "not_loaded")
        if state == "loading":
            self.model_label.setText("Model: loading… (one-time, ~5 s)")
            self.set_input_enabled(False)
        elif state == "loaded":
            describe = payload.get("describe") or "model ready"
            path = payload.get("model_path", "")
            self.model_label.setText(f"Model: {describe}")
            self.model_label.setToolTip(path)
            self.set_input_enabled(True)
        elif state == "error":
            self.model_label.setText(f"Model error: {payload.get('error')}")
            self.model_label.setToolTip(str(payload.get("error", "")))
            self.set_input_enabled(False)
        else:
            self.model_label.setText("Model: not loaded")
            self.set_input_enabled(False)

    def set_mode(self, payload: Dict[str, Any]) -> None:
        mode = str(payload.get("mode", "live"))
        frozen = payload.get("frozen")
        if mode == "frozen":
            self.mode_badge.setText("FROZEN")
            self.mode_badge.setObjectName("modeBadgeFrozen")
            frame_id = frozen.get("frame_id") if frozen else "?"
            self.mode_label.setText(f"Mode: frozen frame #{frame_id}")
            self.hint_label.setText(
                "All questions refer to the frozen frame — the camera keeps "
                "running and the vision encoding is reused."
            )
        else:
            self.mode_badge.setText("LIVE")
            self.mode_badge.setObjectName("modeBadgeLive")
            self.mode_label.setText("Mode: live snapshot")
            self.hint_label.setText(
                "Every question captures a brand-new camera frame."
            )
        self.mode_badge.style().unpolish(self.mode_badge)
        self.mode_badge.style().polish(self.mode_badge)

    def set_busy(self, busy: bool, detail: str = "") -> None:
        self._busy = busy
        self.ask_btn.setEnabled(busy is False and self.input.isEnabled())
        if busy:
            self.inference_label.setText(f"Inference: running{(' — ' + detail) if detail else ''}")
            self.inference_label.setProperty("state", "running")
        else:
            self.inference_label.setText("Inference: idle")
            self.inference_label.setProperty("state", "idle")
        self.inference_label.style().unpolish(self.inference_label)
        self.inference_label.style().polish(self.inference_label)

    def set_latency(self, seconds: Optional[float]) -> None:
        self.latency_label.setText(f"Latency: {format_seconds(seconds)}")

    def add_user_message(self, text: str, timestamp: Optional[float] = None) -> None:
        self._add_entry("user", text, f"you · {_clock(timestamp)}")

    def add_answer(self, result: Dict[str, Any]) -> None:
        status = result.get("status", "success")
        if status != "success":
            body = result.get("error") or "The request failed."
            meta = f"error · {_clock(result.get('timestamp'))}"
            self._add_entry("error", str(body), meta)
            return

        answer = str(result.get("answer") or "").strip() or "(empty answer)"
        frame_id = result.get("frame_id")
        mode = result.get("mode", "")
        parts = [
            f"frame #{frame_id}" if frame_id is not None else "no frame",
            f"{format_seconds(result.get('inference_time_seconds'))}",
            str(result.get("device") or ""),
        ]
        if result.get("encoding_reused"):
            parts.append("encoding reused")
        if mode == "frozen":
            parts.append("frozen")
        elif mode:
            parts.append("live")
        meta = f"{_clock(result.get('timestamp'))} · " + " · ".join(p for p in parts if p)
        self._add_entry("assistant", answer, meta)
        self.set_latency(result.get("inference_time_seconds"))

    def add_system_message(self, text: str, level: str = "info") -> None:
        self._add_entry("error" if level == "error" else "system", text, _clock())

    def clear(self) -> None:
        while self._history_layout.count():
            item = self._history_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._history_layout.addStretch(1)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            self.input.setFocus()
            return
        self.question_submitted.emit(text)

    def _add_entry(self, role: str, text: str, meta: str) -> None:
        bubble = QFrame()
        bubble.setObjectName(
            {"user": "bubbleUser", "assistant": "bubbleAssistant"}.get(role, "bubbleSystem")
        )
        layout = QVBoxLayout(bubble)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if role in {"user", "assistant"}:
            body.setObjectName("bubbleBody")
        layout.addWidget(body)

        caption = QLabel(meta)
        caption.setObjectName("bubbleMeta")
        caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(caption)

        # Insert above the trailing stretch.
        index = max(0, self._history_layout.count() - 1)
        self._history_layout.insertWidget(index, bubble)

        # Keep the newest entry visible.
        QTimer.singleShot(0, self._scroll_to_bottom)
        if role == "user":
            self.input.clear()

    def _scroll_to_bottom(self) -> None:
        bar = self.history.verticalScrollBar()
        bar.setValue(bar.maximum())
