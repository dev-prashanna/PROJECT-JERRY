"""Chat transcript and text-command box.

The pipeline always reports through :class:`RequestRecord`; this panel renders
user turns, assistant turns, and the route/intent that produced them.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_CSS = """
QTextEdit { background: #10131a; color: #d5d9e0; border: 1px solid #2c3140;
            border-radius: 6px; padding: 6px; font-size: 13px; }
QLineEdit { background: #10131a; color: #d5d9e0; border: 1px solid #2c3140;
            border-radius: 6px; padding: 6px 8px; font-size: 13px; }
QPushButton { background: #243049; color: #d5d9e0; border: none;
              border-radius: 6px; padding: 6px 14px; }
QPushButton:hover { background: #2f3d5c; }
QPushButton:disabled { background: #1a1f2b; color: #555c6b; }
"""


class ConversationPanel(QFrame):
    submit_requested = Signal(str)   # emitted for both text and voice input
    stop_speaking_requested = Signal()

    def __init__(self, agent: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._agent = agent
        self.setObjectName("ConversationPanel")
        self.setStyleSheet(_CSS + "#ConversationPanel { border: 1px solid #2c3140; border-radius: 8px; }")

        title = QLabel("Conversation")
        title.setStyleSheet("font-weight: 600; color: #d5d9e0;")

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setPlaceholderText("Say or type something…")

        self.input = QLineEdit()
        self.input.setPlaceholderText('Try: "what do you see?" or "explain gravity"')
        self.input.returnPressed.connect(self._send)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send)

        self.interrupt_btn = QPushButton("Stop")
        self.interrupt_btn.clicked.connect(self.stop_speaking_requested.emit)

        row = QHBoxLayout()
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        row.addWidget(self.interrupt_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.chat, 1)
        layout.addLayout(row)

        self.chat.append(
            '<span style="color:#7f8798;">[AURA] Ready. Text or press-and-hold the mic to talk.</span>'
        )

    # ------------------------------------------------------------------
    def _send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.add_user(text)
        self.submit_requested.emit(text)

    def add_user(self, text: str) -> None:
        self._append(f'<b style="color:#7aa2f7;">You:</b> {self._esc(text)}')

    def add_assistant(self, text: str, meta: str = "") -> None:
        tag = f' <span style="color:#7f8798;font-size:11px;">{self._esc(meta)}</span>' if meta else ""
        self._append(f'<b style="color:#9ece6a;">AURA:</b> {self._esc(text)}{tag}')

    def add_status(self, text: str) -> None:
        self._append(f'<span style="color:#7f8798;font-size:11px;">[status] {self._esc(text)}</span>')

    def add_error(self, text: str) -> None:
        self._append(f'<span style="color:#f7768e;">[error] {self._esc(text)}</span>')

    # ------------------------------------------------------------------
    def _append(self, html: str) -> None:
        self.chat.moveCursor(QTextCursor.MoveOperation.End)
        self.chat.insertHtml(html + "<br>")
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    @staticmethod
    def _esc(text: str) -> str:
        return (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
