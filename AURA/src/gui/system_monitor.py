"""System monitor strip: state, models, VRAM, queue depth, last error.

Polls ``agent.status()`` on a timer — cheap enough (~sub-ms) to run at 2 Hz.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class SystemMonitor(QFrame):
    def __init__(self, agent: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._agent = agent
        self.setObjectName("SystemMonitor")
        self.setStyleSheet(
            "#SystemMonitor { border: 1px solid #2c3140; border-radius: 8px; }"
            "QLabel { color: #d5d9e0; font-size: 12px; }"
            "QLabel[key] { color: #7f8798; }"
        )

        title = QLabel("System")
        title.setStyleSheet("font-weight: 600;")

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        self._cells: dict[str, QLabel] = {}
        keys = [
            ("State", "state"),
            ("Mode", "mode"),
            ("Router", "router"),
            ("STT", "stt"),
            ("VLM", "vlm"),
            ("TTS", "tts"),
            ("VRAM", "vram"),
            ("Queue", "queue"),
            ("Turns", "turns"),
            ("Error", "error"),
        ]
        for i, (label, key) in enumerate(keys):
            k = QLabel(label + ":")
            k.setProperty("key", True)
            v = QLabel("—")
            v.setMinimumWidth(90)
            grid.addWidget(k, i // 5, (i % 5) * 2)
            grid.addWidget(v, i // 5, (i % 5) * 2 + 1)
            self._cells[key] = v

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(grid)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            st = self._agent.status()
        except Exception:
            return
        vram = st.get("vram") or {}
        queue = st.get("queue") or {}
        self._set("state", st.get("state", "—"))
        self._set("mode", st.get("mode", "—"))
        self._set("router", st.get("router", "—"))
        stt = st.get("stt") or {}
        self._set("stt", f"{stt.get('model', '—')} {'on' if stt.get('loaded') else 'off'}")
        vlm = st.get("vlm") or {}
        self._set("vlm", f"{vlm.get('status', '—')}")
        tts = st.get("tts") or {}
        self._set("tts", f"{tts.get('status', '—')}")
        total = vram.get("total_mb")
        used = vram.get("used_mb")
        self._set("vram", f"{used}/{total} MB" if total else "n/a")
        self._set("queue", f"{queue.get('depth', 0)} queued · {queue.get('processed', 0)} done")
        self._set("turns", str(st.get("memory_turns", 0)))
        err = st.get("last_error")
        self._set("error", err if err else "none")

    def _set(self, key: str, text: str) -> None:
        label = self._cells.get(key)
        if label is not None:
            label.setText(text if len(text) < 60 else text[:57] + "…")
            label.setToolTip(text)

    def shutdown(self) -> None:
        self._timer.stop()
