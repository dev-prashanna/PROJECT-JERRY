"""Voice controls: push-to-talk record, speak toggle, barge-in interrupt."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget


class _RecordWorker(QThread):
    """Runs one blocking ``agent.listen_once`` off the GUI thread."""

    finished_ok = Signal(object)    # Recording
    failed = Signal(str)

    def __init__(self, agent: Any, seconds: Optional[float], parent=None) -> None:
        super().__init__(parent)
        self._agent = agent
        self._seconds = seconds

    def run(self) -> None:
        try:
            rec = self._agent.listen_once(self._seconds)
            self.finished_ok.emit(rec)
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
            self.failed.emit(str(exc))


class SpeechPanel(QFrame):
    recording_finished = Signal(object)   # Recording -> submit to agent
    recording_failed = Signal(str)        # mic / capture error text
    stop_speaking_requested = Signal()

    def __init__(self, agent: Any, speak_default: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._agent = agent
        self._worker: Optional[_RecordWorker] = None

        self.setObjectName("SpeechPanel")
        self.setStyleSheet(
            "#SpeechPanel { border: 1px solid #2c3140; border-radius: 8px; }"
            "QPushButton { background: #243049; color: #d5d9e0; border: none;"
            " border-radius: 6px; padding: 8px 16px; font-weight: 600; }"
            "QPushButton:hover { background: #2f3d5c; }"
            "QPushButton:pressed { background: #3d4f75; }"
            "QPushButton:disabled { background: #1a1f2b; color: #555c6b; }"
            "QPushButton#Record { background: #4a2832; }"
            "QPushButton#Record:checked { background: #b23a48; }"
        )

        title = QLabel("Voice")
        title.setStyleSheet("font-weight: 600; color: #d5d9e0;")

        self.record_btn = QPushButton("Hold to talk")
        self.record_btn.setObjectName("Record")
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip("Record one utterance (VAD auto-stops).")
        self.record_btn.clicked.connect(self._toggle_record)

        self.stop_btn = QPushButton("Stop speaking")
        self.stop_btn.clicked.connect(self.stop_speaking_requested.emit)

        self.speak_toggle = QCheckBox("Speak replies")
        self.speak_toggle.setChecked(speak_default)
        self.speak_toggle.setStyleSheet("color: #d5d9e0;")

        self.state_label = QLabel("idle")
        self.state_label.setStyleSheet("color: #7f8798; font-size: 11px;")

        row = QHBoxLayout()
        row.addWidget(title)
        row.addSpacing(8)
        row.addWidget(self.record_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.speak_toggle)
        row.addStretch(1)
        row.addWidget(self.state_label)

        layout = QHBoxLayout(self)
        layout.addLayout(row)

    # ------------------------------------------------------------------
    @property
    def speak_enabled(self) -> bool:
        return self.speak_toggle.isChecked()

    def set_state(self, name: str) -> None:
        self.state_label.setText(name)
        busy = name in {"capturing", "transcribing", "interpreting", "generating", "speaking"}
        self.record_btn.setEnabled(not busy)

    def _toggle_record(self, checked: bool) -> None:
        if checked:
            self._start_recording()
        else:
            # user released the button; VAD usually already finished the clip
            pass

    def _start_recording(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self.record_btn.setText("Recording…")
        self.state_label.setText("capturing")
        worker = _RecordWorker(self._agent, seconds=None, parent=self)
        worker.finished_ok.connect(self._on_recorded)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _on_recorded(self, recording: object) -> None:
        self._reset_button()
        self.recording_finished.emit(recording)

    def _on_failed(self, message: str) -> None:
        self._reset_button()
        self.state_label.setText("mic error")
        self.state_label.setToolTip(message)
        self.recording_failed.emit(message)

    def _reset_button(self) -> None:
        self.record_btn.setChecked(False)
        self.record_btn.setText("Hold to talk")

    def shutdown(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
