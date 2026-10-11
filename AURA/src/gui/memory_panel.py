"""Memory panel: stored entities/facts, face enrolments, forget action.

Polls ``agent.memory_engine`` on a timer (2 Hz is plenty — the store is
local SQLite).  Deletions run off the GUI thread and require an explicit
confirmation dialog; person deletions get a stronger warning.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.memory.models import ForgetRequest


class _ForgetWorker(QThread):
    """Runs one blocking ``engine.delete_memory`` off the GUI thread."""

    finished_ok = Signal(object)  # MemoryOperationResult
    failed = Signal(str)

    def __init__(self, engine: Any, request: ForgetRequest, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._request = request

    def run(self) -> None:
        try:
            result = self._engine.delete_memory(self._request)
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
            self.failed.emit(str(exc))


class MemoryPanel(QFrame):
    forget_requested = Signal(str)  # human-readable summary for the transcript

    def __init__(self, agent: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._agent = agent
        self._worker: Optional[_ForgetWorker] = None
        self._entity_rows: dict[str, dict] = {}
        self._fact_rows: dict[str, dict] = {}

        self.setObjectName("MemoryPanel")
        self.setStyleSheet(
            "#MemoryPanel { border: 1px solid #2c3140; border-radius: 8px; }"
            "QLabel { color: #d5d9e0; font-size: 12px; }"
            "QLabel[key] { color: #7f8798; }"
            "QListWidget { background: #12161f; color: #d5d9e0; border: 1px solid #2c3140;"
            " border-radius: 6px; font-size: 12px; }"
            "QListWidget::item:selected { background: #2f3d5c; }"
            "QPushButton { background: #243049; color: #d5d9e0; border: none;"
            " border-radius: 6px; padding: 6px 12px; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background: #2f3d5c; }"
            "QPushButton:disabled { background: #1a1f2b; color: #555c6b; }"
            "QPushButton#Forget { background: #4a2832; }"
            "QPushButton#Forget:hover { background: #b23a48; }"
        )

        title = QLabel("Memory")
        title.setStyleSheet("font-weight: 600;")
        self._status = QLabel("—")
        self._status.setWordWrap(True)

        self._entities = QListWidget()
        self._entities.setMinimumHeight(90)
        self._facts = QListWidget()
        self._facts.setMinimumHeight(110)
        self._history = QListWidget()
        self._history.setMinimumHeight(110)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        self._forget_btn = QPushButton("Forget selected")
        self._forget_btn.setObjectName("Forget")
        self._forget_btn.clicked.connect(self._forget_selected)
        buttons = QHBoxLayout()
        buttons.addWidget(self._refresh_btn)
        buttons.addWidget(self._forget_btn)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self._status)
        ent_label = QLabel("Entities")
        ent_label.setProperty("key", True)
        layout.addWidget(ent_label)
        layout.addWidget(self._entities)
        fact_label = QLabel("Facts")
        fact_label.setProperty("key", True)
        layout.addWidget(fact_label)
        layout.addWidget(self._facts)
        hist_label = QLabel("Recent conversation")
        hist_label.setProperty("key", True)
        layout.addWidget(hist_label)
        layout.addWidget(self._history)
        layout.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    # ------------------------------------------------------------------
    def _engine(self) -> Any:
        return getattr(self._agent, "memory_engine", None)

    def refresh(self) -> None:
        engine = self._engine()
        if engine is None or not getattr(engine, "enabled", False):
            self._status.setText("Memory is disabled (set MEMORY_ENABLED=1).")
            self._entities.clear()
            self._facts.clear()
            self._history.clear()
            return
        try:
            result = engine.list_memories()
        except Exception:  # noqa: BLE001 - never let polling crash the GUI
            return
        if not result.ok:
            self._status.setText(f"Memory error: {result.message or result.error_code}")
            return

        counts = result.payload.get("counts", {}) or {}
        self._status.setText(
            f"{counts.get('entities', 0)} entities · "
            f"{counts.get('declarative_memory', 0)} facts · "
            f"{counts.get('face_embeddings', 0)} faces · "
            f"{counts.get('object_embeddings', 0)} objects"
        )

        self._entities.blockSignals(True)
        self._facts.blockSignals(True)
        self._history.blockSignals(True)
        try:
            self._populate_entities(result.payload.get("entities", []))
            self._populate_facts(result.payload.get("facts", []))
        finally:
            self._entities.blockSignals(False)
            self._facts.blockSignals(False)
            self._history.blockSignals(False)
        self._refresh_history(engine)

    def _populate_entities(self, entities: list[dict]) -> None:
        selected = self._selected_entity_id()
        self._entities.clear()
        self._entity_rows.clear()
        for e in entities:
            text = f"{e.get('name', '?')}  ({e.get('type', '?')}, {e.get('status', '?')})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, e.get("entity_id"))
            item.setToolTip(str(e.get("entity_id", "")))
            self._entities.addItem(item)
            self._entity_rows[str(e.get("entity_id"))] = e
        if selected:
            for i in range(self._entities.count()):
                if self._entities.item(i).data(Qt.ItemDataRole.UserRole) == selected:
                    self._entities.setCurrentRow(i)
                    break

    def _populate_facts(self, facts: list[dict]) -> None:
        self._facts.clear()
        self._fact_rows.clear()
        for f in facts:
            text = f"{f.get('predicate', '?')}: {f.get('object_value', '?')}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, f.get("fact_id"))
            item.setToolTip(str(f.get("fact_id", "")))
            self._facts.addItem(item)
            self._fact_rows[str(f.get("fact_id"))] = f

    def _selected_entity_id(self) -> Optional[str]:
        item = self._entities.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _refresh_history(self, engine: Any) -> None:
        """Populate the recent-conversation list from the durable event log."""
        store = getattr(engine, "store", None)
        if store is None or getattr(engine, "recorder", None) is None:
            self._history.clear()
            return
        try:
            events = store.recent_events(limit=20)
        except Exception:  # noqa: BLE001 - history is optional enrichment
            return
        self._history.clear()
        for ev in reversed(events):
            label = ev.event_type.value.replace("_", " ")
            text = f"[{label}] {ev.content[:90]}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, ev.event_id)
            item.setToolTip(ev.content)
            self._history.addItem(item)

    # ------------------------------------------------------------------
    def _forget_selected(self) -> None:
        engine = self._engine()
        if engine is None:
            return
        entity_item = self._entities.currentItem()
        fact_item = self._facts.currentItem()
        if entity_item is None and fact_item is None:
            QMessageBox.information(self, "Forget", "Select an entity or fact first.")
            return

        if entity_item is not None:
            entity_id = entity_item.data(Qt.ItemDataRole.UserRole)
            info = self._entity_rows.get(str(entity_id), {})
            name = info.get("name", entity_id)
            is_person = info.get("type") == "person"
            prompt = (
                f"Delete everything about the person '{name}', including their "
                f"enrolled face embeddings?"
                if is_person
                else f"Delete '{name}' and everything associated with it?"
            )
            title = "Forget person" if is_person else "Forget"
            answer = QMessageBox.warning(
                self,
                title,
                prompt,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            request = ForgetRequest(entity_id=str(entity_id))
            summary = f"forgot {name}"
        else:
            fact_id = fact_item.data(Qt.ItemDataRole.UserRole)
            info = self._fact_rows.get(str(fact_id), {})
            prompt = f"Delete the fact '{info.get('predicate')}: {info.get('object_value')}'?"
            answer = QMessageBox.warning(
                self,
                "Forget fact",
                prompt,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            request = ForgetRequest(fact_id=str(fact_id))
            summary = f"forgot fact {info.get('predicate')}"

        self._forget_btn.setEnabled(False)
        self._worker = _ForgetWorker(engine, request, self)
        self._worker.finished_ok.connect(self._on_forget_done)
        self._worker.failed.connect(self._on_forget_failed)
        self._worker._summary = summary  # type: ignore[attr-defined]
        self._worker.start()

    def _on_forget_done(self, result: Any) -> None:
        self._forget_btn.setEnabled(True)
        summary = getattr(self._worker, "_summary", "forgot memory") if self._worker else "forgot memory"
        if getattr(result, "ok", False):
            self.forget_requested.emit(summary)
        else:
            QMessageBox.warning(
                self, "Forget failed", getattr(result, "message", "unknown error")
            )
        self.refresh()

    def _on_forget_failed(self, message: str) -> None:
        self._forget_btn.setEnabled(True)
        QMessageBox.warning(self, "Forget failed", message)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self._timer.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
