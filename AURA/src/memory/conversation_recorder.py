"""Durable conversation event recorder (PECM write path).

Records user transcripts, VLM observations, tool results and assistant
responses into SQLite as soon as each artifact is available — never waiting
for embeddings or summarization.  The original interaction is authoritative;
derived indexes (FTS, embeddings, episodes) are updated afterwards.

AURA is thread-based (not asyncio), so the API is synchronous; callers on
the audio/GUI thread should hand off to the scheduler.  All writes go
through the shared :class:`MemoryStore` (already RLock-protected).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from src.memory.event_models import (
    ConversationEvent,
    ConversationSession,
    EventType,
    RecordEventAck,
)
from src.memory.models import new_id, utcnow

log = logging.getLogger("aura.memory.recorder")


class ConversationRecorder:
    """Writes durable conversation events; commit-acknowledged, idempotent."""

    def __init__(self, store, *, enabled: bool = True):
        self.store = store
        self.enabled = enabled
        self._lock = threading.RLock()
        self._session_id: Optional[str] = None
        self._seq = 0

    # -- session lifecycle ----------------------------------------------
    def ensure_session(self, session_id: Optional[str] = None,
                       metadata: Optional[dict] = None) -> str:
        """Open (or adopt) the active session; idempotent."""
        with self._lock:
            if session_id:
                self._session_id = session_id
            if self._session_id is None:
                self._session_id = new_id("sess")
            if self.store is not None and self.enabled:
                try:
                    self.store.upsert_session(self._session_id, metadata=metadata)
                except Exception as exc:  # noqa: BLE001
                    log.warning("session upsert failed: %s", exc)
            return self._session_id

    def end_session(self) -> None:
        with self._lock:
            if self._session_id and self.store is not None and self.enabled:
                try:
                    self.store.close_session(self._session_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning("session close failed: %s", exc)
            self._session_id = None
            self._seq = 0

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- generic write ---------------------------------------------------
    def _write(self, event_type: EventType, content: str, *,
               turn_id: str = "", source_model: str = "",
               source_tool: str = "", frame_id: str = "",
               metadata: Optional[dict] = None) -> RecordEventAck:
        if not self.enabled or self.store is None:
            return RecordEventAck(ok=False, error="conversation memory disabled")
        if not content or not content.strip():
            return RecordEventAck(ok=False, error="empty content")
        try:
            session_id = self.ensure_session()
            event = ConversationEvent(
                session_id=session_id,
                turn_id=turn_id or new_id("turn"),
                event_sequence=self._next_seq(),
                event_type=event_type,
                content=content.strip(),
                source_model=source_model or None,
                source_tool=source_tool or None,
                frame_id=frame_id or None,
                timestamp=utcnow(),
                metadata=metadata or {},
            )
            inserted = self.store.insert_event(event)
            return RecordEventAck(
                ok=True,
                event_id=event.event_id,
                session_id=session_id,
                duplicate=not inserted,
                # immediate keyword indexing happens inside insert_event
                indexed_immediately=inserted,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("event record failed (%s): %s", event_type.value, exc)
            return RecordEventAck(ok=False, error=str(exc))

    # -- typed writers ----------------------------------------------------
    def record_user_transcript(self, session_id: Optional[str], turn_id: str,
                               text: str, timestamp: Any = None) -> RecordEventAck:
        if session_id:
            self.ensure_session(session_id)
        return self._write(EventType.USER_TRANSCRIPT, text, turn_id=turn_id)

    def record_user_text(self, session_id: Optional[str], turn_id: str,
                         text: str) -> RecordEventAck:
        if session_id:
            self.ensure_session(session_id)
        return self._write(EventType.USER_TEXT, text, turn_id=turn_id)

    def record_visual_observation(self, session_id: Optional[str], turn_id: str,
                                  frame_id: str, question: str, observation: str,
                                  model_id: str = "moondream2",
                                  metadata: Optional[dict] = None) -> RecordEventAck:
        """Persist a VLM answer together with the question that produced it."""
        if session_id:
            self.ensure_session(session_id)
        meta = dict(metadata or {})
        meta.setdefault("user_question", question)
        meta.setdefault("camera_status_at_observation", "active")
        return self._write(
            EventType.VLM_OBSERVATION, observation, turn_id=turn_id,
            source_model=model_id, frame_id=frame_id, metadata=meta,
        )

    def record_tool_result(self, session_id: Optional[str], turn_id: str,
                           tool_name: str, result_text: str,
                           metadata: Optional[dict] = None) -> RecordEventAck:
        if session_id:
            self.ensure_session(session_id)
        return self._write(
            EventType.TOOL_RESULT, result_text, turn_id=turn_id,
            source_tool=tool_name, metadata=metadata,
        )

    def record_assistant_response(self, session_id: Optional[str], turn_id: str,
                                  text: str, model_id: str = "") -> RecordEventAck:
        if session_id:
            self.ensure_session(session_id)
        return self._write(
            EventType.ASSISTANT_RESPONSE, text, turn_id=turn_id,
            source_model=model_id or None,
        )

    def record_correction(self, session_id: Optional[str], turn_id: str,
                          text: str) -> RecordEventAck:
        if session_id:
            self.ensure_session(session_id)
        return self._write(EventType.MEMORY_CORRECTION, text, turn_id=turn_id)
