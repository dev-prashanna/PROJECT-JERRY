"""Typed contracts for persistent episodic & conversational memory (PECM).

These models describe the durable event log, derived episodes, retrieval
requests/results and indexing jobs.  They extend — never replace — the core
MMS contracts in :mod:`src.memory.models`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from src.memory.models import new_id, utcnow


class EventType(str, Enum):
    USER_TRANSCRIPT = "user_transcript"
    USER_TEXT = "user_text"
    VLM_OBSERVATION = "vlm_observation"
    TOOL_RESULT = "tool_result"
    ASSISTANT_RESPONSE = "assistant_response"
    MEMORY_CORRECTION = "memory_correction"


class EpisodeType(str, Enum):
    VISUAL_OBSERVATION = "visual_observation"
    CONVERSATION = "conversation"
    IDENTITY_ENROLLMENT = "identity_enrollment"
    OBJECT_ENROLLMENT = "object_enrollment"
    FACT_DECLARATION = "fact_declaration"
    TOOL_INTERACTION = "tool_interaction"


class RecallStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    INDEX_PENDING = "index_pending"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    PERMISSION_DENIED = "permission_denied"


class ConversationSession(BaseModel):
    session_id: str = Field(default_factory=lambda: new_id("sess"))
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: Optional[datetime] = None
    session_metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationEvent(BaseModel):
    """One durable interaction record — the authoritative history."""

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    session_id: str
    turn_id: str = ""
    event_sequence: int = 0
    event_type: EventType
    content: str = Field(min_length=1)
    source_model: Optional[str] = None
    source_tool: Optional[str] = None
    frame_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    persistence_status: str = "complete"
    fts_rowid: Optional[int] = None

    def metadata_json(self) -> str:
        return json.dumps(self.metadata, default=str)

    @classmethod
    def from_row(cls, row: Any) -> "ConversationEvent":
        """Build from a sqlite3.Row or tuple returned by store queries."""
        data = dict(row) if not isinstance(row, dict) else row
        meta_raw = data.get("metadata_json") or "{}"
        try:
            metadata = json.loads(meta_raw)
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return cls(
            event_id=data["event_id"],
            session_id=data["session_id"],
            turn_id=data.get("turn_id", ""),
            event_sequence=int(data.get("event_sequence") or 0),
            event_type=EventType(data["event_type"]),
            content=data["content"],
            source_model=data.get("source_model"),
            source_tool=data.get("source_tool"),
            frame_id=data.get("frame_id"),
            timestamp=data["timestamp"],
            metadata=metadata,
            persistence_status=data.get("persistence_status", "complete"),
            fts_rowid=data.get("fts_rowid"),
        )


class Episode(BaseModel):
    """A derived summary of one conversational turn or observation."""

    episode_id: str = Field(default_factory=lambda: new_id("ep"))
    session_id: str
    turn_id: str = ""
    episode_summary: str = Field(min_length=1)
    start_timestamp: datetime = Field(default_factory=utcnow)
    end_timestamp: datetime = Field(default_factory=utcnow)
    source_event_ids: list[str] = Field(default_factory=list)
    frame_id: Optional[str] = None
    episode_type: EpisodeType = EpisodeType.CONVERSATION
    created_at: datetime = Field(default_factory=utcnow)

    def source_event_ids_json(self) -> str:
        return json.dumps(self.source_event_ids)

    @classmethod
    def from_row(cls, row: Any) -> "Episode":
        data = dict(row) if not isinstance(row, dict) else row
        try:
            ids = json.loads(data.get("source_event_ids_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            ids = []
        return cls(
            episode_id=data["episode_id"],
            session_id=data["session_id"],
            turn_id=data.get("turn_id", ""),
            episode_summary=data["episode_summary"],
            start_timestamp=data["start_timestamp"],
            end_timestamp=data["end_timestamp"],
            source_event_ids=ids,
            frame_id=data.get("frame_id"),
            episode_type=EpisodeType(data["episode_type"]),
            created_at=data.get("created_at") or utcnow(),
        )


class IndexJob(BaseModel):
    """A durable background indexing task (embedding / episode build)."""

    job_id: str = Field(default_factory=lambda: new_id("ij"))
    job_type: str  # "embed_event" | "build_episode" | "rebuild_index"
    source_id: str
    status: str = "pending"  # pending | done | failed | cancelled
    attempts: int = 0
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# retrieval contracts
# ---------------------------------------------------------------------------

class TemporalHint(BaseModel):
    """Parsed temporal / referential intent from a user query."""

    refers_to_history: bool = False
    refers_to_last_observation: bool = False
    refers_to_previous_response: bool = False
    refers_to_recent_turn: bool = False
    entity_hint: Optional[str] = None
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    raw_phrases: list[str] = Field(default_factory=list)


class RetrievedEvent(BaseModel):
    event: ConversationEvent
    score: float = 0.0
    rank_keyword: Optional[int] = None
    rank_semantic: Optional[int] = None
    rank_recent: Optional[int] = None
    source: str = "keyword"  # keyword | semantic | recent | entity


class RetrievedEpisode(BaseModel):
    episode: Episode
    score: float = 0.0
    source: str = "keyword"


class RecallResult(BaseModel):
    """Uniform envelope for episodic recall operations."""

    status: RecallStatus = RecallStatus.NOT_FOUND
    query: str = ""
    events: list[RetrievedEvent] = Field(default_factory=list)
    episodes: list[RetrievedEpisode] = Field(default_factory=list)
    context_block: str = ""
    notes: list[str] = Field(default_factory=list)
    duration_s: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.events or self.episodes)


class RagContextRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = ""
    turn_id: str = ""
    camera_active: bool = False
    entity_hint: Optional[str] = None
    max_events: int = 8
    max_episodes: int = 4
    token_budget: int = 800


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: Optional[str] = None
    event_type: Optional[EventType] = None
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    limit: int = Field(default=10, ge=1, le=100)


class DeleteConversationRequest(BaseModel):
    session_id: str = Field(min_length=1)


class RecordEventAck(BaseModel):
    """Commit acknowledgment — durable only after this is returned."""

    ok: bool = True
    event_id: str = ""
    session_id: str = ""
    duplicate: bool = False
    indexed_immediately: bool = False
    error: Optional[str] = None


def coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None
