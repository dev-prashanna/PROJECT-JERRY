"""Public data contracts for the Multimodal Memory System.

All service-level inputs and outputs are Pydantic models so tool handlers,
the GUI and tests share one validated schema.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class EntityType(str, Enum):
    PERSON = "person"
    OBJECT = "object"
    CONCEPT = "concept"
    LOCATION = "location"
    PROJECT = "project"


class EntityStatus(str, Enum):
    ACTIVE = "active"
    PROVISIONAL = "provisional"
    ARCHIVED = "archived"


class FactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"


class FactSourceType(str, Enum):
    USER_ASSERTION = "user_assertion"
    OBSERVATION = "observation"
    INFERENCE = "inference"


class MatchStatus(str, Enum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class MemoryEntity(BaseModel):
    entity_id: str = Field(default_factory=lambda: new_id("ent"))
    canonical_name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    status: EntityStatus = EntityStatus.ACTIVE
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_seen_at: Optional[datetime] = None


class SnapshotReference(BaseModel):
    frame_id: str
    captured_at: datetime = Field(default_factory=utcnow)
    camera_id: str = "camera0"
    width: int = 0
    height: int = 0
    image_reference: Optional[str] = None
    request_id: str = ""


class FaceObservation(BaseModel):
    observation_id: str = Field(default_factory=lambda: new_id("fobs"))
    frame_id: str
    bbox_xyxy: tuple[int, int, int, int]
    embedding: Optional[list[float]] = None
    detection_quality: Optional[float] = None
    source_model: str = ""
    timestamp: datetime = Field(default_factory=utcnow)


class ObjectObservation(BaseModel):
    observation_id: str = Field(default_factory=lambda: new_id("oobs"))
    frame_id: str
    bbox_xyxy: tuple[int, int, int, int]
    segmentation_mask_reference: Optional[str] = None
    visual_embedding: Optional[list[float]] = None
    embedding_model_id: str = ""
    embedding_dimension: int = 0
    candidate_label: Optional[str] = None
    grounding_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=utcnow)


class MemoryFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: new_id("fact"))
    subject_entity_id: Optional[str] = None
    subject_alias: str = ""
    predicate: str
    object_value: str
    value_type: str = "text"
    source_transcript: str = ""
    source_turn_id: str = ""
    source_type: FactSourceType = FactSourceType.USER_ASSERTION
    asserted_by_entity_id: Optional[str] = None
    status: FactStatus = FactStatus.ACTIVE
    supersedes_fact_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    extraction_method: str = "deterministic"


class MatchResult(BaseModel):
    """Modality-specific match outcome.

    ``similarity_score`` is a raw metric (cosine for objects, inverse
    distance for faces) — it is NOT a calibrated probability.
    """

    entity_id: Optional[str] = None
    status: MatchStatus = MatchStatus.UNKNOWN
    raw_distance: Optional[float] = None
    similarity_score: Optional[float] = None
    second_best_score: Optional[float] = None
    embedding_model_id: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    entity_name: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class EntityAssociation(BaseModel):
    association_id: str = Field(default_factory=lambda: new_id("assoc"))
    source_entity_id: str
    target_entity_id: str
    relation: str = "related_to"
    weight: float = 1.0
    created_at: datetime = Field(default_factory=utcnow)
    source_transcript: str = ""
    source_turn_id: str = ""


class ConsentRecord(BaseModel):
    consent_id: str = Field(default_factory=lambda: new_id("consent"))
    subject_description: str
    scope: str = "face_enrollment"
    granted: bool = False
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    method: str = "explicit_utterance"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    query_id: str = Field(default_factory=lambda: new_id("q"))
    matched_entities: list[MemoryEntity] = Field(default_factory=list)
    retrieved_facts: list[MemoryFact] = Field(default_factory=list)
    visual_matches: list[MatchResult] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)


class EnrollPersonRequest(BaseModel):
    entity_name: str = Field(min_length=1, max_length=120)
    frame: Any = None
    snapshot: Optional[SnapshotReference] = None
    selected_face_id: Optional[str] = None
    consent_token: Optional[str] = None
    consent_context: dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


class EnrollObjectRequest(BaseModel):
    object_name: str = Field(min_length=1, max_length=120)
    frame: Any = None
    text_query: Optional[str] = None
    request_id: str = ""


class RecognizeObjectsRequest(BaseModel):
    frame: Any = None
    text_query: Optional[str] = None
    request_id: str = ""


class RememberFactRequest(BaseModel):
    transcript: str = Field(min_length=1)
    turn_id: str = ""
    speaker_entity_id: Optional[str] = None
    use_llm: bool = False
    request_id: str = ""


class CorrectFactRequest(BaseModel):
    transcript: str = Field(min_length=1)
    turn_id: str = ""
    fact_id: Optional[str] = None
    request_id: str = ""


class ForgetRequest(BaseModel):
    entity_id: Optional[str] = None
    fact_id: Optional[str] = None
    name: Optional[str] = None
    request_id: str = ""


class MemoryOperationResult(BaseModel):
    """Uniform envelope for every public engine operation."""

    ok: bool = True
    operation: str = ""
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    error_code: Optional[str] = None
    retryable: bool = False
    duration_s: float = 0.0

    @classmethod
    def failure(
        cls,
        operation: str,
        error_code: str,
        message: str,
        *,
        retryable: bool = False,
        payload: Optional[dict[str, Any]] = None,
    ) -> "MemoryOperationResult":
        return cls(
            ok=False,
            operation=operation,
            message=message,
            error_code=error_code,
            retryable=retryable,
            payload=payload or {},
        )


def parse_dt(value: Any) -> Optional[datetime]:
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


def dt_to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def normalize_name(name: str) -> str:
    return " ".join(str(name or "").lower().split())
