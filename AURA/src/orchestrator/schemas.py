"""Schema-validated data contracts between the LLM and the orchestrator.

The LLM only ever *proposes* a plan.  Every proposal is validated here and
again by :mod:`src.orchestrator.tool_registry` before anything executes.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class Intent(str, Enum):
    GREETING = "greeting"
    GENERAL_QUESTION = "general_question"
    VISUAL_QUESTION = "visual_question"
    SCENE_DESCRIPTION = "scene_description"
    OBJECT_REASONING = "object_reasoning"
    CLARIFICATION = "clarification"
    REPEAT_RESPONSE = "repeat_response"
    CAMERA_CONTROL = "camera_control"
    SYSTEM_STATUS = "system_status"
    FUTURE_CAPABILITY = "future_capability"
    UNSUPPORTED = "unsupported"


#: Intents that require a fresh camera frame + VLM inference.
VISUAL_INTENTS: frozenset[Intent] = frozenset(
    {Intent.VISUAL_QUESTION, Intent.SCENE_DESCRIPTION, Intent.OBJECT_REASONING}
)

#: Intents that never need the cloud LLM for the *routing* step alone.
NO_TOOL_INTENTS: frozenset[Intent] = frozenset(
    {Intent.GREETING, Intent.GENERAL_QUESTION, Intent.REPEAT_RESPONSE}
)


class RoutePlan(BaseModel):
    """The structured task proposal produced by the LLM router."""

    intent: Intent
    requires_camera: bool = False
    requires_vlm: bool = False
    normalized_question: str = Field(default="", max_length=500)
    tool: str = "respond_directly"
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=300)
    needs_offline_notice: bool = False

    @field_validator("normalized_question", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return " ".join(str(v or "").split())

    @property
    def is_visual(self) -> bool:
        return self.intent in VISUAL_INTENTS or self.requires_vlm

    def model_post_init(self, __context: Any) -> None:
        # Consistency enforcement: a VLM request must request the camera.
        if self.requires_vlm:
            self.requires_camera = True
        if self.intent in VISUAL_INTENTS and self.tool in {"respond_directly", "chat"}:
            self.tool = "describe_scene" if self.intent is Intent.SCENE_DESCRIPTION else "analyze_camera"


class VisualEvidence(BaseModel):
    """An imperfect observation produced by a local perception model."""

    source: str = "moondream2"
    frame_id: str
    question: str
    raw_answer: str
    status: str = "completed"          # completed | failed | unavailable
    uncertainty_notes: list[str] = Field(default_factory=list)
    inference_time_ms: Optional[float] = None
    frame_timestamp: Optional[float] = None
    image_age_s: Optional[float] = None
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.status == "completed" and bool(self.raw_answer.strip())


class ToolResult(BaseModel):
    tool: str
    ok: bool = True
    payload: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0

    @classmethod
    def failure(cls, tool: str, error: str) -> "ToolResult":
        return cls(tool=tool, ok=False, error=error)


class ConversationTurn(BaseModel):
    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    user_transcript: str = ""
    intent: Optional[str] = None
    frame_id: Optional[str] = None
    visual_observation: Optional[str] = None
    final_response: str = ""
    timestamp: float = Field(default_factory=time.time)
    route_tool: Optional[str] = None
    error: Optional[str] = None


class RequestRecord(BaseModel):
    """Everything the orchestrator knows about one user request."""

    request_id: str = Field(default_factory=lambda: new_id("req"))
    transcript: str = ""
    plan: Optional[RoutePlan] = None
    evidence: list[VisualEvidence] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    response: str = ""
    mode: str = "online"               # online | offline
    response_strategy: str = "full"
    metadata: dict = Field(default_factory=dict)
    started_at: float = Field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def evidence_text(self) -> str:
        return "\n".join(e.raw_answer for e in self.evidence if e.usable)


def validate_route_plan(raw: Any) -> RoutePlan:
    """Validate an LLM proposal; raise ``ValidationError`` when malformed."""
    if isinstance(raw, RoutePlan):
        return raw
    # model_validate raises a proper pydantic ValidationError for any type.
    return RoutePlan.model_validate(raw)


# Canonical plan produced when the cloud LLM is unavailable.
FALLBACK_GENERAL = dict(
    intent=Intent.GENERAL_QUESTION,
    requires_camera=False,
    requires_vlm=False,
    tool="respond_directly",
    tool_arguments={},
    confidence=0.0,
    rationale="offline fallback: cloud router unavailable",
    needs_offline_notice=True,
)

FALLBACK_VISUAL = dict(
    intent=Intent.VISUAL_QUESTION,
    requires_camera=True,
    requires_vlm=True,
    tool="analyze_camera",
    tool_arguments={},
    confidence=0.0,
    rationale="offline fallback: visual request routed locally",
    needs_offline_notice=True,
)
