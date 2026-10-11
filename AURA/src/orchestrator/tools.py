"""Built-in tools available to the LLM router.

Every handler is a plain Python function returning a :class:`ToolResult`.
There is intentionally no tool that can execute shell commands, write files
or move hardware: physical actuation must be added later behind separate
safety validation, never through the language model.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.orchestrator.schemas import ToolResult, VisualEvidence
from src.orchestrator.tool_registry import (
    AnalyzeCameraArgs,
    DescribeSceneArgs,
    EmptyArgs,
    TextArgs,
    ToolRegistry,
)
from src.utils.logger import get_logger

log = get_logger("builtin_tools")


class CameraArgs(BaseModel):
    capture_mode: str = Field(default="fresh", pattern="^(latest|fresh)$")
    question: str = Field(default="", max_length=400)


class MemoryTranscriptArgs(BaseModel):
    transcript: str = Field(default="", max_length=1000)
    turn_id: str = Field(default="", max_length=80)


class MemoryQueryArgs(BaseModel):
    query: str = Field(default="", max_length=500)
    transcript: str = Field(default="", max_length=1000)


class LearnObjectArgs(BaseModel):
    object_name: str = Field(default="unnamed object", max_length=120)
    transcript: str = Field(default="", max_length=1000)
    capture_mode: str = Field(default="fresh", pattern="^(latest|fresh)$")
    selected_bbox: Optional[list[int]] = None


class RememberIdentityArgs(BaseModel):
    transcript: str = Field(default="", max_length=1000)
    capture_mode: str = Field(default="fresh", pattern="^(latest|fresh)$")
    entity_name: str = Field(default="", max_length=120)
    selected_face_id: Optional[str] = None
    consent_token: Optional[str] = None


class ForgetMemoryArgs(BaseModel):
    transcript: str = Field(default="", max_length=1000)
    entity_name: Optional[str] = Field(default=None, max_length=120)
    fact_id: Optional[str] = Field(default=None, max_length=80)


class RecallArgs(BaseModel):
    query: str = Field(default="", max_length=500)
    transcript: str = Field(default="", max_length=1000)


def build_registry(ctx: "ToolContext") -> ToolRegistry:
    """Create a registry bound to a live :class:`ToolContext`."""
    reg = ToolRegistry()

    @reg.decorator(
        "analyze_camera",
        "Capture a frame from the robot camera and ask the local vision model a question. "
        "Returns the raw observation.",
        args_model=CameraArgs,
        requires_camera=True,
        requires_vlm=True,
        timeout_s=ctx.tool_timeout,
        tags=("perception",),
    )
    def analyze_camera(capture_mode: str = "fresh", question: str = "") -> ToolResult:
        return ctx.analyze_camera(question, capture_mode)

    @reg.decorator(
        "describe_scene",
        "Capture a frame and describe everything visible in it.",
        args_model=DescribeSceneArgs,
        requires_camera=True,
        requires_vlm=True,
        timeout_s=ctx.tool_timeout,
        tags=("perception",),
    )
    def describe_scene(detail: str = "normal") -> ToolResult:
        question = {
            "brief": "List the main objects you can see.",
            "normal": "Describe everything you can see in front of the camera.",
            "detailed": "Describe the scene in detail: objects, their positions, colours and any notable relationships.",
        }.get(detail, "Describe everything you can see in front of the camera.")
        return ctx.analyze_camera(question, "fresh")

    @reg.decorator(
        "capture_frame",
        "Take a camera snapshot and report its metadata (no vision model call).",
        args_model=EmptyArgs,
        requires_camera=True,
        timeout_s=10.0,
        tags=("perception",),
    )
    def capture_frame() -> ToolResult:
        frame = ctx.camera.snapshot()
        return ToolResult(
            tool="capture_frame",
            text=f"captured {frame.tag} ({frame.width}x{frame.height})",
            payload={"frame_id": frame.tag, "age_s": frame.age_s, "timestamp": frame.timestamp},
        )

    @reg.decorator(
        "get_camera_status",
        "Report whether the camera is running, its fps, resolution and last error.",
        args_model=EmptyArgs,
        timeout_s=5.0,
        tags=("system",),
    )
    def get_camera_status() -> ToolResult:
        info = ctx.camera.info()
        text = (
            f"camera status {info['status']}, {info['fps']} fps, {info['resolution']}, "
            f"last frame {info['last_frame_tag'] or 'none'}"
        )
        if info.get("error"):
            text += f", error: {info['error']}"
        return ToolResult(tool="get_camera_status", text=text, payload=info)

    @reg.decorator(
        "get_system_status",
        "Report AURA's system status: models, GPU, mode and queue state.",
        args_model=EmptyArgs,
        timeout_s=5.0,
        tags=("system",),
    )
    def get_system_status() -> ToolResult:
        status = ctx.system_status()
        text = (
            f"mode={status.get('mode')}, state={status.get('state')}, "
            f"models={status.get('models')}, gpu={status.get('gpu')}"
        )
        return ToolResult(tool="get_system_status", text=text, payload=status)

    @reg.decorator(
        "repeat_last_response",
        "Repeat AURA's previous spoken answer verbatim.",
        args_model=EmptyArgs,
        timeout_s=5.0,
        tags=("conversation",),
    )
    def repeat_last_response() -> ToolResult:
        last = ctx.memory.last_response
        if not last:
            return ToolResult(tool="repeat_last_response", text="", payload={"available": False})
        return ToolResult(
            tool="repeat_last_response",
            text=last,
            payload={"available": True, "response": last},
        )

    @reg.decorator(
        "explain_limitation",
        "Explain that AURA cannot perform the requested action yet.",
        args_model=TextArgs,
        timeout_s=5.0,
        tags=("conversation",),
    )
    def explain_limitation(text: str = "") -> ToolResult:
        message = text.strip() or (
            "I cannot do that yet. I can talk with you, look through my camera, "
            "and answer questions about what I see."
        )
        return ToolResult(tool="explain_limitation", text=message, payload={"fixed_reply": True})

    @reg.decorator(
        "respond_directly",
        "Answer the user from language knowledge without using any hardware tool.",
        args_model=EmptyArgs,
        timeout_s=1.0,
        tags=("conversation",),
    )
    def respond_directly() -> ToolResult:
        # Handled by the response stage; registered so the LLM can propose it.
        return ToolResult(tool="respond_directly", text="", payload={"skip_tools": True})

    # -- Multimodal Memory System --------------------------------------
    @reg.decorator(
        "remember_fact",
        "Store a durable fact the user explicitly asked AURA to remember.",
        args_model=MemoryTranscriptArgs,
        timeout_s=ctx.tool_timeout,
        tags=("memory",),
    )
    def remember_fact(transcript: str = "", turn_id: str = "") -> ToolResult:
        return ctx.memory_tool("remember_fact", {"transcript": transcript, "turn_id": turn_id})

    @reg.decorator(
        "correct_memory",
        "Correct a previously stored fact with new information.",
        args_model=MemoryTranscriptArgs,
        timeout_s=ctx.tool_timeout,
        tags=("memory",),
    )
    def correct_memory(transcript: str = "", turn_id: str = "") -> ToolResult:
        return ctx.memory_tool("correct_fact", {"transcript": transcript, "turn_id": turn_id})

    @reg.decorator(
        "remember_identity",
        "Enroll the speaker's face under a name. Requires an explicit consent request.",
        args_model=RememberIdentityArgs,
        requires_camera=True,
        timeout_s=ctx.tool_timeout,
        tags=("memory", "biometric"),
    )
    def remember_identity(
        transcript: str = "",
        capture_mode: str = "fresh",
        entity_name: str = "",
        selected_face_id: Optional[str] = None,
        consent_token: Optional[str] = None,
    ) -> ToolResult:
        return ctx.memory_tool(
            "enroll_person",
            {
                "transcript": transcript,
                "entity_name": entity_name,
                "selected_face_id": selected_face_id,
                "consent_token": consent_token,
                "capture_mode": capture_mode,
            },
        )

    @reg.decorator(
        "learn_visual_object",
        "Capture a frame and remember a named object the user pointed at.",
        args_model=LearnObjectArgs,
        requires_camera=True,
        timeout_s=ctx.tool_timeout,
        tags=("memory",),
    )
    def learn_visual_object(
        object_name: str = "unnamed object",
        transcript: str = "",
        capture_mode: str = "fresh",
        selected_bbox: Optional[list[int]] = None,
    ) -> ToolResult:
        return ctx.memory_tool(
            "enroll_object",
            {
                "object_name": object_name,
                "transcript": transcript,
                "capture_mode": capture_mode,
                "selected_bbox": selected_bbox,
            },
        )

    @reg.decorator(
        "recognize_person",
        "Check whether the visible face matches an enrolled identity.",
        args_model=EmptyArgs,
        requires_camera=True,
        timeout_s=ctx.tool_timeout,
        tags=("memory", "biometric"),
    )
    def recognize_person() -> ToolResult:
        return ctx.memory_tool("recognize_faces", {"capture_mode": "fresh"})

    @reg.decorator(
        "recognize_object",
        "Check whether the visible object matches a previously learned object.",
        args_model=MemoryQueryArgs,
        requires_camera=True,
        timeout_s=ctx.tool_timeout,
        tags=("memory",),
    )
    def recognize_object(query: str = "", transcript: str = "") -> ToolResult:
        return ctx.memory_tool(
            "recognize_objects",
            {"text_query": query or transcript, "capture_mode": "fresh"},
        )

    @reg.decorator(
        "query_memory",
        "Answer a question from stored memories (facts, people, objects).",
        args_model=MemoryQueryArgs,
        timeout_s=ctx.tool_timeout,
        tags=("memory",),
    )
    def query_memory(query: str = "", transcript: str = "") -> ToolResult:
        return ctx.memory_tool("retrieve", {"query": query or transcript})

    @reg.decorator(
        "forget_memory",
        "Delete stored memories at the user's request.",
        args_model=ForgetMemoryArgs,
        timeout_s=ctx.tool_timeout,
        tags=("memory",),
    )
    def forget_memory(transcript: str = "", entity_name: Optional[str] = None, fact_id: Optional[str] = None) -> ToolResult:
        return ctx.memory_tool(
            "delete_memory",
            {"transcript": transcript, "name": entity_name, "fact_id": fact_id},
        )

    @reg.decorator(
        "list_memories",
        "Summarise what AURA currently remembers.",
        args_model=EmptyArgs,
        timeout_s=ctx.tool_timeout,
        tags=("memory",),
    )
    def list_memories() -> ToolResult:
        return ctx.memory_tool("list_memories", {})

    @reg.decorator(
        "recall_memory",
        "Recall something said earlier in the conversation from durable memory. "
        "Works without the camera. Use for 'what did you say earlier' style questions.",
        args_model=RecallArgs,
        timeout_s=ctx.tool_timeout,
        tags=("memory", "recall"),
    )
    def recall_memory(query: str = "", transcript: str = "") -> ToolResult:
        return ctx.memory_tool("recall", {"query": query, "transcript": transcript})

    @reg.decorator(
        "recall_visual_memory",
        "Recall a past visual observation from durable memory (no camera needed). "
        "Use for 'what color was X earlier' or questions about something previously seen.",
        args_model=RecallArgs,
        timeout_s=ctx.tool_timeout,
        tags=("memory", "recall"),
    )
    def recall_visual_memory(query: str = "", transcript: str = "") -> ToolResult:
        return ctx.memory_tool("recall_visual", {"query": query, "transcript": transcript})

    return reg


class ToolContext:
    """Shared services injected into tool handlers."""

    def __init__(
        self,
        camera,
        memory,
        tool_timeout: float = 30.0,
        system_status_fn: Optional[Callable[[], dict]] = None,
        vlm_engine=None,
        memory_engine=None,
    ):
        self.camera = camera
        self.memory = memory
        self.tool_timeout = tool_timeout
        self._system_status_fn = system_status_fn
        self._vlm = vlm_engine
        self.memory_engine = memory_engine
        #: Snapshot taken by the agent right before the PERCEIVING stage, so
        #: the answer is tied to the frame the orchestrator approved.
        self.pending_snapshot = None
        #: Transcript of the turn currently being processed — memory tools
        #: fall back to it when the router omitted `transcript` arguments.
        self.turn_transcript = ""

    # -- services --------------------------------------------------------
    @property
    def vlm(self):
        if self._vlm is None:
            from src.perception.vlm.registry import create_vlm

            self._vlm = create_vlm()
        return self._vlm

    def analyze_camera(self, question: str, capture_mode: str = "fresh") -> ToolResult:
        started = time.perf_counter()
        question = (question or "").strip() or "Describe everything visible in front of the camera."
        pending = getattr(self, "pending_snapshot", None)
        try:
            if pending is not None and pending.age_s <= 1.0:
                frame = pending
            elif capture_mode == "latest":
                frame = self.camera.buffer.latest_fresh()
                if frame is None:
                    frame = self.camera.snapshot(wait_s=1.0)
            else:
                frame = self.camera.snapshot(wait_s=2.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("snapshot failed: %s", exc)
            return ToolResult.failure("analyze_camera", f"camera snapshot failed: {exc}")

        self.pending_snapshot = None
        evidence: VisualEvidence = self.vlm.ask(frame, question)
        self.memory.record_evidence(evidence)

        if not evidence.usable:
            return ToolResult(
                tool="analyze_camera",
                ok=False,
                text="",
                error=evidence.error or "vision model returned no observation",
                payload={"evidence": evidence.model_dump()},
                duration_s=time.perf_counter() - started,
            )

        age = f", image age {evidence.image_age_s:.2f}s" if evidence.image_age_s is not None else ""
        return ToolResult(
            tool="analyze_camera",
            ok=True,
            text=evidence.raw_answer,
            payload={"evidence": evidence.model_dump(), "frame_id": evidence.frame_id},
            duration_s=time.perf_counter() - started,
        )

    def system_status(self) -> dict[str, Any]:
        if self._system_status_fn:
            try:
                return self._system_status_fn()
            except Exception as exc:  # noqa: BLE001
                log.warning("system status provider failed: %s", exc)
        from src.utils.device_manager import gpu_snapshot

        return {
            "platform": platform.node(),
            "gpu": gpu_snapshot().get("nvsmi", {}),
            "mode": "unknown",
            "state": "unknown",
            "models": {},
        }

    # -- memory --------------------------------------------------------
    def _memory_snapshot(self, capture_mode: str = "fresh"):
        """Return a fresh-enough Frame for a memory vision operation."""
        pending = getattr(self, "pending_snapshot", None)
        if pending is not None and pending.age_s <= 5.0:
            self.pending_snapshot = None
            return pending
        if self.camera is None:
            return None
        if capture_mode == "latest":
            frame = self.camera.buffer.latest_fresh()
            if frame is not None:
                return frame
        return self.camera.snapshot(wait_s=2.0)

    def memory_tool(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Dispatch a memory operation through the MMS facade."""
        from src.memory.exceptions import MemoryError
        from src.memory.models import (
            CorrectFactRequest,
            EnrollObjectRequest,
            EnrollPersonRequest,
            ForgetRequest,
            RememberFactRequest,
        )

        engine = self.memory_engine
        if engine is None or not getattr(engine, "enabled", False):
            return ToolResult.failure(
                operation, "multimodal memory is not enabled or not initialized"
            )

        started = time.perf_counter()

        def _transcript() -> str:
            # The router may omit `transcript` in tool_arguments (LLM plans);
            # fall back to the turn the agent is actually processing.
            return (arguments.get("transcript") or "").strip() or (
                getattr(self, "turn_transcript", "") or ""
            ).strip()

        try:
            if operation == "remember_fact":
                transcript = _transcript()
                if not transcript:
                    return ToolResult.failure(
                        operation, "no transcript available to remember"
                    )
                result = engine.remember_fact(
                    RememberFactRequest(
                        transcript=transcript,
                        turn_id=arguments.get("turn_id", ""),
                    )
                )
            elif operation == "correct_fact":
                transcript = _transcript()
                if not transcript:
                    return ToolResult.failure(
                        operation, "no transcript available to correct"
                    )
                result = engine.correct_fact(
                    CorrectFactRequest(
                        transcript=transcript,
                        turn_id=arguments.get("turn_id", ""),
                    )
                )
            elif operation in {"enroll_person", "recognize_faces"}:
                frame = self._memory_snapshot(arguments.get("capture_mode", "fresh"))
                if frame is None:
                    return ToolResult.failure(operation, "camera snapshot failed")
                if operation == "enroll_person":
                    name = (arguments.get("entity_name") or "").strip()
                    if not name:
                        name = _name_from_transcript(_transcript())
                    if not name:
                        return ToolResult.failure(
                            operation, "could not determine a name for enrollment"
                        )
                    result = engine.enroll_person(
                        EnrollPersonRequest(
                            entity_name=name,
                            frame=frame.image if hasattr(frame, "image") else frame,
                            snapshot=None,
                            selected_face_id=arguments.get("selected_face_id"),
                            consent_token=arguments.get("consent_token"),
                            consent_context={"transcript": _transcript()},
                        )
                    )
                else:
                    result = engine.recognize_faces(
                        EnrollPersonRequest(
                            entity_name="",
                            frame=frame.image if hasattr(frame, "image") else frame,
                        )
                    )
            elif operation == "enroll_object":
                frame = self._memory_snapshot(arguments.get("capture_mode", "fresh"))
                if frame is None:
                    return ToolResult.failure(operation, "camera snapshot failed")
                result = engine.enroll_object(
                    EnrollObjectRequest(
                        object_name=arguments.get("object_name", "unnamed object"),
                        frame=frame.image if hasattr(frame, "image") else frame,
                        text_query=arguments.get("transcript") or arguments.get("object_name"),
                    )
                )
            elif operation == "recognize_objects":
                frame = self._memory_snapshot(arguments.get("capture_mode", "fresh"))
                if frame is None:
                    return ToolResult.failure(operation, "camera snapshot failed")
                frame_image = frame.image if hasattr(frame, "image") else frame
                text_query = arguments.get("text_query") or "the object in view"

                class _Req:
                    pass

                req = _Req()
                req.frame = frame_image
                req.text_query = text_query
                result = engine.recognize_objects(req)
            elif operation == "retrieve":
                offline = bool(getattr(engine, "_offline_hint", False))
                query = (arguments.get("query") or "").strip() or _transcript()
                result = engine.retrieve(query, offline=offline)
            elif operation == "delete_memory":
                name = arguments.get("name")
                fact_id = arguments.get("fact_id")
                transcript = _transcript()
                if not name and not fact_id:
                    name = _name_from_forget_transcript(transcript)
                    if not name and "face" in transcript.lower():
                        # List people and delete the single enrolled person
                        listing = engine.list_memories()
                        people = (listing.payload or {}).get("enrolled_people") or []
                        if len(people) == 1:
                            name = people[0].get("name")
                result = engine.delete_memory(
                    ForgetRequest(name=name, fact_id=fact_id, entity_id=None)
                )
            elif operation == "list_memories":
                result = engine.list_memories()
            elif operation == "recall":
                query = (arguments.get("query") or "").strip() or _transcript()
                if not query:
                    return ToolResult.failure(operation, "no query available to recall")
                result = engine.search_conversations(query)
            elif operation == "recall_visual":
                query = (arguments.get("query") or "").strip() or _transcript()
                result = engine.recall_last_visual_observation(query)
            else:
                return ToolResult.failure(operation, f"unknown memory operation {operation}")

            payload = dict(result.payload or {})
            payload["operation"] = result.operation
            text = result.message
            if not result.ok:
                return ToolResult(
                    tool=operation if operation != "correct_fact" else "correct_memory",
                    ok=False,
                    text="",
                    error=result.message,
                    payload={"error_code": result.error_code, **payload},
                    duration_s=time.perf_counter() - started,
                )
            tool_name = {
                "remember_fact": "remember_fact",
                "correct_fact": "correct_memory",
                "enroll_person": "remember_identity",
                "recognize_faces": "recognize_person",
                "enroll_object": "learn_visual_object",
                "recognize_objects": "recognize_object",
                "retrieve": "query_memory",
                "delete_memory": "forget_memory",
                "list_memories": "list_memories",
                "recall": "recall_memory",
                "recall_visual": "recall_visual_memory",
            }.get(operation, operation)
            return ToolResult(
                tool=tool_name,
                ok=True,
                text=text,
                payload=payload,
                duration_s=time.perf_counter() - started,
            )
        except MemoryError as exc:
            return ToolResult.failure(operation, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("memory tool %s crashed", operation)
            return ToolResult.failure(operation, f"memory operation failed: {exc}")


def _name_from_transcript(transcript: str) -> str:
    import re

    m = re.search(r"\b(?:as|named|called)\s+([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)", transcript or "")
    if m:
        return m.group(1).strip()[:120]
    m = re.search(r"\bmy name is\s+([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)", transcript or "", re.IGNORECASE)
    if m:
        return m.group(1).strip()[:120]
    return ""


def _name_from_forget_transcript(transcript: str) -> str:
    import re

    m = re.search(
        r"\b(?:forget|delete|remove)\s+(?:the\s+)?(?:memory\s+)?(?:for\s+|about\s+)?([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)",
        transcript or "",
    )
    if m:
        candidate = m.group(1).strip()
        if candidate.lower() not in {"my", "the", "that", "this", "everything", "all"}:
            return candidate[:120]
    return ""
