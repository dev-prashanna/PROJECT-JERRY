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
    ):
        self.camera = camera
        self.memory = memory
        self.tool_timeout = tool_timeout
        self._system_status_fn = system_status_fn
        self._vlm = vlm_engine
        #: Snapshot taken by the agent right before the PERCEIVING stage, so
        #: the answer is tied to the frame the orchestrator approved.
        self.pending_snapshot = None

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
