"""Strict allow-list of tools the LLM may invoke.

Rules enforced here (never in the model):

* Only registered tools can run.
* Arguments must validate against the tool's pydantic input model.
* Each execution is time-boxed.
* Results are structured and serialisable.
* There is no path from an LLM string to a shell command, a file write, or a
  motor actuator.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from src.orchestrator.schemas import ToolResult
from src.utils.exceptions import InvalidToolCallError
from src.utils.logger import get_logger

log = get_logger("tools")


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class AnalyzeCameraArgs(BaseModel):
    question: str = Field(min_length=1, max_length=400)
    capture_mode: str = Field(default="latest", pattern="^(latest|fresh)$")


class DescribeSceneArgs(BaseModel):
    detail: str = Field(default="normal", pattern="^(brief|normal|detailed)$")


class EmptyArgs(BaseModel):
    pass


class TextArgs(BaseModel):
    text: str = Field(default="", max_length=2000)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., ToolResult]
    timeout_s: float = 30.0
    requires_camera: bool = False
    requires_vlm: bool = False
    enabled: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()

    # -- registration ---------------------------------------------------
    def register(self, tool: Tool) -> None:
        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"tool already registered: {tool.name}")
            self._tools[tool.name] = tool

    def decorator(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel] = EmptyArgs,
        timeout_s: float = 30.0,
        requires_camera: bool = False,
        requires_vlm: bool = False,
        tags: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
        def wrap(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    args_model=args_model,
                    handler=fn,
                    timeout_s=timeout_s,
                    requires_camera=requires_camera,
                    requires_vlm=requires_vlm,
                    tags=tags,
                )
            )
            return fn

        return wrap

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    # -- introspection ---------------------------------------------------
    def get(self, name: str) -> Optional[Tool]:
        with self._lock:
            return self._tools.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def enabled_tools(self) -> list[Tool]:
        with self._lock:
            return [t for t in self._tools.values() if t.enabled]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.json_schema() for t in self.enabled_tools()]

    def enable(self, name: str, flag: bool = True) -> None:
        with self._lock:
            tool = self._tools.get(name)
            if tool:
                tool.enabled = flag

    # -- execution -------------------------------------------------------
    def execute(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
        *,
        default_timeout_s: float = 30.0,
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            raise InvalidToolCallError(f"unknown tool '{name}'")
        if not tool.enabled:
            raise InvalidToolCallError(f"tool '{name}' is disabled")

        try:
            args: BaseModel = tool.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            raise InvalidToolCallError(
                f"invalid arguments for '{name}': {exc.errors()[0].get('msg')}",
            ) from exc

        started = time.perf_counter()
        timeout = tool.timeout_s or default_timeout_s
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, Any] = {}
        done = threading.Event()

        def runner() -> None:
            try:
                result_holder["value"] = tool.handler(**args.model_dump())
            except BaseException as exc:  # noqa: BLE001
                error_holder["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=runner, name=f"tool:{name}", daemon=True)
        thread.start()
        finished = done.wait(timeout)
        elapsed = time.perf_counter() - started

        if not finished:
            log.error("tool %s exceeded %.1fs timeout", name, timeout)
            return ToolResult.failure(name, f"tool '{name}' timed out after {timeout:.0f}s")

        if "error" in error_holder:
            exc = error_holder["error"]
            log.warning("tool %s failed: %s", name, exc)
            return ToolResult.failure(name, str(exc))

        result = result_holder.get("value")
        if not isinstance(result, ToolResult):
            result = ToolResult(tool=name, text=str(result), payload={"value": result})
        result.duration_s = elapsed
        return result

    # -- authorization checks -------------------------------------------
    def authorize(self, tool_name: str, plan_requires_camera: bool) -> bool:
        """Refuse tool calls that exceed what the validated plan allows."""
        tool = self.get(tool_name)
        if tool is None or not tool.enabled:
            return False
        if tool.requires_camera and not plan_requires_camera:
            log.warning("refused tool %s: plan does not authorise camera access", tool_name)
            return False
        return True
