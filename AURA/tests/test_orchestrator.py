"""Orchestrator unit tests: state machine, routing schema, tool authorization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.orchestrator.schemas import (
    Intent,
    RoutePlan,
    validate_route_plan,
)
from src.orchestrator.state_machine import ALLOWED, InvalidTransition, State, StateMachine
from src.orchestrator.tool_registry import EmptyArgs, Tool, ToolRegistry
from src.orchestrator.schemas import ToolResult
from src.utils.exceptions import InvalidToolCallError
from tests.conftest import make_agent, plan_dict


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_initial_state_is_idle(self):
        assert StateMachine().state == State.IDLE

    def test_happy_path_visual(self):
        sm = StateMachine()
        for target in (
            State.LISTENING, State.TRANSCRIBING, State.INTERPRETING,
            State.CAPTURING, State.PERCEIVING, State.GENERATING,
            State.SPEAKING, State.IDLE,
        ):
            sm.transition(target, "test")
        assert sm.state == State.IDLE
        assert [t.current for t in sm.history][-1] is State.IDLE

    def test_happy_path_general_skips_perception(self):
        sm = StateMachine()
        for target in (State.LISTENING, State.TRANSCRIBING, State.INTERPRETING,
                       State.GENERATING, State.SPEAKING, State.IDLE):
            sm.transition(target)
        states = [t.current for t in sm.history]
        assert State.CAPTURING not in states
        assert State.PERCEIVING not in states

    @pytest.mark.parametrize(
        "start,target",
        [
            (State.IDLE, State.SPEAKING),
            (State.IDLE, State.CAPTURING),
            (State.LISTENING, State.SPEAKING),
            (State.GENERATING, State.CAPTURING),
            (State.SHUTDOWN, State.IDLE),
        ],
    )
    def test_invalid_transitions_rejected(self, start, target):
        sm = StateMachine(start)
        with pytest.raises(InvalidTransition):
            sm.transition(target)
        assert sm.state is start

    def test_every_active_state_can_reach_error_or_shutdown(self):
        for state, allowed in ALLOWED.items():
            if state in {State.SHUTDOWN, State.ERROR}:
                continue
            assert State.SHUTDOWN in allowed, f"{state} cannot shut down"
            if state is not State.IDLE:
                assert State.ERROR in allowed, f"{state} cannot report an error"

    def test_error_and_shutdown_are_terminal_enough(self):
        assert ALLOWED[State.SHUTDOWN] == frozenset()
        assert ALLOWED[State.ERROR] == frozenset({State.IDLE, State.SHUTDOWN})

    def test_error_returns_to_idle(self):
        sm = StateMachine(State.GENERATING)
        sm.transition(State.ERROR, "boom")
        sm.transition(State.IDLE, "recovered")
        assert sm.state is State.IDLE

    def test_try_transition_does_not_raise(self):
        sm = StateMachine()
        assert sm.try_transition(State.SPEAKING) is False
        assert sm.state is State.IDLE

    def test_transition_listeners_fire(self):
        sm = StateMachine()
        seen = []
        sm.on_transition(lambda tr: seen.append(tr.current))
        sm.transition(State.LISTENING, "x")
        assert seen == [State.LISTENING]


# ---------------------------------------------------------------------------
# Route plan schema
# ---------------------------------------------------------------------------

class TestRoutePlan:
    def test_valid_plan(self):
        plan = validate_route_plan(
            plan_dict("visual_question", tool="analyze_camera",
                      normalized_question="What is visible?",
                      tool_arguments={"question": "What is visible?"})
        )
        assert plan.is_visual
        assert plan.requires_camera

    def test_vlm_requires_camera_enforced(self):
        plan = RoutePlan.model_validate(
            {"intent": "visual_question", "requires_vlm": True, "requires_camera": False}
        )
        assert plan.requires_camera is True

    def test_visual_intent_gets_camera_tool(self):
        plan = RoutePlan.model_validate(
            {"intent": "scene_description", "tool": "respond_directly",
             "requires_camera": False, "requires_vlm": False}
        )
        assert plan.tool == "describe_scene"

    def test_unknown_intent_rejected(self):
        with pytest.raises(ValidationError):
            RoutePlan.model_validate({"intent": "launch_missiles", "tool": "respond_directly"})

    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValidationError):
            RoutePlan.model_validate({"intent": "greeting", "confidence": 3.1})

    def test_non_dict_input_rejected(self):
        with pytest.raises(ValidationError):
            validate_route_plan(42)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def _registry(self, enabled: bool = True) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(
            Tool(
                name="echo",
                description="echo",
                args_model=EmptyArgs,
                handler=lambda: ToolResult(tool="echo", text="pong"),
                enabled=enabled,
            )
        )
        return reg

    def test_registered_tool_runs(self):
        result = self._registry().execute("echo")
        assert result.ok and result.text == "pong"

    def test_unknown_tool_refused(self):
        with pytest.raises(InvalidToolCallError):
            self._registry().execute("rm_rf")

    def test_disabled_tool_refused(self):
        with pytest.raises(InvalidToolCallError):
            self._registry(enabled=False).execute("echo")

    def test_invalid_arguments_refused(self):
        from pydantic import BaseModel, Field

        reg = ToolRegistry()

        class NeedArgs(BaseModel):
            target: str = Field(min_length=1)

        reg.register(
            Tool("needs", "d", NeedArgs, lambda target: ToolResult(tool="needs"))
        )
        with pytest.raises(InvalidToolCallError):
            reg.execute("needs", {})              # missing required field
        with pytest.raises(InvalidToolCallError):
            reg.execute("needs", {"target": ""})  # empty field

    def test_handler_exception_becomes_failed_result(self):
        reg = ToolRegistry()

        def boom():
            raise RuntimeError("kaput")

        reg.register(Tool("boom", "d", EmptyArgs, boom))
        result = reg.execute("boom")
        assert result.ok is False and "kaput" in (result.error or "")

    def test_timeout_returns_failed_result(self):
        import time as _time

        reg = ToolRegistry()

        def slow():
            _time.sleep(2)
            return ToolResult(tool="slow")

        reg.register(Tool("slow", "d", EmptyArgs, slow, timeout_s=0.2))
        result = reg.execute("slow")
        assert result.ok is False and "timed out" in (result.error or "")

    def test_authorization_blocks_camera_tools_without_plan(self):
        reg = ToolRegistry()
        reg.register(
            Tool("analyze_camera", "d", EmptyArgs, lambda: ToolResult(tool="analyze_camera"),
                 requires_camera=True)
        )
        assert reg.authorize("analyze_camera", plan_requires_camera=False) is False
        assert reg.authorize("analyze_camera", plan_requires_camera=True) is True

    def test_openai_schema_shape(self):
        schema = self._registry().openai_tools()[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert "parameters" in schema["function"]


# ---------------------------------------------------------------------------
# Deterministic router
# ---------------------------------------------------------------------------

class TestDeterministicRouter:
    def _route(self, text: str):
        from src.intelligence.intent_classifier import DeterministicRouter

        return DeterministicRouter().route(text)

    @pytest.mark.parametrize(
        "text,intent",
        [
            ("What do you see in front of you?", "visual_question"),
            ("Describe the room", "scene_description"),
            ("What's on the table?", "visual_question"),
            ("Hello robot!", "greeting"),
            ("Explain gravity", "general_question"),
            ("Repeat your answer", "repeat_response"),
            ("Drive forward two metres", "future_capability"),
            ("Take a photo of us", "unsupported"),
        ],
    )
    def test_routing(self, text, intent):
        decision = self._route(text)
        assert decision.plan.intent.value == intent, f"{text!r} -> {decision.plan}"

    def test_visual_route_requires_camera(self):
        assert self._route("What do you see?").plan.requires_camera is True

    def test_general_route_never_touches_camera(self):
        plan = self._route("Explain gravity").plan
        assert plan.requires_camera is False and plan.requires_vlm is False

    def test_movement_requests_are_refused_locally(self):
        plan = self._route("Turn left and move forward").plan
        assert plan.tool == "explain_limitation"

    def test_ambiguous_reference_is_unconfident(self):
        decision = self._route("what about the other one?")
        assert decision.error or decision.plan.confidence < 0.9

    def test_empty_transcript(self):
        decision = self._route("")
        assert decision.plan.intent is Intent.UNSUPPORTED


# ---------------------------------------------------------------------------
# Agent-level routing
# ---------------------------------------------------------------------------

class TestAgentRouting:
    def test_greeting_never_calls_camera(self):
        agent = make_agent(plan=plan_dict("greeting"), answer="Hello!")
        agent.start(warm_models=False)
        try:
            record = agent.handle_text("Hello robot", speak=False)
        finally:
            agent.shutdown()
        assert record.plan.requires_camera is False
        assert not record.evidence
        assert agent.camera.snapshots == 0

    def test_visual_request_captures_and_perceives(self):
        agent = make_agent(
            plan=plan_dict("visual_question", tool="analyze_camera",
                           normalized_question="What is visible?",
                           tool_arguments={"question": "What is visible?"}),
            answer="I can see a test image.",
        )
        agent.start(warm_models=False)
        try:
            record = agent.handle_text("What do you see?", speak=False)
        finally:
            agent.shutdown()
        assert agent.camera.snapshots >= 1
        assert record.evidence and record.evidence[0].frame_id
        assert record.evidence[0].inference_time_ms is not None

    def test_llm_failure_falls_back_to_local_plan(self):
        from src.intelligence.groq_client import CloudServiceError

        agent = make_agent(fail_client=CloudServiceError("boom"))
        agent.start(warm_models=False)
        try:
            record = agent.handle_text("What do you see in front of you?", speak=False)
        finally:
            agent.shutdown()
        assert record.error or record.response
        assert record.plan is not None
