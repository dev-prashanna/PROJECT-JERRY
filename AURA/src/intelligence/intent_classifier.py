"""Intent classification / task routing.

Two interchangeable routers:

* :class:`LLMRouter`      – cloud Groq model, schema-validated (baseline).
* :class:`DeterministicRouter` – local, free, high-precision keyword routing.

``router_mode`` (setting ``ROUTER_MODE``):

* ``llm``           – cloud router only (default, best on ambiguous input).
* ``deterministic`` – local only, used automatically when offline.
* ``auto``          – deterministic router when it is confident, else cloud.
* ``compare``       – run both, log the disagreement, use the cloud answer.

Correctness always wins over saving one API call: the deterministic router
only claims a route when its precision-critical patterns match exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from config.settings import get_settings
from src.intelligence.groq_client import GroqClient  # noqa: F401 - re-exported
from src.intelligence.llm.base import BaseLLMClient
from src.intelligence.llm.registry import create_llm_client
from src.intelligence.prompt_builder import build_router_messages
from src.orchestrator.schemas import (
    FALLBACK_GENERAL,
    FALLBACK_VISUAL,
    Intent,
    RoutePlan,
    validate_route_plan,
)
from src.utils.exceptions import CloudServiceError
from src.utils.logger import get_logger

log = get_logger("router")


@dataclass
class RouteDecision:
    plan: RoutePlan
    router: str
    latency_s: float = 0.0
    alternatives: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Deterministic router
# ---------------------------------------------------------------------------

_VISUAL_PATTERNS = [
    r"\bwhat (do|can) (you|i) (see|observe|spot)\b",
    r"\bwhat'?s (in|on) (front of|front|the|this|that|my)\b",
    r"\bdescribe (the|this|that|your)?\s*(room|scene|surroundings|view|area|desk|table)\b",
    r"\blook (at|in front)\b",
    r"\bin front of (me|you)\b",
    r"\bwhat objects?\b.*\b(visible|see|there)\b",
    r"\bcan you see\b",
    r"\bhow many .* (are|is) (there|visible)\b",
    r"\bwhat'?s on (the|this|that) (table|desk|floor|counter|shelf|bed)\b",
]

_SCENE_PATTERNS = [
    r"^(describe|explain|tell me about) (the|this|that) (scene|room|view|surroundings|image)\b",
    r"\bwhat does (the|this|that) (room|scene|area) look like\b",
]

_OBJECT_PATTERNS = [
    r"\bwhat colou?r (is|was|are|were)\b",
    r"\bwhere is\b",
    r"\bwhich (one|object|thing|item)\b",
    r"\bis there (a|an|any)\b",
    r"\bwhat (brand|make|model|type)\b",
    r"\bthe (bottle|cup|phone|laptop|book|chair|person|object|item) (on|in|left|right|middle|front|back)\b",
]

_GREETING_PATTERNS = [
    r"^\s*(hey|hi|hello|yo|good (morning|afternoon|evening))\b",
    r"^\s*(hey|hi|hello)[,!\s]+(robot|aura|there)?\s*[.!?]*\s*$",
    r"^(how are you|who are you|what'?s your name)\b",
]

_STATUS_PATTERNS = [
    r"\b(camera|gpu|system|model) status\b",
    r"\bhow (are|is) you doing\b",
    r"\bwhat (are|is) your (capabilities|status)\b",
]

_REPEAT_PATTERNS = [
    r"\b(repeat|say (that|it) again|come again)\b",
    r"\bwhat did you (say|just say)\b",
]

_FUTURE_PATTERNS = [
    r"\b(move|drive|go forward|turn (left|right)|stop moving|back up|rotate)\b",
    r"\b(pick up|grab|lift|open the door)\b",
    r"\b(run|execute) (this|that|a) (script|command|program)\b",
    r"\b(shell|terminal|bash|sudo|rm -|python -c)\b",
]

_LIMITATION_PATTERNS = [
    r"\b(take a (photo|picture|selfie)|record (a )?video|upload (this|the) image)\b",
]

# -- Multimodal Memory System (checked before visual patterns) ------------

_LEARN_IDENTITY_PATTERNS = [
    r"\b(please\s+)?(remember|learn|save|store|enroll)(\s+my)?\s+(face|identity|face under)\b",
    r"\b(remember|learn)\s+me\s+as\b",
    r"\bmy name is .{1,40}\b(remember|save|learn)\b",
    r"\b(remember|learn|save)\b.{0,20}\b(my face|my identity)\b",
    r"\benroll me\b",
]

_LEARN_OBJECT_PATTERNS = [
    r"\b(remember|learn|save|store|note)\s+(this|that|the)\s+\w+\s+as\b",
    r"\b(remember|learn|save|store)\s+(this|that)\s+(object|device|thing|item|board|circuit)\b",
    r"\bcall\s+(this|that)\s+\w*\s*\w*\s*(my|the)\b",
    r"\bremember\s+this\s+as\b",
    r"\bsave\s+this\s+object\s+as\b",
]

_REMEMBER_FACT_PATTERNS = [
    r"\b(please\s+)?(remember|note|save|store|memorize)(\s+that)?\b",
    r"\bmy name is\b",
    r"\bi am\b(?!\s+(not|going|trying))",
    r"\bi'?m\b(?=\s+[A-Z])",
    r"\bcall me\b",
    r"\bmy (robot|project|robotics project|controller|board|device)\b.{0,20}\b(called|named|is)\b",
    r"\bi (have|own|built|made|created|designed)\b",
    # third-person name assertions: "the person's name is X", "his name is X"
    r"\b(person'?s|guy'?s|man'?s|woman'?s|lady'?s|gentleman'?s|his|her|their|someone'?s)\s+name\s+is\b",
    r"\b(he|she|they)\s+(is|goes by|is called|is named)\s+[A-Z]",
]

_CORRECT_MEMORY_PATTERNS = [
    r"\b(actually|correction|no longer|used to be|is now called|is now named|rename)\b",
    r"\b(change|update)\b.{0,30}\b(to|from)\b",
    r"\bthe (project|robot|name) is now\b",
]

_FORGET_MEMORY_PATTERNS = [
    r"\b(forget|delete|remove|erase)\b.{0,30}\b(my face|my identity|that fact|that memory|what you remember|everything you know)\b",
    r"\bforget (about )?(me|my)\b",
    r"\bdelete (my|the) (face|identity|memory|facts?)\b",
    r"\bremove (my|the) (face|identity)\b",
]

_QUERY_MEMORY_PATTERNS = [
    r"\b(do you (remember|recall|know)|did you (remember|recall))\b",
    r"\b(who am i|what is my name)\b",
    r"\bwhat did i (tell you|say)\b",
    r"\bwhat (do you know|have you learned) about\b",
    r"\bdo you recognise me\b",
    r"\bdo you recognize me\b",
    r"\b(anything|what) (about )?(my|the) (face|name|project|robot)\b",
    r"\bwho (is|do you think) i am\b",
]

_LIST_MEMORIES_PATTERNS = [
    r"\b(show|list|what are)\b.{0,30}\b(memories|facts|things you remember|what you remember)\b",
    r"\bwhat do you remember\b",
    r"\bshow what you remember\b",
]

# -- Persistent Episodic & Conversational Memory (PECM) ---------------------
# Historical questions about something *seen or said earlier* — answered from
# the durable event log without requiring a fresh camera frame.

_RECALL_VISUAL_PATTERNS = [
    r"\bwhat colou?r (was|were) the\b",
    r"\bwhat (did|was) you (see|observe|notice) (earlier|before|previously|last|a moment ago)\b",
    r"\b(earlier|previously|before|last time|a moment ago)\b.{0,40}\b(colou?r|see|observe|look)\b",
    r"\b(colou?r|see|observe|look)\b.{0,40}\b(earlier|previously|before|last time)\b",
    r"\bwhen the camera was (off|disabled|unavailable)\b",
    r"\bwhat did the camera see\b",
    r"\bwhat (was|is) (the|that|that thing|the object|my).{0,25}(colou?r|shape|size)\b.{0,20}\b(earlier|before|last|previously)\b",
    r"\bwhat did you last (see|observe)\b",
    r"\brepeat (what|that) you (saw|observed|noticed)\b",
    r"\bwhat was on (the|that) (table|desk|shelf) (earlier|before|previously)\b",
    r"\bdo you (still )?remember (seeing|what you saw|the color|what was there)\b",
]

_RECALL_MEMORY_PATTERNS = [
    r"\bwhat did you (say|tell me|answer) (earlier|before|previously|last|a moment ago|just now)\b",
    r"\b(earlier|previously|before|last time|a moment ago)\b.{0,40}\b(did you say|tell me|answer|say)\b",
    r"\b(say|tell me|answer)\b.{0,40}\b(earlier|previously|before|last time)\b",
    r"\bwhat (was|were) you (saying|talking about) (earlier|before|previously)\b",
    r"\bcan you (repeat|recall) (that|what you said|your last answer)\b",
    r"\bwhat did we (discuss|talk about|cover)\b",
    r"\bremind me (of|about) (what|that|the thing)\b.{0,30}\b(earlier|before|said|discussed)\b",
    r"\bwhat happened (earlier|before|previously|last)\b",
    r"\bwalk me through (what|the) (happened|was said|we discussed)\b.{0,20}\b(earlier|before)\b",
]


def _matches_any(text: str, patterns: list[str]) -> Optional[re.Match]:
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m
    return None


def _object_name_from(transcript: str) -> str:
    """Pull the user-provided object name out of an enrolment utterance."""
    patterns = [
        # "Remember this object: my custom motor controller"
        r"(?:remember|learn|save|store|note|call)\s+(?:this|that|the)\s+\w+\s*[:\-–]\s*(.+)$",
        # "remember this as my motor controller"
        r"(?:remember|learn|save|store)\s+(?:this|that)\s+as\s+(.+)$",
        # "remember this object my motor controller" / "... object called X"
        r"(?:remember|learn|save|store)\s+(?:this|that|the)\s+(?:object|device|thing|item|board|circuit)\s+(?:called|named)?\s+([A-Za-z].+)$",
        # legacy: "... this <word> as <name>"
        r"(?:remember|learn|save|store|note|call)\s+(?:this|that|the)\s+\w+\s+as\s+(.+)$",
        r"call\s+(?:this|that)\s+(?:my|the)\s+(.+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, transcript, re.IGNORECASE)
        if not m:
            continue
        name = m.group(1).strip().rstrip(".!?").strip()
        name = re.sub(r"\b(please|for me)\b\.?$", "", name, flags=re.IGNORECASE).strip()
        if name and re.search(r"[A-Za-z]", name):
            return name[:120]
    return "unnamed object"


class DeterministicRouter:
    """Local, zero-cost, deliberately conservative router."""

    name = "deterministic"

    def __init__(self, confidence_threshold: float = 0.92):
        self.threshold = confidence_threshold

    def route(self, transcript: str, history_context: str = "") -> RouteDecision:
        text = (transcript or "").strip()
        lowered = text.lower()

        if not text:
            plan = RoutePlan.model_validate(
                {**FALLBACK_GENERAL, "intent": Intent.UNSUPPORTED, "rationale": "empty transcript"}
            )
            return RouteDecision(plan=plan, router=self.name, error="empty transcript")

        # 1. Movement / prohibited requests are always refused locally.
        if _matches_any(lowered, _FUTURE_PATTERNS):
            return self._simple(
                Intent.FUTURE_CAPABILITY,
                "explain_limitation",
                {},
                confidence=0.97,
                rationale="physical action request - safety refusal",
            )

        if _matches_any(lowered, _LIMITATION_PATTERNS):
            return self._simple(
                Intent.UNSUPPORTED,
                "explain_limitation",
                {},
                confidence=0.95,
                rationale="capability outside the allowed tool set",
            )

        # 1b. Multimodal memory operations.
        memory = self._route_memory(lowered, text)
        if memory is not None:
            return memory

        # 2. Open scene description.
        if _matches_any(lowered, _SCENE_PATTERNS):
            return RouteDecision(
                plan=RoutePlan.model_validate(
                    {
                        "intent": Intent.SCENE_DESCRIPTION,
                        "requires_camera": True,
                        "requires_vlm": True,
                        "tool": "describe_scene",
                        "tool_arguments": {"detail": "normal"},
                        "normalized_question": "Describe everything visible in front of the camera.",
                        "confidence": 0.97,
                        "rationale": "explicit scene description request",
                    }
                ),
                router=self.name,
            )

        # 3. Visual questions.
        if _matches_any(lowered, _VISUAL_PATTERNS) or _matches_any(lowered, _OBJECT_PATTERNS):
            # Object questions about something never mentioned before still
            # need the camera, but the question must be self-contained.
            question = self._visual_question(text, history_context)
            intent = (
                Intent.OBJECT_REASONING
                if _matches_any(lowered, _OBJECT_PATTERNS)
                and not _matches_any(lowered, _VISUAL_PATTERNS)
                else Intent.VISUAL_QUESTION
            )
            return RouteDecision(
                plan=RoutePlan.model_validate(
                    {
                        "intent": intent,
                        "requires_camera": True,
                        "requires_vlm": True,
                        "tool": "analyze_camera",
                        "tool_arguments": {
                            "question": question,
                            "capture_mode": "fresh",
                        },
                        "normalized_question": question,
                        "confidence": 0.95,
                        "rationale": "visual pattern matched",
                    }
                ),
                router=self.name,
            )

        # 4. Repeats and status.
        if _matches_any(lowered, _REPEAT_PATTERNS):
            return self._simple(Intent.REPEAT_RESPONSE, "repeat_last_response", {}, 0.96, "repeat request")

        if _matches_any(lowered, _STATUS_PATTERNS):
            return self._simple(Intent.SYSTEM_STATUS, "get_system_status", {}, 0.9, "status request")

        # 5. Greetings (short utterances only; otherwise fall through).
        if _matches_any(lowered, _GREETING_PATTERNS) and len(lowered.split()) <= 8:
            return self._simple(Intent.GREETING, "respond_directly", {}, 0.98, "greeting")

        # 6. Everything else *probably* is general knowledge, but phrasing
        #    like "what about the other one?" is ambiguous -> stay uncertain.
        if self._looks_like_knowledge_question(lowered):
            return self._simple(Intent.GENERAL_QUESTION, "respond_directly", {}, 0.93, "knowledge question")

        return RouteDecision(
            plan=RoutePlan.model_validate({**FALLBACK_GENERAL, "confidence": 0.0}),
            router=self.name,
            error="no confident deterministic route",
        )

    def _route_memory(self, lowered: str, original: str) -> Optional[RouteDecision]:
        """High-precision memory routing; None when no pattern matches."""
        # Historical visual recall — before the live-visual patterns so
        # "what color was the curtain earlier" never requests the camera.
        if _matches_any(lowered, _RECALL_VISUAL_PATTERNS):
            return self._simple(
                Intent.RECALL_VISUAL_MEMORY,
                "recall_visual_memory",
                {"query": original, "transcript": original},
                0.93,
                "historical visual observation recall from durable event log",
            )

        if _matches_any(lowered, _RECALL_MEMORY_PATTERNS):
            return self._simple(
                Intent.RECALL_MEMORY,
                "recall_memory",
                {"query": original, "transcript": original},
                0.93,
                "historical conversation recall from durable event log",
            )

        # Enrolment of an identity — explicit, consent-bearing.
        if _matches_any(lowered, _LEARN_IDENTITY_PATTERNS):
            return RouteDecision(
                plan=RoutePlan.model_validate(
                    {
                        "intent": Intent.LEARN_IDENTITY,
                        "requires_camera": True,
                        "requires_vlm": False,
                        "tool": "remember_identity",
                        "tool_arguments": {
                            "transcript": original,
                            "capture_mode": "fresh",
                        },
                        "normalized_question": original,
                        "confidence": 0.96,
                        "rationale": "explicit face enrollment request",
                    }
                ),
                router=self.name,
            )

        # Enrolment of a visual object.
        if _matches_any(lowered, _LEARN_OBJECT_PATTERNS):
            name = _object_name_from(original)
            return RouteDecision(
                plan=RoutePlan.model_validate(
                    {
                        "intent": Intent.LEARN_VISUAL_OBJECT,
                        "requires_camera": True,
                        "requires_vlm": False,
                        "tool": "learn_visual_object",
                        "tool_arguments": {
                            "object_name": name,
                            "transcript": original,
                            "capture_mode": "fresh",
                        },
                        "normalized_question": original,
                        "confidence": 0.95,
                        "rationale": "explicit object enrollment request",
                    }
                ),
                router=self.name,
            )

        # Corrections take priority over plain remember/query.
        if _matches_any(lowered, _CORRECT_MEMORY_PATTERNS) and (
            _matches_any(lowered, _REMEMBER_FACT_PATTERNS)
            or "project" in lowered
            or "robot" in lowered
            or "name" in lowered
        ):
            return self._simple(
                Intent.CORRECT_MEMORY,
                "correct_memory",
                {"transcript": original},
                0.93,
                "correction of previously stored information",
            )

        if _matches_any(lowered, _FORGET_MEMORY_PATTERNS):
            return self._simple(
                Intent.FORGET_MEMORY,
                "forget_memory",
                {"transcript": original},
                0.94,
                "explicit deletion request",
            )

        if _matches_any(lowered, _LIST_MEMORIES_PATTERNS):
            return self._simple(
                Intent.LIST_MEMORIES,
                "list_memories",
                {},
                0.93,
                "request to list stored memories",
            )

        if _matches_any(lowered, _QUERY_MEMORY_PATTERNS):
            tool_args = {"query": original}
            decision = self._simple(
                Intent.QUERY_MEMORY,
                "query_memory",
                tool_args,
                0.94,
                "question about stored memories",
            )
            if _matches_any(lowered, [r"\b(recognise|recognize) me\b", r"\bwho am i\b"]) or (
                "face" in lowered and _matches_any(lowered, [r"\b(do you|who)\b"])
            ):
                decision.plan.requires_camera = True
                decision.plan.requires_vlm = False
                decision.plan.tool = "recognize_person"
            return decision

        if _matches_any(lowered, _REMEMBER_FACT_PATTERNS):
            return self._simple(
                Intent.REMEMBER_FACT,
                "remember_fact",
                {"transcript": original},
                0.93,
                "explicit request to remember a fact",
            )

        return None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _simple(intent: Intent, tool: str, args: dict, confidence: float, rationale: str) -> RouteDecision:
        visual = intent.value in {"visual_question", "scene_description", "object_reasoning"}
        plan = RoutePlan.model_validate(
            {
                "intent": intent,
                "requires_camera": visual,
                "requires_vlm": visual,
                "tool": tool,
                "tool_arguments": args,
                "normalized_question": args.get("question", ""),
                "confidence": confidence,
                "rationale": rationale,
            }
        )
        return RouteDecision(plan=plan, router="deterministic")

    @staticmethod
    def _visual_question(text: str, history_context: str) -> str:
        """Turn the utterance into a standalone visual question."""
        cleaned = re.sub(
            r"^\s*(hey|hi|hello)\s+(robot|aura)\s*[,!.\s]*", "", text, flags=re.IGNORECASE
        ).strip() or text.strip()
        cleaned = cleaned.rstrip("?").strip()
        if not cleaned.lower().startswith(
            ("what", "which", "how", "where", "who", "describe", "is there", "are there", "can you")
        ):
            cleaned = f"Regarding: {cleaned}"
        return cleaned[:400]

    @staticmethod
    def _looks_like_knowledge_question(lowered: str) -> bool:
        starters = (
            "what ", "why ", "how ", "who ", "when ", "where ", "which ",
            "explain ", "define ", "tell me ", "can you ", "could you ",
            "does ", "do ", "is ", "are ", "will ", "would ", "should ",
        )
        if lowered.startswith(starters):
            # Reference words imply context we cannot resolve locally.
            ambiguous = {
                "that", "this", "those", "these", "it", "they", "them", "their",
                "other", "another", "one", "ones", "previous", "prior", "last",
                "earlier", "before", "former", "latter", "again", "abovementioned",
            }
            words = set(re.findall(r"[a-z']+", lowered))
            if words & ambiguous:
                return False
            return True
        return False


# ---------------------------------------------------------------------------
# Cloud LLM router
# ---------------------------------------------------------------------------

class LLMRouter:
    name = "llm"

    def __init__(self, client: Optional[BaseLLMClient] = None):
        self.client = client or create_llm_client()

    def route(self, transcript: str, history_context: str = "", history: Sequence[dict] = ()) -> RouteDecision:
        import time

        t0 = time.perf_counter()
        messages = build_router_messages(transcript, history=history, extra=history_context)
        try:
            raw, result = self.client.chat_json(
                messages,
                temperature=0.0,
                max_tokens=700,
            )
        except CloudServiceError as exc:
            log.warning("cloud router failed: %s", exc)
            return RouteDecision(
                plan=RoutePlan.model_validate(FALLBACK_GENERAL),
                router=self.name,
                latency_s=time.perf_counter() - t0,
                error=str(exc),
            )
        try:
            plan = validate_route_plan(_coerce_route_dict(raw))
        except Exception as exc:  # noqa: BLE001 - validation errors
            log.warning("router returned an invalid plan: %s", exc)
            return RouteDecision(
                plan=RoutePlan.model_validate(FALLBACK_GENERAL),
                router=self.name,
                latency_s=time.perf_counter() - t0,
                error=f"schema validation failed: {exc}",
            )
        return RouteDecision(
            plan=plan,
            router=self.name,
            latency_s=time.perf_counter() - t0,
            alternatives={"model": result.model},
        )


def _coerce_route_dict(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json

        return json.loads(raw)
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    raise TypeError(f"cannot use {type(raw).__name__} as a route plan")


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class IntentRouter:
    def __init__(self, client: Optional[BaseLLMClient] = None, mode: Optional[str] = None):
        settings = get_settings().orchestrator
        self.mode = (mode or settings.router_mode).lower()
        self.threshold = settings.deterministic_router_confidence
        self.deterministic = DeterministicRouter(self.threshold)
        self.llm = LLMRouter(client)

    @property
    def offline(self) -> bool:
        return self.llm.client.offline or get_settings().orchestrator.offline

    def route(self, transcript: str, history_context: str = "", history: Sequence[dict] = ()) -> RouteDecision:
        det = self.deterministic.route(transcript, history_context)

        if self.mode == "deterministic" or self.offline:
            if det.error:
                # Offline with no confident local route: assume general.
                det.plan = RoutePlan.model_validate({**FALLBACK_GENERAL, "confidence": 0.4})
                det.error = None
            return det

        if self.mode == "auto" and not det.error and det.plan.confidence >= self.threshold:
            log.info("deterministic route accepted: %s", det.plan.intent.value)
            return det

        cloud = self.llm.route(transcript, history_context, history)
        if self.mode == "compare" and not det.error:
            if det.plan.intent != cloud.plan.intent:
                log.info(
                    "route disagreement: deterministic=%s cloud=%s (using cloud)",
                    det.plan.intent.value,
                    cloud.plan.intent.value,
                )
            cloud.alternatives["deterministic"] = det.plan.intent.value

        if cloud.error and not det.error:
            log.warning("falling back to deterministic route (%s)", cloud.error)
            det.alternatives["cloud_error"] = cloud.error
            return det
        return cloud
