"""Grounded spoken-response generation.

Two strategies:

* **full** – Groq LLM turns evidence + question + history into a natural
  spoken answer (best conversational quality).
* **direct** – the raw observation is phrased locally with no cloud call
  (lowest latency, used when the evidence already answers the question).

``auto`` picks per request.  Offline mode can only use ``direct`` for
visual answers, and must state its limitation for general questions.

The generator also *audits* its own output (:func:`audit_response`) so
unsupported visual claims can be measured by the benchmark suite instead of
being assumed away.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from config.settings import get_settings
from src.intelligence.groq_client import LLMResult  # noqa: F401 - re-exported
from src.intelligence.llm.base import BaseLLMClient
from src.intelligence.llm.registry import create_llm_client
from src.intelligence.prompt_builder import build_direct_response, build_response_messages
from src.orchestrator.schemas import Intent, RequestRecord, VISUAL_INTENTS
from src.utils.exceptions import CloudServiceError, OfflineError
from src.utils.logger import get_logger

log = get_logger("response")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "there", "here",
    "you", "your", "i", "we", "they", "he", "she", "with", "for", "from",
    "by", "as", "at", "not", "no", "can", "see", "looking", "visible", "image",
    "picture", "camera", "frame", "robot", "aura", "seems", "appears", "some",
    "any", "all", "front", "of", "what", "which", "who", "when", "where",
    "how", "many", "much", "about", "into", "over", "under", "left", "right",
    "here", "now", "well", "okay", "sure", "actually", "indeed", "certainly",
    "overall", "basically", "simply", "let", "make", "out", "thing", "stuff",
    "behind", "front", "above", "below", "beside", "near", "next", "between",
    "him", "his", "her", "hers", "them", "they", "their", "ours", "yours",
    "like", "look", "looks", "looking", "seem", "seems", "appear", "appears",
    "wearing", "worn", "focused", "visible", "partial", "partially",
    "one", "two", "three", "also", "then", "than", "very", "just", "only",
}

CLAIM_PHRASES = (
    "i can see", "i see", "i notice", "i observe", "there is", "there are",
    "visible", "in the image", "in front",
)

STOPWORDS_CLAIM = {s.split()[-1] for s in CLAIM_PHRASES}


@dataclass
class ResponseResult:
    text: str
    strategy: str                 # full | direct | offline
    model: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    audit: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None


class ResponseGenerator:
    def __init__(self, client: Optional[BaseLLMClient] = None):
        self.settings = get_settings()
        self.client = client or create_llm_client(self.settings.llm)

    # -- strategy selection ---------------------------------------------
    def choose_strategy(self, record: RequestRecord, requested: str = "auto") -> str:
        requested = (requested or "auto").lower()
        offline = record.mode == "offline" or self.client.offline

        if offline:
            # No cloud reasoning available: only local phrasing is possible.
            if record.plan and record.plan.is_visual:
                return "direct"
            return "offline"

        if requested in {"full", "direct"}:
            return requested
        if requested != "auto":
            log.warning("unknown RESPONSE_STRATEGY %r; using auto", requested)

        if not (record.plan and record.plan.is_visual):
            return "full"
        if any(not t.ok for t in record.tool_results):
            return "full"  # failures need explanation
        evidence = [e for e in record.evidence if e.usable]
        if not evidence:
            return "full"
        answer_len = len(evidence[-1].raw_answer)
        if answer_len > 220:
            return "full"  # long observation benefits from summarising
        if record.plan.intent == Intent.CLARIFICATION:
            return "full"
        return "direct"

    # -- generation ------------------------------------------------------
    def generate(
        self,
        record: RequestRecord,
        history: Sequence[dict[str, str]] = (),
        strategy: str = "auto",
    ) -> ResponseResult:
        chosen = self.choose_strategy(record, strategy)
        record.response_strategy = chosen
        started = time.perf_counter()

        try:
            if chosen == "direct":
                result = self._direct(record)
            elif chosen == "offline":
                result = self._offline(record)
            else:
                result = self._full(record, history)
        except CloudServiceError as exc:
            log.warning("cloud response failed (%s); falling back to local phrasing", exc)
            result = self._direct(record, note="cloud reasoning unavailable")
            result.error = str(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("response generation failed")
            result = ResponseResult(
                text="Sorry, I could not compose a response just now.",
                strategy=chosen,
                error=str(exc),
            )

        result.latency_s = time.perf_counter() - started
        result.audit = audit_response(record, result.text)
        if result.audit:
            log.warning("response audit findings: %s", result.audit)
        record.response = result.text
        return result

    # -- strategies ------------------------------------------------------
    def _full(self, record: RequestRecord, history: Sequence[dict[str, str]]) -> ResponseResult:
        offline_notice = record.mode == "offline"
        messages = build_response_messages(record, history=history, offline_notice=offline_notice)
        llm: LLMResult = self.client.chat(
            messages,
            temperature=0.2,
            max_tokens=self.settings.llm.max_completion_tokens,
        )
        text = (llm.text or "").strip()
        if not text:
            # Reasoning models occasionally spend every token on reasoning.
            llm = self.client.chat(
                messages,
                temperature=0.2,
                max_tokens=max(600, self.settings.llm.max_completion_tokens),
            )
            text = (llm.text or "").strip()
        if not text and record.evidence:
            return self._direct(record, note="empty LLM reply")
        if not text:
            text = "Sorry, I did not get a response."
        return ResponseResult(
            text=_tidy(text),
            strategy="full",
            model=llm.model,
            latency_s=llm.latency_s,
            prompt_tokens=llm.prompt_tokens,
            completion_tokens=llm.completion_tokens,
        )

    @staticmethod
    def _direct(record: RequestRecord, note: str = "") -> ResponseResult:
        usable = [e for e in record.evidence if e.usable]
        if usable:
            text = build_direct_response(usable[-1])
        elif record.plan and record.plan.is_visual:
            text = "I could not get a clear look at that just now."
        else:
            text = record.response or "I am not sure how to answer that right now."
        return ResponseResult(text=_tidy(text), strategy="direct")

    @staticmethod
    def _offline(record: RequestRecord) -> ResponseResult:
        if record.plan and record.plan.intent == Intent.GENERAL_QUESTION:
            text = (
                "I am offline at the moment, so I cannot look that up. "
                "I can still answer questions about what the camera sees."
            )
        elif record.plan and record.plan.intent == Intent.GREETING:
            text = "Hello! I am AURA. I can describe what is in front of the camera."
        elif record.plan and record.plan.intent in {Intent.FUTURE_CAPABILITY, Intent.UNSUPPORTED}:
            text = "I cannot do that yet. I can talk, look through my camera, and reason about simple things."
        else:
            text = "I am running offline, so I can only handle simple requests right now."
        return ResponseResult(text=_tidy(text), strategy="offline")

    def ping(self) -> bool:
        return self.client.ping()


# ---------------------------------------------------------------------------
# Tidying + audit
# ---------------------------------------------------------------------------

_MD_NOISE = re.compile(r"^[\s>*_#`\-•]+|[\s*_`#]+$")


def _tidy(text: str) -> str:
    text = " ".join((text or "").split())
    text = _MD_NOISE.sub("", text).strip()
    # Strip markdown emphasis that would be read aloud oddly.
    text = re.sub(r"[*_`]{1,3}", "", text)
    text = text.strip('"“”')
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if t not in STOPWORDS and len(t) > 2}


def audit_response(record: RequestRecord, text: str) -> list[str]:
    """Flag possible unsupported visual claims.

    Heuristic by design and reported as *findings*, not proof:

    * ``visual_claim_without_evidence`` – the answer asserts visual access
      although no usable observation exists.
    * ``novel_visual_entity`` – a content word appears in the answer that is
      in neither the evidence nor the question (candidate invention).
    * ``contradicts_uncertainty`` – evidence was uncertain/failed but the
      answer is stated with certainty about specifics.
    """
    findings: list[str] = []
    lowered = (text or "").lower()
    if not lowered.strip():
        return ["empty_response"]

    is_visual_request = bool(record.plan and record.plan.is_visual)
    usable = [e for e in record.evidence if e.usable]

    claims_vision = any(p in lowered for p in CLAIM_PHRASES)
    if is_visual_request and not usable and claims_vision:
        findings.append("visual_claim_without_evidence")

    if usable:
        evidence_tokens: set[str] = set()
        question_tokens: set[str] = set()
        for e in usable:
            evidence_tokens |= _tokens(e.raw_answer)
        if record.plan:
            question_tokens |= _tokens(record.plan.normalized_question)
        question_tokens |= _tokens(record.transcript)

        answer_tokens = _tokens(text)
        # Numbers are high-value claims: any number in the answer should also
        # appear in the evidence.
        evidence_numbers = set(re.findall(r"\d+", " ".join(e.raw_answer for e in usable)))
        for num in re.findall(r"\d+", text):
            if num not in evidence_numbers and num not in re.findall(r"\d+", record.transcript):
                findings.append(f"novel_number:{num}")

        novel = answer_tokens - evidence_tokens - question_tokens
        # Ignore generic connective words the model may add.
        novel -= {"could", "would", "should", "may", "might", "looks", "seems"}
        if novel:
            findings.append("novel_visual_entity:" + ",".join(sorted(novel)[:8]))

    return findings
