"""Linguistic fact extraction for the Multimodal Memory System.

High-precision deterministic patterns cover the everyday cases AURA must
remember ("my name is Alex", "my robot is called NATERIDA").  An optional
LLM path handles complex utterances and is validated with Pydantic before
anything is persisted.

Statements are classified first so hypotheticals, negations, questions and
quoted speech never become memories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

log = get_logger("memory.facts")


class StatementClass(str, Enum):
    EXPLICIT_ASSERTION = "explicit_assertion"
    CORRECTION = "correction"
    QUESTION = "question"
    HYPOTHETICAL = "hypothetical"
    NEGATION = "negation"
    QUOTED_SPEECH = "quoted_speech"
    OTHER = "other"


class ExtractedFact(BaseModel):
    subject: str = Field(default="current_speaker", max_length=120)
    predicate: str = Field(min_length=1, max_length=80)
    object_value: str = Field(min_length=1, max_length=500)
    value_type: str = "text"
    classification: StatementClass = StatementClass.EXPLICIT_ASSERTION
    supersedes_predicate: Optional[str] = None
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("predicate", mode="before")
    @classmethod
    def _norm_predicate(cls, v: object) -> str:
        return str(v or "").strip().lower().replace(" ", "_")

    @field_validator("object_value", mode="before")
    @classmethod
    def _strip_object(cls, v: object) -> str:
        return " ".join(str(v or "").split()).strip()


@dataclass
class FactExtractionResult:
    facts: list[ExtractedFact] = field(default_factory=list)
    classification: StatementClass = StatementClass.OTHER
    corrections: list[ExtractedFact] = field(default_factory=list)
    method: str = "deterministic"
    skipped_reason: str = ""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_HYPOTHETICAL_MARKERS = re.compile(
    r"\b(if i were|if i was|suppose|hypothetically|pretending|imagining|"
    r"maybe someday|in theory|were i|if someone called)\b",
    re.IGNORECASE,
)

_NEGATION_MARKERS = re.compile(
    r"\b(is not|isn'?t|are not|aren'?t|was not|wasn'?t|do not|don'?t|"
    r"did not|didn'?t|cannot|can'?t|never|no longer|not called|not named)\b",
    re.IGNORECASE,
)

_QUESTION_MARKERS = re.compile(r"\?\s*$|^\s*(who|what|when|where|why|how|is|are|do|does|did|can|could|would|should)\b", re.IGNORECASE)

_QUOTED_MARKERS = re.compile(
    r"\b(someone said|he said|she said|they said|quoted|according to)\b",
    re.IGNORECASE,
)

_CORRECTION_MARKERS = re.compile(
    r"\b(actually|correction|instead|no longer|used to be|has changed|"
    r"now called|now named|rename|updated? to|from .+ to )\b",
    re.IGNORECASE,
)


def classify_statement(transcript: str) -> StatementClass:
    text = (transcript or "").strip()
    if not text:
        return StatementClass.OTHER
    if _HYPOTHETICAL_MARKERS.search(text):
        return StatementClass.HYPOTHETICAL
    if _QUOTED_MARKERS.search(text):
        return StatementClass.QUOTED_SPEECH
    if _CORRECTION_MARKERS.search(text):
        return StatementClass.CORRECTION
    if _QUESTION_MARKERS.search(text):
        return StatementClass.QUESTION
    if _NEGATION_MARKERS.search(text) and not re.search(
        r"\b(no longer|used to be|from .+ to )\b", text, re.IGNORECASE
    ):
        return StatementClass.NEGATION
    return StatementClass.EXPLICIT_ASSERTION


# ---------------------------------------------------------------------------
# Deterministic extraction
# ---------------------------------------------------------------------------

_NAME_PATTERNS = [
    re.compile(r"\bmy name is ([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)", re.IGNORECASE),
    re.compile(r"\bi am ([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)\b(?![ ](?:not|called|named))", re.IGNORECASE),
    re.compile(r"\bi'?m ([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)\b(?![ ](?:not|called|named))", re.IGNORECASE),
    re.compile(r"\bcall me ([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)", re.IGNORECASE),
    re.compile(r"\bi go by ([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)", re.IGNORECASE),
    # third-person: "the person's name is Ninja", "his name is Alex", "her name is Sam"
    re.compile(r"\b(?:the\s+)?(?:person'?s|guy'?s|man'?s|woman'?s|lady'?s|gentleman'?s|"
               r"his|her|their|someone'?s|whoever'?s)\s+name\s+is\s+"
               r"([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)", re.IGNORECASE),
    re.compile(r"\b(?:he|she|they)\s+(?:is|goes by|is called|is named)\s+"
               r"([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)", re.IGNORECASE),
]

_NAMED_PATTERNS = [
    re.compile(
        r"\bmy (?:robot|project|robotics project|controller|board|device|gadget|"
        r"machine|system|car|bike|drone|arm|hand|setup|rig|build|creation) "
        r"(?:is|was) called ([A-Za-z0-9][\w\-' ]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy (?:robot|project|robotics project|controller|board|device|gadget|"
        r"machine|system) (?:is|goes by|is named) ([A-Za-z0-9][\w\-' ]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:remember|note|keep in mind)(?: that)? my "
        r"(?:robot|project|robotics project|controller|board|device|gadget) "
        r"is called ([A-Za-z0-9][\w\-' ]*)",
        re.IGNORECASE,
    ),
    re.compile(r"\bthe (?:project|robot|name) is ([A-Z][\w\-' ]+)", re.IGNORECASE),
]

_OWNERSHIP_PATTERNS = [
    re.compile(
        r"\bi (?:have|own|built|made|created|designed|wrote|programmed) "
        r"(?:a |an |the |my )?([^.,;!?]{3,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy ([a-z][a-z ]{2,30}?)(?: is| was) called ([^.,;!?]{2,60})",
        re.IGNORECASE,
    ),
]

_REMEMBER_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:remember|note|save|store|memorize)(?: that)?\s+",
    re.IGNORECASE,
)

_CORRECTION_NAME = re.compile(
    r"\b(?:actually,?\s+)?(?:my name|i am|i'?m|call me)(?: is| was)?\s+"
    r"(?:now\s+)?([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)",
    re.IGNORECASE,
)

_CORRECTION_RENAME = re.compile(
    r"\b(?:the |my )?(?:project|robot|device|controller|board|name) is now "
    r"(?:called|named)?\s*([A-Za-z0-9][\w\-' ]*)",
    re.IGNORECASE,
)

_CORRECTION_FROM_TO = re.compile(
    r"\b(?:rename|change|update)\s+(?:it|them|the\s+\w+)\s+from\s+"
    r"([^.,;!?]+?)\s+to\s+([^.,;!?]+)",
    re.IGNORECASE,
)


def _clean(value: str) -> str:
    value = value.strip().strip(".,!?;:'\"")
    value = re.sub(r"\s+", " ", value)
    # Capitalise simple names for canonical display
    if value and value.islower():
        value = value[0].upper() + value[1:]
    return value


def extract_deterministic(transcript: str) -> FactExtractionResult:
    text = (transcript or "").strip()
    result = FactExtractionResult(classification=classify_statement(text))
    if not text:
        result.skipped_reason = "empty transcript"
        return result

    cls = result.classification
    if cls in {StatementClass.QUESTION, StatementClass.HYPOTHETICAL, StatementClass.NEGATION, StatementClass.QUOTED_SPEECH, StatementClass.OTHER}:
        if cls != StatementClass.CORRECTION:
            result.skipped_reason = f"statement classified as {cls.value}"
            return result

    working = _REMEMBER_PREFIX.sub("", text)

    if cls == StatementClass.CORRECTION:
        m = _CORRECTION_RENAME.search(working) or _CORRECTION_FROM_TO.search(working)
        if m:
            if m.lastindex and m.lastindex >= 2:
                obj = _clean(m.group(2))
            else:
                obj = _clean(m.group(1))
            result.corrections.append(
                ExtractedFact(
                    subject="current_speaker",
                    predicate="name",
                    object_value=obj,
                    classification=StatementClass.CORRECTION,
                    confidence=0.9,
                )
            )
            return result
        m = _CORRECTION_NAME.search(working)
        if m:
            result.corrections.append(
                ExtractedFact(
                    subject="current_speaker",
                    predicate="name",
                    object_value=_clean(m.group(1)),
                    classification=StatementClass.CORRECTION,
                    confidence=0.85,
                )
            )
            return result
        # fall through: some "actually" utterances are plain assertions

    facts: list[ExtractedFact] = []

    for pat in _NAME_PATTERNS:
        m = pat.search(working)
        if m:
            value = _clean(m.group(1))
            if value.lower() not in {"not", "called", "named"} and len(value) >= 2:
                # Third-person possessives / pronouns refer to someone other
                # than the speaker ("the person's name is X", "he is X").
                matched = working[m.start():m.end()].lower()
                if re.search(
                    r"\b(person'?s|guy'?s|man'?s|woman'?s|lady'?s|gentleman'?s|"
                    r"his|her|their|someone'?s|whoever'?s|he|she|they)\b",
                    matched,
                ):
                    subject = "the_person"
                else:
                    subject = "current_speaker"
                facts.append(
                    ExtractedFact(
                        subject=subject,
                        predicate="name",
                        object_value=value,
                        classification=cls if cls != StatementClass.CORRECTION else StatementClass.EXPLICIT_ASSERTION,
                    )
                )
            break

    for pat in _NAMED_PATTERNS:
        m = pat.search(working)
        if m:
            value = _clean(m.group(1))
            subject_phrase = working[: m.start()].lower()
            subject = "my_robot" if "robot" in subject_phrase or "robot" in working.lower() else "my_project"
            if "project" in working.lower():
                subject = "my_project"
                predicate = "project_name"
            else:
                predicate = "name"
            if len(value) >= 2:
                facts.append(
                    ExtractedFact(
                        subject=subject,
                        predicate=predicate,
                        object_value=value,
                        classification=cls if cls != StatementClass.CORRECTION else StatementClass.EXPLICIT_ASSERTION,
                    )
                )
            break

    if not facts:
        for pat in _OWNERSHIP_PATTERNS:
            m = pat.search(working)
            if m:
                if m.lastindex and m.lastindex >= 2:
                    subject = _clean(m.group(1)).lower().replace(" ", "_")
                    value = _clean(m.group(2))
                    predicate = "name"
                else:
                    subject = "current_speaker"
                    value = _clean(m.group(1))
                    predicate = re.match(r"(\w+)", working.lower().split()[0] if working else "") or None
                    verb = re.search(r"\b(have|own|built|made|created|designed|wrote|programmed)\b", working, re.IGNORECASE)
                    predicate = (verb.group(1).lower() if verb else "has")
                if len(value) >= 3:
                    facts.append(
                        ExtractedFact(
                            subject=subject,
                            predicate=predicate,
                            object_value=value,
                            classification=cls,
                            confidence=0.8,
                        )
                    )
                break

    result.facts = facts
    if not facts:
        result.skipped_reason = "no deterministic fact pattern matched"
    return result


# ---------------------------------------------------------------------------
# Optional LLM-backed extraction
# ---------------------------------------------------------------------------

LLM_SYSTEM = """You extract durable facts a robot should remember from one utterance.
Return JSON: {"facts": [{"subject": "...", "predicate": "...", "object": "..."}], "correction": <bool>}
Rules:
- Only extract explicit assertions about the speaker or named things.
- Never extract questions, hypotheticals, negations, or quoted speech.
- subject is "current_speaker" for I/my statements, or a short snake_case phrase.
- predicate is a short snake_case relation (name, project_name, owns, built...).
- object is the value only.
- If the utterance corrects earlier information, set correction=true.
- Return {"facts": [], "correction": false} when nothing durable is asserted."""


def build_llm_messages(transcript: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LLM_SYSTEM},
        {"role": "user", "content": transcript},
    ]


def parse_llm_reply(raw: object) -> FactExtractionResult:
    import json

    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise TypeError("LLM fact reply must be an object")
    result = FactExtractionResult(method="llm")
    result.classification = (
        StatementClass.CORRECTION if raw.get("correction") else StatementClass.EXPLICIT_ASSERTION
    )
    for item in raw.get("facts", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            fact = ExtractedFact(
                subject=str(item.get("subject", "current_speaker")),
                predicate=str(item.get("predicate", "")),
                object_value=str(item.get("object", "")),
                classification=result.classification,
                confidence=0.75,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("rejected LLM fact %r: %s", item, exc)
            continue
        (result.corrections if result.classification == StatementClass.CORRECTION else result.facts).append(fact)
    return result


def extract_facts(
    transcript: str,
    *,
    client=None,
    use_llm: bool = False,
) -> FactExtractionResult:
    """Deterministic first; optionally refine/extend with the LLM."""
    result = extract_deterministic(transcript)
    if not use_llm or client is None:
        return result
    if result.classification not in {StatementClass.EXPLICIT_ASSERTION, StatementClass.CORRECTION}:
        return result
    try:
        raw, _meta = client.chat_json(build_llm_messages(transcript), temperature=0.0, max_tokens=400)
        llm_result = parse_llm_reply(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM fact extraction failed, using deterministic only: %s", exc)
        return result
    # Prefer deterministic facts when present (higher precision); otherwise
    # accept the LLM's structured output.
    if not result.facts and not result.corrections:
        return llm_result
    if llm_result.corrections and not result.corrections:
        result.corrections = llm_result.corrections
    result.method = "deterministic+llm"
    return result
