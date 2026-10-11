"""Temporal / referential query parsing for episodic recall.

Recognizes phrasing like "earlier", "previously", "last time", "what did you
say before", "what was the color…" and turns it into a structured
:class:`TemporalHint` the retriever can act on.  Relative times are resolved
against the actual current time and session boundaries — never against
arbitrary fixed assumptions.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.memory.event_models import TemporalHint

_HISTORY_PATTERNS = [
    r"\bearlier\b", r"\bpreviously\b", r"\bbefore\b", r"\blast time\b",
    r"\byesterday\b", r"\bin the past\b", r"\ba moment ago\b",
    r"\bwhat did you (?:say|see|observe|tell)\b", r"\bwhat was the\b",
    r"\bwas the\b.*\?", r"\bwhen the camera was\b", r"\bused to\b",
    r"\bremember (?:when|that time)\b", r"\bformerly\b",
]
_LAST_OBS_PATTERNS = [
    r"\blast (?:time|observation|seen|saw)\b", r"\bmost recently\b",
    r"\bthe most recent\b", r"\byou last\b", r"\bprevious observation\b",
    r"\bwhat did you last\b", r"\bwhat you (?:saw|observed|noticed)\b",
    r"\brepeat what you\b",
]
_PREV_RESPONSE_PATTERNS = [
    r"\bwhat did you (?:just |)say\b", r"\brepeat (?:that|your last|it)\b",
    r"\byou said\b", r"\bcan you repeat\b", r"\bagain\b",
]
_RECENT_TURN_PATTERNS = [
    r"\bjust now\b", r"\bright now\b", r"\bmoments ago\b",
    r"\bthe last (?:question|answer|thing)\b",
]
_ENTITY_PATTERNS = [
    r"\bthe ([a-z][a-z0-9-]{2,30})\b",
    r"\bmy ([a-z][a-z0-9-]{2,30})\b",
    r"\ba ([a-z][a-z0-9-]{2,30})\b",
]
_STOP_ENTITIES = {
    "color", "what", "when", "where", "that", "this", "these", "those",
    "time", "last", "first", "same", "other", "answer", "question",
    "camera", "memory", "previous", "earlier", "recent", "moment",
}


def _match_any(text: str, patterns: list[str]) -> list[str]:
    found = []
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            found.append(m.group(0))
    return found


def parse_temporal(text: str, *, now: Optional[datetime] = None) -> TemporalHint:
    lowered = (text or "").strip().lower()
    now = now or datetime.now(timezone.utc)
    hint = TemporalHint()

    history = _match_any(lowered, _HISTORY_PATTERNS)
    last_obs = _match_any(lowered, _LAST_OBS_PATTERNS)
    prev_resp = _match_any(lowered, _PREV_RESPONSE_PATTERNS)
    recent = _match_any(lowered, _RECENT_TURN_PATTERNS)

    hint.refers_to_history = bool(history or last_obs or prev_resp or recent)
    hint.refers_to_last_observation = bool(last_obs) or bool(
        history and re.search(r"\b(color|saw|observe|look)\b", lowered)
    )
    hint.refers_to_previous_response = bool(prev_resp)
    hint.refers_to_recent_turn = bool(recent) or bool(prev_resp)
    hint.raw_phrases = history + last_obs + prev_resp + recent

    # crude day-level anchoring for "yesterday"
    if re.search(r"\byesterday\b", lowered):
        hint.time_range_end = now
        hint.time_range_start = (now - timedelta(days=2)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    # entity hint: first noun after "the"/"my"/"a" that is not a stop word
    for p in _ENTITY_PATTERNS:
        m = re.search(p, lowered)
        if m:
            candidate = m.group(1).strip()
            if candidate and candidate not in _STOP_ENTITIES and len(candidate) >= 3:
                hint.entity_hint = candidate
                break
    return hint
