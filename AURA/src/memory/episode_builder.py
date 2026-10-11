"""Deterministic episodic memory construction.

Builds :class:`Episode` records from the raw conversation events of a turn.
Simple VQA turns (question + observation + answer) are summarised
deterministically — no cloud call.  Complex turns may optionally use an LLM,
but the original events remain the authoritative source.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.memory.event_models import ConversationEvent, Episode, EventType, EpisodeType
from src.memory.models import new_id, utcnow

log = logging.getLogger("aura.memory.episodes")

_VISUAL_TYPES = {EventType.VLM_OBSERVATION}


def _first(events: list[ConversationEvent], types: set[EventType]) -> Optional[ConversationEvent]:
    for e in events:
        if e.event_type in types:
            return e
    return None


def _shorten(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def classify_episode_type(events: list[ConversationEvent]) -> EpisodeType:
    if _first(events, _VISUAL_TYPES):
        return EpisodeType.VISUAL_OBSERVATION
    types = {e.event_type for e in events}
    if EventType.MEMORY_CORRECTION in types:
        return EpisodeType.FACT_DECLARATION
    if any(e.source_tool in {"remember_identity", "learn_visual_object"} for e in events):
        return (
            EpisodeType.IDENTITY_ENROLLMENT
            if any(e.source_tool == "remember_identity" for e in events)
            else EpisodeType.OBJECT_ENROLLMENT
        )
    if any(e.source_tool for e in events if e.event_type == EventType.TOOL_RESULT):
        return EpisodeType.TOOL_INTERACTION
    return EpisodeType.CONVERSATION


def build_episode_deterministic(
    events: list[ConversationEvent],
    *,
    session_id: str = "",
    turn_id: str = "",
) -> Optional[Episode]:
    """Build one episode from a turn's events; None when nothing is usable."""
    if not events:
        return None
    session_id = session_id or events[0].session_id
    turn_id = turn_id or events[0].turn_id
    if not session_id:
        return None

    user_q = _first(events, {EventType.USER_TRANSCRIPT, EventType.USER_TEXT})
    obs = _first(events, _VISUAL_TYPES)
    answer = _first(events, {EventType.ASSISTANT_RESPONSE})
    tools = [e for e in events if e.event_type == EventType.TOOL_RESULT]

    ep_type = classify_episode_type(events)

    parts: list[str] = []
    if ep_type == EpisodeType.VISUAL_OBSERVATION and user_q and obs:
        parts.append(f"The user asked: \"{_shorten(user_q.content)}\"")
        source = obs.source_model or "the vision model"
        parts.append(f"{source} observed: {_shorten(obs.content)}")
        if answer:
            parts.append(f"AURA answered: {_shorten(answer.content)}")
    elif user_q and answer:
        parts.append(f"The user said: \"{_shorten(user_q.content)}\"")
        parts.append(f"AURA answered: {_shorten(answer.content)}")
    elif user_q:
        parts.append(f"The user said: \"{_shorten(user_q.content)}\"")
    elif answer:
        parts.append(f"AURA answered: \"{_shorten(answer.content)}\"")
    else:
        return None

    for t in tools[:3]:
        parts.append(f"Tool {t.source_tool or 'unknown'} returned: {_shorten(t.content, 120)}")

    timestamps = [e.timestamp for e in events]
    frame_id = (obs.frame_id if obs else None) or next(
        (e.frame_id for e in events if e.frame_id), None
    )
    return Episode(
        episode_id=new_id("ep"),
        session_id=session_id,
        turn_id=turn_id,
        episode_summary=" ".join(parts),
        start_timestamp=min(timestamps),
        end_timestamp=max(timestamps),
        source_event_ids=[e.event_id for e in events],
        frame_id=frame_id,
        episode_type=ep_type,
        created_at=utcnow(),
    )


class EpisodeBuilder:
    """Facade used by the engine / indexing worker."""

    def __init__(self, store, *, llm_client=None):
        self.store = store
        self.llm_client = llm_client

    def build_for_turn(self, turn_id: str, *, session_id: str = "") -> Optional[Episode]:
        """Build (and persist) the episode for a turn if one does not exist yet."""
        if self.store is None or not turn_id:
            return None
        existing = self.store.episode_for_turn(turn_id)
        if existing is not None:
            return existing
        events = self.store.events_for_turn(turn_id)
        episode = build_episode_deterministic(events, session_id=session_id, turn_id=turn_id)
        if episode is None:
            return None
        try:
            self.store.insert_episode(episode)
        except Exception as exc:  # noqa: BLE001
            log.warning("episode persist failed for turn %s: %s", turn_id, exc)
        return episode
