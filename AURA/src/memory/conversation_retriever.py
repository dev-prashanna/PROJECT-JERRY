"""Hybrid conversational / episodic retrieval (PECM read path).

Implements the two-level retrieval model from the spec:

* **Level 1 — immediate**: active-session event history + recent committed
  events + direct keyword matching.  Works before any embedding job runs.
* **Level 2 — long-term**: FTS5 keyword search, stored semantic embeddings,
  entity links and historical episodes, fused with Reciprocal Rank Fusion
  and reranked with temporal + source-aware adjustments.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

from src.memory.event_models import (
    ConversationEvent,
    Episode,
    EventType,
    RecallResult,
    RecallStatus,
    RetrievedEvent,
    TemporalHint,
)
from src.memory.hybrid_search import rrf_fuse
from src.memory.temporal_resolver import parse_temporal

log = logging.getLogger("aura.memory.conv_retriever")

_SOURCE_PRIORITY = {
    EventType.VLM_OBSERVATION: 1.0,
    EventType.ASSISTANT_RESPONSE: 0.9,
    EventType.USER_TRANSCRIPT: 0.8,
    EventType.USER_TEXT: 0.8,
    EventType.TOOL_RESULT: 0.6,
    EventType.MEMORY_CORRECTION: 0.95,
}


class ConversationRetriever:
    """Hybrid retriever over conversation events and episodes."""

    def __init__(self, store, *, text_index=None, embedder=None,
                 max_results: int = 10):
        self.store = store
        self.text_index = text_index   # TextEmbeddingIndex or None
        self.embedder = embedder       # text embedder or None
        self.max_results = max_results

    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, session_id: Optional[str] = None,
                 hint: Optional[TemporalHint] = None,
                 max_events: int = 10,
                 max_episodes: int = 4) -> RecallResult:
        t0 = time.perf_counter()
        result = RecallResult(query=query)
        if self.store is None:
            result.status = RecallStatus.STORAGE_UNAVAILABLE
            result.notes.append("memory store unavailable")
            result.duration_s = time.perf_counter() - t0
            return result

        hint = hint or parse_temporal(query)
        try:
            # -- Level 1: immediate recent context -----------------------
            recent = self._recent_ranking(query, session_id=session_id, hint=hint)
            # -- Level 2: keyword (FTS) ----------------------------------
            keyword = self._keyword_ranking(query, hint=hint)
            # -- Level 2: semantic ---------------------------------------
            semantic = self._semantic_ranking(query, hint=hint)

            ranked_lists: dict[str, list] = {
                "recent": [e for e, _ in recent],
                "keyword": keyword,
            }
            if semantic:
                ranked_lists["semantic"] = [e for e, _ in semantic]
            weights = {"recent": 2.0 if hint.refers_to_recent_turn or hint.refers_to_previous_response else 1.0,
                       "keyword": 1.0, "semantic": 0.8}
            fused = rrf_fuse(ranked_lists, k=60, weights=weights)

            # -- rerank: source-aware + temporal --------------------------
            now = datetime.now().astimezone()
            reranked: list[RetrievedEvent] = []
            for event, score, per_source in fused:
                adjusted = score * _SOURCE_PRIORITY.get(event.event_type, 0.7)
                # recent events get a mild boost when no explicit history phrasing
                if not hint.refers_to_history:
                    age_h = max(0.0, (now - event.timestamp).total_seconds() / 3600.0)
                    adjusted *= 1.0 / (1.0 + age_h / 24.0)
                if hint.time_range_start and event.timestamp < hint.time_range_start:
                    continue
                if hint.time_range_end and event.timestamp > hint.time_range_end:
                    continue
                if hint.entity_hint and hint.entity_hint not in query.lower():
                    # entity-scoped query: require the hint to appear
                    if hint.entity_hint.lower() not in event.content.lower():
                        continue
                src = "keyword"
                if "recent" in per_source and per_source["recent"]["rank"] == 0:
                    src = "recent"
                elif "semantic" in per_source:
                    src = "semantic"
                reranked.append(RetrievedEvent(
                    event=event, score=adjusted,
                    rank_keyword=per_source.get("keyword", {}).get("rank"),
                    rank_semantic=per_source.get("semantic", {}).get("rank"),
                    rank_recent=per_source.get("recent", {}).get("rank"),
                    source=src,
                ))
            reranked.sort(key=lambda r: r.score, reverse=True)
            result.events = reranked[:max_events]

            # -- episodes -------------------------------------------------
            try:
                episodes = self.store.episodes_fts(query, limit=max_episodes)
            except Exception:  # noqa: BLE001
                episodes = []
            if not episodes:
                try:
                    episodes = self.store.recent_episodes(
                        session_id=session_id, limit=max_episodes
                    )
                except Exception:  # noqa: BLE001
                    episodes = []
            from src.memory.event_models import RetrievedEpisode

            result.episodes = [RetrievedEpisode(episode=ep, score=1.0 - i * 0.05)
                               for i, ep in enumerate(episodes[:max_episodes])]

            if result.found:
                result.status = RecallStatus.FOUND
            else:
                result.status = RecallStatus.NOT_FOUND
                result.notes.append("no matching conversation events or episodes")
        except Exception as exc:  # noqa: BLE001
            log.warning("conversation retrieval failed: %s", exc)
            result.status = RecallStatus.STORAGE_UNAVAILABLE
            result.notes.append(str(exc))
        result.duration_s = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    def _recent_ranking(self, query: str, *, session_id: Optional[str],
                        hint: TemporalHint) -> list[tuple[ConversationEvent, float]]:
        limit = 12 if hint.refers_to_recent_turn or hint.refers_to_previous_response else 6
        try:
            events = self.store.recent_events(session_id=session_id, limit=limit)
        except Exception:  # noqa: BLE001
            return []
        q_tokens = set(_tokens(query))
        scored: list[tuple[ConversationEvent, float]] = []
        for e in events:
            overlap = len(q_tokens & set(_tokens(e.content)))
            # recency rank: newer events rank better
            scored.append((e, float(overlap)))
        scored.sort(key=lambda t: t[1], reverse=True)
        # return in recency order for the "recent" ranking (newest first)
        by_recency = sorted(scored, key=lambda t: (t[0].timestamp, t[0].event_sequence), reverse=True)
        return by_recency

    def _keyword_ranking(self, query: str, *, hint: TemporalHint) -> list[ConversationEvent]:
        try:
            return self.store.search_events_fts(query, limit=self.max_results)
        except Exception:  # noqa: BLE001
            return []

    def _semantic_ranking(self, query: str, *, hint: TemporalHint) -> list[tuple[ConversationEvent, float]]:
        if self.text_index is None or self.embedder is None or len(self.text_index) == 0:
            return []
        try:
            q_vec = self.embedder.encode(query)
            hits = self.text_index.search(q_vec, k=self.max_results)
        except Exception:  # noqa: BLE001
            return []
        out: list[tuple[ConversationEvent, float]] = []
        for emb_id, score, meta in hits:
            source_id = meta.get("source_id")
            if not source_id:
                continue
            try:
                event = self.store.get_event(source_id)
            except Exception:  # noqa: BLE001
                continue
            if event is not None:
                out.append((event, float(score)))
        return out


def _tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))
