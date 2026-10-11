"""Cross-modal retrieval over entities, facts and embeddings.

Modality-specific candidate sets are built independently; entity
associations provide the cross-modal bridge.  Euclidean face distances and
cosine object similarities are never combined into one scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.memory.models import (
    EntityType,
    EntityStatus,
    MatchResult,
    MatchStatus,
    MemoryContext,
    MemoryEntity,
    MemoryFact,
    new_id,
    utcnow,
)
from src.memory.vector_index import (
    ExactNumpyIndex,
    InvalidEmbedding,
    deserialize_vector,
    l2_normalize,
)
from src.utils.logger import get_logger

log = get_logger("memory.retriever")

_STOPWORDS = {
    "the", "and", "for", "you", "your", "what", "when", "where", "which",
    "that", "this", "with", "from", "tell", "said", "asked", "about", "was",
    "were", "are", "did", "have", "has", "had", "can", "could", "would",
    "should", "please", "remember", "recall", "know", "much", "many",
    "anything", "something", "there", "here", "into", "onto", "over",
}


def _significant_tokens(text: str) -> list[str]:
    import re

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-_']{2,}", text or "")
    out: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _STOPWORDS or low.startswith(("wh", "how", "who")):
            continue
        out.append(tok)
    return out[:8]


@dataclass
class RetrieverLimits:
    max_entities: int = 5
    max_facts: int = 10
    token_budget: int = 600


@dataclass
class RetrievalState:
    entities: list[MemoryEntity] = field(default_factory=list)
    facts: list[MemoryFact] = field(default_factory=list)
    matches: list[MatchResult] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class MemoryRetriever:
    def __init__(
        self,
        store,
        limits: Optional[RetrieverLimits] = None,
        *,
        face_index: Optional[ExactNumpyIndex] = None,
        object_index: Optional[ExactNumpyIndex] = None,
    ):
        self.store = store
        self.limits = limits or RetrieverLimits()
        self.face_index = face_index
        self.object_index = object_index

    # -- by name / id ---------------------------------------------------
    def retrieve_by_name(self, name: str, entity_type: Optional[EntityType] = None) -> list[MemoryEntity]:
        matches = self.store.find_entities_by_name(name, entity_type)
        active = [m for m in matches if m.status == EntityStatus.ACTIVE]
        return (active or matches)[: self.limits.max_entities]

    def retrieve_by_entity_id(self, entity_id: str) -> Optional[MemoryEntity]:
        return self.store.get_entity(entity_id)

    def facts_for(self, entity_id: str) -> list[MemoryFact]:
        return self.store.facts_for_entity(entity_id, limit=self.limits.max_facts)

    # -- by fact search -------------------------------------------------
    def retrieve_by_fact(self, needle: str, limit: Optional[int] = None) -> list[MemoryFact]:
        return self.store.search_facts(needle=needle, limit=limit or self.limits.max_facts)

    # -- by face --------------------------------------------------------
    def retrieve_by_face(self, embedding: np.ndarray) -> tuple[Optional[MemoryEntity], MatchResult]:
        if self.face_index is None or len(self.face_index) == 0:
            return None, MatchResult(status=MatchStatus.UNAVAILABLE, notes=["face index empty"])
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        hits = self.face_index.search(vec, k=2)
        if not hits:
            return None, MatchResult(status=MatchStatus.UNKNOWN)
        best_id, best_d, meta = hits[0]
        second = hits[1][1] if len(hits) > 1 else None
        entity_id = meta.get("entity_id")
        entity = self.store.get_entity(entity_id) if entity_id else None
        cosine = 1.0 - (best_d ** 2) / 2.0
        if best_d > 0.55:
            return None, MatchResult(
                status=MatchStatus.UNKNOWN,
                raw_distance=best_d,
                similarity_score=cosine,
                second_best_score=second,
                embedding_model_id=self.face_index.model_id,
                evidence_ids=[best_id],
            )
        if second is not None and (second - best_d) < 0.05:
            return None, MatchResult(
                status=MatchStatus.AMBIGUOUS,
                raw_distance=best_d,
                similarity_score=cosine,
                second_best_score=second,
                embedding_model_id=self.face_index.model_id,
                evidence_ids=[best_id],
            )
        return entity, MatchResult(
            entity_id=entity_id,
            status=MatchStatus.MATCHED if entity else MatchStatus.UNKNOWN,
            raw_distance=best_d,
            similarity_score=cosine,
            second_best_score=second,
            embedding_model_id=self.face_index.model_id,
            evidence_ids=[best_id],
            entity_name=entity.canonical_name if entity else None,
        )

    # -- by object ------------------------------------------------------
    def retrieve_by_object(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 3,
        min_similarity: float = 0.65,
    ) -> list[MatchResult]:
        if self.object_index is None or len(self.object_index) == 0:
            return [MatchResult(status=MatchStatus.UNAVAILABLE, notes=["object index empty"])]
        vec = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        hits = self.object_index.search(vec, k=top_k)
        out: list[MatchResult] = []
        for emb_id, score, meta in hits:
            entity_id = meta.get("entity_id")
            entity = self.store.get_entity(entity_id) if entity_id else None
            status = MatchStatus.MATCHED if score >= min_similarity and entity else (
                MatchStatus.MATCHED if entity and score >= min_similarity else MatchStatus.UNKNOWN
            )
            if score < min_similarity:
                status = MatchStatus.UNKNOWN
            out.append(
                MatchResult(
                    entity_id=entity_id,
                    status=status,
                    similarity_score=score,
                    embedding_model_id=self.object_index.model_id,
                    evidence_ids=[emb_id],
                    entity_name=entity.canonical_name if entity else None,
                    notes=[] if score >= min_similarity else ["similarity below novelty threshold"],
                )
            )
        return out

    # -- multimodal -----------------------------------------------------
    def retrieve_multimodal(
        self,
        *,
        text_query: str = "",
        face_embedding: Optional[np.ndarray] = None,
        object_embedding: Optional[np.ndarray] = None,
        entity_hint: Optional[str] = None,
    ) -> MemoryContext:
        state = RetrievalState()
        seed_entities: list[MemoryEntity] = []

        if entity_hint:
            seed_entities.extend(self.retrieve_by_name(entity_hint))

        if face_embedding is not None:
            entity, match = self.retrieve_by_face(face_embedding)
            state.matches.append(match)
            if entity is not None:
                seed_entities.append(entity)
                state.references.append(f"face:{match.evidence_ids[0] if match.evidence_ids else ''}")
            else:
                state.notes.append(f"face match status={match.status.value}")

        if object_embedding is not None:
            for match in self.retrieve_by_object(object_embedding):
                state.matches.append(match)
                if match.entity_id:
                    entity = self.store.get_entity(match.entity_id)
                    if entity is not None:
                        seed_entities.append(entity)
                        if match.evidence_ids:
                            state.references.append(f"object:{match.evidence_ids[0]}")

        if text_query:
            for entity in self.retrieve_by_name(text_query):
                seed_entities.append(entity)
            # Keyword expansion: single-token lookups ("NATERIDA", "robot")
            for token in _significant_tokens(text_query):
                for entity in self.retrieve_by_name(token):
                    seed_entities.append(entity)
                for fact in self.retrieve_by_fact(token):
                    state.facts.append(fact)
            for fact in self.retrieve_by_fact(text_query):
                state.facts.append(fact)
            if not seed_entities and not state.facts:
                state.notes.append("no stored memory matched the text query")

        # Expand through associations (cross-modal bridge).
        seen: set[str] = set()
        for entity in seed_entities:
            if entity.entity_id in seen:
                continue
            seen.add(entity.entity_id)
            state.entities.append(entity)
            for assoc in self.store.associations_for(entity.entity_id):
                other_id = (
                    assoc.target_entity_id
                    if assoc.source_entity_id == entity.entity_id
                    else assoc.source_entity_id
                )
                if other_id in seen:
                    continue
                other = self.store.get_entity(other_id)
                if other is not None and other.status == EntityStatus.ACTIVE:
                    seen.add(other.entity_id)
                    state.entities.append(other)

        state.entities = state.entities[: self.limits.max_entities]

        # Attach subject entities of any facts found via text search.
        for fact in list(state.facts):
            if fact.subject_entity_id and fact.subject_entity_id not in seen:
                subject = self.store.get_entity(fact.subject_entity_id)
                if subject is not None:
                    seen.add(subject.entity_id)
                    state.entities.append(subject)

        # De-duplicate facts collected from multiple search paths.
        deduped: dict[str, MemoryFact] = {f.fact_id: f for f in state.facts}
        state.facts = list(deduped.values())

        fact_ids = {f.fact_id for f in state.facts}
        for entity in state.entities:
            for fact in self.facts_for(entity.entity_id):
                if fact.fact_id not in fact_ids:
                    state.facts.append(fact)
                    fact_ids.add(fact.fact_id)
        state.facts = state.facts[: self.limits.max_facts]

        return MemoryContext(
            matched_entities=state.entities,
            retrieved_facts=state.facts,
            visual_matches=state.matches,
            source_references=[r for r in state.references if r],
            uncertainty_notes=state.notes,
            generated_at=utcnow(),
        )
