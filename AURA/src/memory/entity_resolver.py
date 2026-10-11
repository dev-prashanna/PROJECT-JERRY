"""Entity resolution and cross-modal identity binding.

Resolution priority (highest first):

1. Explicit entity id from a verified operation
2. User-confirmed target selection
3. Strong contextual reference to an existing entity
4. Unambiguous name / alias match
5. Similarity-based candidate identification (never alone as proof)
6. Request clarification / return unknown
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.memory.models import (
    EntityType,
    EntityStatus,
    MemoryEntity,
    normalize_name,
    utcnow,
)
from src.utils.logger import get_logger

log = get_logger("memory.entities")

_SUBJECT_MAP = {
    "current_speaker": EntityType.PERSON,
    "i": EntityType.PERSON,
    "me": EntityType.PERSON,
    "myself": EntityType.PERSON,
    "my_robot": EntityType.PROJECT,
    "my_project": EntityType.PROJECT,
    "my_controller": EntityType.OBJECT,
    "my_board": EntityType.OBJECT,
    "my_device": EntityType.OBJECT,
}


@dataclass
class ResolutionResult:
    entity: Optional[MemoryEntity] = None
    status: str = "unknown"  # matched | created | ambiguous | unknown
    candidates: list[MemoryEntity] = field(default_factory=list)
    note: str = ""


class EntityResolver:
    def __init__(self, store):
        self.store = store

    def resolve_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
    ) -> ResolutionResult:
        matches = self.store.find_entities_by_name(name, entity_type)
        active = [m for m in matches if m.status == EntityStatus.ACTIVE]
        pool = active or matches
        if not pool:
            return ResolutionResult(status="unknown", note="no entity with that name")
        if len(pool) > 1:
            return ResolutionResult(
                status="ambiguous",
                candidates=pool,
                note="multiple entities share that name",
            )
        return ResolutionResult(entity=pool[0], status="matched")

    def get_or_create(
        self,
        name: str,
        entity_type: EntityType,
        *,
        aliases: Optional[list[str]] = None,
        provisional: bool = False,
        extra_aliases: Optional[list[str]] = None,
    ) -> MemoryEntity:
        resolution = self.resolve_by_name(name, entity_type)
        if resolution.entity is not None:
            entity = resolution.entity
            for alias in list(aliases or []) + list(extra_aliases or []):
                if alias and normalize_name(alias) not in {normalize_name(a) for a in entity.aliases}:
                    self.store.add_alias(entity.entity_id, alias)
                    entity.aliases.append(alias)
            self.store.touch_entity(entity.entity_id)
            return entity
        if resolution.status == "ambiguous":
            # Do not silently pick one; caller decides.  We still return the
            # first candidate tagged via note for logging.
            log.warning("ambiguous name %r: %d candidates", name, len(resolution.candidates))
            return resolution.candidates[0]

        entity = MemoryEntity(
            canonical_name=name.strip(),
            entity_type=entity_type,
            aliases=list(aliases or []),
            status=EntityStatus.PROVISIONAL if provisional else EntityStatus.ACTIVE,
            last_seen_at=utcnow(),
        )
        for alias in extra_aliases or []:
            if alias and alias != entity.canonical_name:
                entity.aliases.append(alias)
        self.store.upsert_entity(entity)
        return entity

    def resolve_subject(
        self,
        subject: str,
        *,
        speaker_entity: Optional[MemoryEntity] = None,
    ) -> ResolutionResult:
        key = normalize_name(subject)
        if key in {"current_speaker", "i", "me", "myself"} and speaker_entity is not None:
            return ResolutionResult(entity=speaker_entity, status="matched")
        if key in {"current_speaker", "i", "me", "myself"}:
            # No bound speaker; caller may create a provisional person.
            return ResolutionResult(status="unknown", note="no bound speaker entity")

        entity_type = _SUBJECT_MAP.get(key)
        if entity_type is None:
            entity_type = EntityType.PROJECT if key.startswith("my_") else EntityType.CONCEPT
        display = subject.replace("_", " ").strip()
        if not display:
            return ResolutionResult(status="unknown", note="empty subject")
        return self.resolve_by_name(display, entity_type)

    def link_association(
        self,
        source: MemoryEntity,
        target: MemoryEntity,
        relation: str,
        *,
        transcript: str = "",
        turn_id: str = "",
    ) -> None:
        from src.memory.models import EntityAssociation

        existing = self.store.associations_for(source.entity_id)
        for assoc in existing:
            other = (
                assoc.target_entity_id
                if assoc.source_entity_id == source.entity_id
                else assoc.source_entity_id
            )
            if other == target.entity_id and assoc.relation == relation:
                return
        self.store.add_association(
            EntityAssociation(
                source_entity_id=source.entity_id,
                target_entity_id=target.entity_id,
                relation=relation,
                source_transcript=transcript[:500],
                source_turn_id=turn_id,
            )
        )

    def merge(self, source_entity_id: str, target_entity_id: str) -> dict[str, int]:
        return self.store.merge_entities(source_entity_id, target_entity_id)

    def activate(self, entity_id: str) -> Optional[MemoryEntity]:
        entity = self.store.get_entity(entity_id)
        if entity is None:
            return None
        entity.status = EntityStatus.ACTIVE
        entity.updated_at = utcnow()
        self.store.upsert_entity(entity)
        return entity
