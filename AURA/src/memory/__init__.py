"""AURA Multimodal Memory System (MMS).

Public surface:

* :class:`MultimodalMemoryEngine` – narrow facade used by the orchestrator
* typed contracts in :mod:`src.memory.models`
* typed errors in :mod:`src.memory.exceptions`
"""

from __future__ import annotations

from src.memory.engine import MultimodalMemoryEngine
from src.memory.exceptions import MemoryError
from src.memory.models import (
    CorrectFactRequest,
    EnrollObjectRequest,
    EnrollPersonRequest,
    EntityType,
    ForgetRequest,
    MatchResult,
    MatchStatus,
    MemoryContext,
    MemoryEntity,
    MemoryFact,
    MemoryOperationResult,
    RememberFactRequest,
)

__all__ = [
    "MultimodalMemoryEngine",
    "MemoryError",
    "MemoryOperationResult",
    "MemoryEntity",
    "MemoryFact",
    "MemoryContext",
    "MatchResult",
    "MatchStatus",
    "EntityType",
    "EnrollPersonRequest",
    "EnrollObjectRequest",
    "RememberFactRequest",
    "CorrectFactRequest",
    "ForgetRequest",
]
