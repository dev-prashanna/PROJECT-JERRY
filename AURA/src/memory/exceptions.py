"""Typed errors for the Multimodal Memory System.

Every error maps to a structured tool result so one failed memory operation
never crashes AURA's speech or conversation pipeline.
"""

from __future__ import annotations

from typing import Any, Optional


class MemoryError(Exception):
    """Base class for all MMS errors."""

    code = "MemoryError"
    retryable = False

    def __init__(self, message: str, *, detail: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "message": str(self),
            "retryable": self.retryable,
            "detail": self.detail,
        }


class MemoryStoreUnavailable(MemoryError):
    code = "MemoryStoreUnavailable"
    retryable = True


class MemoryMigrationError(MemoryError):
    code = "MemoryMigrationError"


class MemoryPermissionDenied(MemoryError):
    code = "MemoryPermissionDenied"


class FaceNotDetected(MemoryError):
    code = "FaceNotDetected"


class MultipleFacesAmbiguous(MemoryError):
    code = "MultipleFacesAmbiguous"
    retryable = True


class FaceQualityInsufficient(MemoryError):
    code = "FaceQualityInsufficient"


class FaceRecognitionUnavailable(MemoryError):
    code = "FaceRecognitionUnavailable"


class ObjectNotLocated(MemoryError):
    code = "ObjectNotLocated"
    retryable = True


class ObjectSegmentationUnavailable(MemoryError):
    code = "ObjectSegmentationUnavailable"


class InvalidEmbedding(MemoryError):
    code = "InvalidEmbedding"


class EmbeddingModelMismatch(MemoryError):
    code = "EmbeddingModelMismatch"


class MemoryEntityNotFound(MemoryError):
    code = "MemoryEntityNotFound"


class ConflictingMemory(MemoryError):
    code = "ConflictingMemory"
    retryable = True


class SnapshotExpired(MemoryError):
    code = "SnapshotExpired"


class MemoryQueueFull(MemoryError):
    code = "MemoryQueueFull"
    retryable = True


class MemoryOperationCancelled(MemoryError):
    code = "MemoryOperationCancelled"


class ConsentRequired(MemoryError):
    code = "ConsentRequired"
    retryable = True


class InvalidMemoryRequest(MemoryError):
    code = "InvalidMemoryRequest"
