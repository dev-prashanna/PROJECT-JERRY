"""Vector index implementations for MMS.

SQLite remains the authoritative store; these indexes are derived views kept
synchronized with the ``face_embeddings`` / ``object_embeddings`` tables.

``ExactNumpyIndex`` is the correctness baseline.  An optional sqlite-vec
adapter can be added later behind the same protocol without touching callers.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

import numpy as np

from src.memory.exceptions import EmbeddingModelMismatch, InvalidEmbedding
from src.utils.logger import get_logger

log = get_logger("memory.vector")


def serialize_vector(vector: np.ndarray) -> tuple[bytes, int, str]:
    """Deterministic float32 serialization; raises on NaN/inf/zero-length."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise InvalidEmbedding(f"expected a non-empty 1-D vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise InvalidEmbedding("embedding contains NaN or infinite components")
    return arr.tobytes(), int(arr.size), "float32"


def deserialize_vector(blob: bytes, dimension: int, dtype: str = "float32") -> np.ndarray:
    expected_dtype = np.dtype(dtype if dtype in {"float32", "float64"} else "float32")
    expected_bytes = int(dimension) * expected_dtype.itemsize
    if len(blob) != expected_bytes:
        raise InvalidEmbedding(
            f"blob length {len(blob)} does not match dimension {dimension} ({expected_bytes} bytes)"
        )
    arr = np.frombuffer(blob, dtype=expected_dtype).astype(np.float32, copy=True)
    if not np.all(np.isfinite(arr)):
        raise InvalidEmbedding("deserialized embedding contains non-finite values")
    return arr


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise InvalidEmbedding("cannot normalize a near-zero vector")
    return arr / norm


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


@runtime_checkable
class VectorIndex(Protocol):
    model_id: str
    dimension: int

    def add(self, embedding_id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None: ...

    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[str, float, dict]]: ...

    def delete(self, embedding_id: str) -> None: ...

    def rebuild(self, entries: Iterable[tuple[str, np.ndarray, dict]]) -> None: ...


class ExactNumpyIndex:
    """Brute-force L2 (faces) or cosine (objects) search over float32 rows."""

    def __init__(
        self,
        model_id: str,
        dimension: int,
        *,
        metric: str = "l2",
        normalize: bool = False,
    ):
        if metric not in {"l2", "cosine"}:
            raise ValueError(f"unsupported metric {metric!r}")
        self.model_id = model_id
        self.dimension = int(dimension)
        self.metric = metric
        self.normalize = bool(normalize or metric == "cosine")
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._rows: list[np.ndarray] = []
        self._meta: list[dict] = []
        self._id_to_pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, embedding_id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dimension,):
            raise EmbeddingModelMismatch(
                f"vector shape {arr.shape} does not match index dimension {self.dimension}"
            )
        if not np.all(np.isfinite(arr)):
            raise InvalidEmbedding("refusing to index non-finite vector")
        if self.normalize:
            arr = l2_normalize(arr)
        with self._lock:
            if embedding_id in self._id_to_pos:
                pos = self._id_to_pos[embedding_id]
                self._rows[pos] = arr
                self._meta[pos] = dict(metadata or {})
                return
            self._id_to_pos[embedding_id] = len(self._ids)
            self._ids.append(embedding_id)
            self._rows.append(arr.copy())
            self._meta.append(dict(metadata or {}))

    def delete(self, embedding_id: str) -> None:
        with self._lock:
            pos = self._id_to_pos.pop(embedding_id, None)
            if pos is None:
                return
            self._ids[pos] = self._ids[-1]
            self._rows[pos] = self._rows[-1]
            self._meta[pos] = self._meta[-1]
            self._ids.pop()
            self._rows.pop()
            self._meta.pop()
            if pos < len(self._ids):
                self._id_to_pos[self._ids[pos]] = pos

    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[str, float, dict]]:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dimension,):
            raise EmbeddingModelMismatch(
                f"query shape {arr.shape} does not match index dimension {self.dimension}"
            )
        if self.normalize:
            arr = l2_normalize(arr)
        with self._lock:
            if not self._rows:
                return []
            matrix = np.stack(self._rows)
            if self.metric == "l2":
                scores = np.linalg.norm(matrix - arr, axis=1)
            else:
                scores = matrix @ arr
            keep = list(range(len(self._ids)))
            if filters:
                keep = [
                    i
                    for i in keep
                    if all(self._meta[i].get(fk) == fv for fk, fv in filters.items())
                ]
            if not keep:
                return []
            kept_scores = scores[keep]
            order = np.argsort(kept_scores)
            if self.metric == "cosine":
                order = order[::-1]
            out: list[tuple[str, float, dict]] = []
            for idx in order[: max(1, k)]:
                pos = keep[int(idx)]
                out.append((self._ids[pos], float(scores[pos]), dict(self._meta[pos])))
            return out

    def rebuild(self, entries: Iterable[tuple[str, np.ndarray, dict]]) -> None:
        with self._lock:
            self._ids.clear()
            self._rows.clear()
            self._meta.clear()
            self._id_to_pos.clear()
        for embedding_id, vector, metadata in entries:
            self.add(embedding_id, vector, metadata)


def create_index(model_id: str, dimension: int, *, backend: str = "numpy", metric: str = "l2") -> ExactNumpyIndex:
    """Factory; ``backend`` currently only supports exact numpy search."""
    if backend not in {"numpy", "exact", "auto"}:
        log.warning("vector backend %r not available; using exact numpy index", backend)
    return ExactNumpyIndex(model_id, dimension, metric=metric, normalize=(metric == "cosine"))
