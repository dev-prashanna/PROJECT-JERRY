"""Local text embedding for episodic memory.

Default embedder is a deterministic hashing projector (`hash-embed-v1`):
no model download, sub-millisecond, good enough to give FTS a complementary
semantic signal for short assistant-memory texts.  If sentence-transformers
models are locally cached, they can be selected via configuration; the raw
memory (events + FTS) remains fully functional when no embedder is available.

Embedding policy (per spec §11):
* embed episodic summaries and relevant conversational text once per source;
* store model id, dimension, and creation time;
* reject cross-model comparisons;
* never block fresh recall on embedding completion.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

import numpy as np

from src.memory.exceptions import EmbeddingModelMismatch, InvalidEmbedding
from src.memory.vector_index import l2_normalize, serialize_vector

log = logging.getLogger("aura.memory.text_embeddings")

TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


class HashTextEmbedder:
    """Deterministic hashing bag-of-words embedder (no downloads)."""

    name = "hash-embed-v1"
    dimension = 256

    def available(self) -> bool:
        return True

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for tok in _tokens(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
            # second hash for slightly better collision behaviour
            idx2 = int.from_bytes(digest[2:6], "little") % self.dimension
            vec[idx2] += 0.5 * sign
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            vec[0] = 1.0
            norm = 1.0
        return vec / norm


class SentenceTransformerEmbedder:
    """Optional wrapper around a locally-cached sentence-transformers model."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.name = model_name
        self._model = None
        self.dimension = 384  # bge-small / MiniLM default; fixed on load

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
            self.dimension = int(self._model.get_sentence_embedding_dimension())
            return self._model
        except Exception as exc:  # noqa: BLE001
            log.info("sentence-transformers unavailable (%s); hash embedder will be used", exc)
            self._model = False
            return None

    def available(self) -> bool:
        return self._load() is not None

    def encode(self, text: str) -> np.ndarray:
        model = self._load()
        if model is None:
            raise InvalidEmbedding("sentence-transformers model unavailable")
        vec = np.asarray(model.encode(text, normalize_embeddings=True), dtype=np.float32)
        return l2_normalize(vec)


def create_text_embedder(name: str = "auto", *, model_name: str = "",
                         allow_download: bool = False) -> object:
    """Resolve the text embedder; never downloads large weights silently."""
    resolved = (name or "auto").strip().lower()
    if resolved in {"hash", "hash-embed-v1", "none", "off"}:
        return HashTextEmbedder()
    if resolved in {"sentence-transformers", "st", "bge", "minilm", "auto"}:
        candidate = model_name or "BAAI/bge-small-en-v1.5"
        try:
            import os

            cache_root = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or ""
            if resolved == "auto" and not allow_download and not cache_root:
                # auto → prefer hash unless a cached model is clearly present
                return HashTextEmbedder()
            st = SentenceTransformerEmbedder(candidate)
            if st.available():
                return st
        except Exception:  # noqa: BLE001
            pass
        return HashTextEmbedder()
    log.warning("unknown text embedder %r; using hash-embed-v1", name)
    return HashTextEmbedder()


class TextEmbeddingIndex:
    """In-memory exact index over stored text embeddings (model-checked)."""

    def __init__(self, model_id: str, dimension: int):
        self.model_id = model_id
        self.dimension = dimension
        self._ids: list[str] = []
        self._rows: list[np.ndarray] = []
        self._meta: list[dict] = []
        self._pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, embedding_id: str, vector: np.ndarray, meta: Optional[dict] = None) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size != self.dimension:
            raise EmbeddingModelMismatch(
                f"expected {self.dimension}-D, got {vec.size}"
            )
        if not np.all(np.isfinite(vec)):
            raise InvalidEmbedding("non-finite values in text embedding")
        if embedding_id in self._pos:
            pos = self._pos[embedding_id]
            self._rows[pos] = vec
            self._meta[pos] = meta or {}
            return
        self._pos[embedding_id] = len(self._ids)
        self._ids.append(embedding_id)
        self._rows.append(vec)
        self._meta.append(meta or {})

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[str, float, dict]]:
        if not self._rows:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.size != self.dimension:
            raise EmbeddingModelMismatch(
                f"query dim {q.size} != index dim {self.dimension}"
            )
        mat = np.stack(self._rows)
        scores = mat @ q  # rows are normalized → cosine
        order = np.argsort(scores)[::-1][:k]
        return [(self._ids[i], float(scores[i]), self._meta[i]) for i in order]
