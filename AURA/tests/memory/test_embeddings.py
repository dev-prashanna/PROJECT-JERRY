"""Vector serialization and exact index behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.exceptions import EmbeddingModelMismatch, InvalidEmbedding
from src.memory.vector_index import (
    ExactNumpyIndex,
    cosine_similarity,
    deserialize_vector,
    euclidean_distance,
    l2_normalize,
    serialize_vector,
)


def test_serialize_roundtrip():
    v = np.random.default_rng(0).standard_normal(128).astype(np.float32)
    blob, dim, dtype = serialize_vector(v)
    assert dim == 128 and dtype == "float32"
    out = deserialize_vector(blob, dim, dtype)
    assert np.allclose(v, out)


def test_nan_rejected():
    v = np.zeros(8, dtype=np.float32)
    v[3] = np.nan
    with pytest.raises(InvalidEmbedding):
        serialize_vector(v)


def test_inf_rejected():
    v = np.zeros(8, dtype=np.float32)
    v[0] = np.inf
    with pytest.raises(InvalidEmbedding):
        serialize_vector(v)


def test_wrong_length_blob_rejected():
    blob, dim, _ = serialize_vector(np.ones(8, dtype=np.float32))
    with pytest.raises(InvalidEmbedding):
        deserialize_vector(blob[:4], dim)


def test_zero_vector_normalize_rejected():
    with pytest.raises(InvalidEmbedding):
        l2_normalize(np.zeros(4, dtype=np.float32))


def test_metrics():
    a = np.array([1, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0], dtype=np.float32)
    assert euclidean_distance(a, a) == pytest.approx(0.0)
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_index_l2_search_order():
    idx = ExactNumpyIndex("m", 3, metric="l2")
    idx.add("far", np.array([10, 0, 0], dtype=np.float32), {"entity_id": "e2"})
    idx.add("near", np.array([1.1, 0, 0], dtype=np.float32), {"entity_id": "e1"})
    hits = idx.search(np.array([1, 0, 0], dtype=np.float32), k=2)
    assert hits[0][0] == "near"
    assert hits[0][1] < hits[1][1]


def test_index_cosine_prefers_direction():
    idx = ExactNumpyIndex("m", 3, metric="cosine")
    idx.add("same", np.array([1, 0, 0], dtype=np.float32), {})
    idx.add("opp", np.array([-1, 0, 0], dtype=np.float32), {})
    hits = idx.search(np.array([0.9, 0.1, 0], dtype=np.float32), k=2)
    assert hits[0][0] == "same"


def test_index_dimension_mismatch():
    idx = ExactNumpyIndex("m", 4)
    with pytest.raises(EmbeddingModelMismatch):
        idx.add("x", np.ones(3, dtype=np.float32), {})
    idx.add("x", np.ones(4, dtype=np.float32), {})
    with pytest.raises(EmbeddingModelMismatch):
        idx.search(np.ones(3, dtype=np.float32), k=1)


def test_index_delete_and_rebuild():
    idx = ExactNumpyIndex("m", 2)
    idx.add("a", np.array([1, 0], dtype=np.float32), {})
    idx.add("b", np.array([0, 1], dtype=np.float32), {})
    assert len(idx) == 2
    idx.delete("a")
    assert len(idx) == 1
    idx.delete("missing")  # no-op
    idx.rebuild([("c", np.array([1, 1], dtype=np.float32), {"k": 1})])
    assert len(idx) == 1
    hits = idx.search(np.array([1, 1], dtype=np.float32), k=1)
    assert hits[0][0] == "c"


def test_index_filters():
    idx = ExactNumpyIndex("m", 2)
    idx.add("a", np.array([1, 0], dtype=np.float32), {"entity_id": "e1"})
    idx.add("b", np.array([0.99, 0.1], dtype=np.float32), {"entity_id": "e2"})
    hits = idx.search(np.array([1, 0], dtype=np.float32), k=5, filters={"entity_id": "e2"})
    assert len(hits) == 1 and hits[0][2]["entity_id"] == "e2"


def test_index_overwrite_same_id():
    idx = ExactNumpyIndex("m", 2)
    idx.add("a", np.array([1, 0], dtype=np.float32), {"v": 1})
    idx.add("a", np.array([0, 1], dtype=np.float32), {"v": 2})
    assert len(idx) == 1
    hits = idx.search(np.array([0, 1], dtype=np.float32), k=1)
    assert hits[0][2]["v"] == 2
