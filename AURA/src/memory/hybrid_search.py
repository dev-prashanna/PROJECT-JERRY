"""Reciprocal Rank Fusion over heterogeneous memory rankings.

Keyword (BM25/FTS), semantic (cosine) and recency rankings live on
incomparable scales; RRF fuses them through rank only:

    rrf_score = sum(1.0 / (k + rank))   over each source's ranking

The constant k damps the contribution of very low-ranked items; k=60 is the
original Cormack et al. value and works well as a starting point here.
"""

from __future__ import annotations

from typing import Any, Iterable


def rrf_fuse(
    ranked_lists: dict[str, Iterable[Any]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[Any, float, dict[str, Any]]]:
    """Fuse ranked iterables into (item, score, per_source_meta) sorted best-first.

    Each input is an iterable of items ordered best-first (rank 0 = best).
    ``weights`` optionally scales each source's contribution.
    """
    scores: dict[Any, float] = {}
    meta: dict[Any, dict[str, Any]] = {}
    for source, items in ranked_lists.items():
        weight = 1.0 if not weights else float(weights.get(source, 1.0))
        for rank, item in enumerate(items):
            key = _key(item)
            contribution = weight / (k + rank + 1)
            scores[key] = scores.get(key, 0.0) + contribution
            slot = meta.setdefault(key, {})
            slot[source] = {"rank": rank, "contribution": contribution}
    keyed_items = {_key(item): item for items in ranked_lists.values() for item in items}
    fused = [
        (keyed_items[key], score, meta[key]) for key, score in scores.items()
    ]
    fused.sort(key=lambda t: t[1], reverse=True)
    return fused


def _key(item: Any) -> Any:
    # ConversationEvent / Episode / RetrievedEvent all carry a stable id field
    for attr in ("event_id", "episode_id", "embedding_id"):
        value = getattr(item, attr, None)
        if value is not None:
            return value
    if isinstance(item, dict):
        for k in ("event_id", "episode_id", "id"):
            if k in item:
                return item[k]
    return id(item)
