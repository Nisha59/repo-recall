"""Reciprocal Rank Fusion: merge several ranked result lists into one.

Each chunk scores sum(1 / (rrf_k + rank)) over the lists it appears in.
Only ranks are used, never raw scores, because cosine similarity and
full-text rank live on unrelated scales. A chunk found by both searches
beats one found by just one of them at a similar rank.
"""

from dataclasses import replace

from repo_recall.retrieval.models import SearchMode, SearchResult


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[SearchResult]], k: int, rrf_k: int = 60
) -> list[SearchResult]:
    """Fuse `{list_name: results}` into the top-k hybrid results.

    Each returned result's `component_ranks` records its rank in every list
    that contained it, keyed by list name. Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    first_seen: dict[str, SearchResult] = {}
    for name, results in ranked_lists.items():
        for result in results:
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1 / (rrf_k + result.rank)
            ranks.setdefault(result.chunk_id, {})[name] = result.rank
            first_seen.setdefault(result.chunk_id, result)

    ordered = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [
        replace(
            first_seen[chunk_id],
            score=scores[chunk_id],
            rank=rank,
            mode=SearchMode.HYBRID,
            component_ranks=ranks[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
