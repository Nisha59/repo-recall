"""Keep one long source from filling every result slot.

Long docs pages and issues are split into several chunks, and those
chunks tend to rank together. Capping chunks per source (by URL: one
docs page or one issue) leaves room for other sources in the top-k.
"""

from dataclasses import replace

from repo_recall.retrieval.models import SearchResult


def limit_per_source(
    results: list[SearchResult], max_per_source: int, k: int
) -> list[SearchResult]:
    """Top-k of `results` (in rank order) with at most `max_per_source` chunks per URL.

    Results are re-ranked 1..n after dropping; `max_per_source <= 0` means no cap.
    """
    kept: list[SearchResult] = []
    per_source: dict[str, int] = {}
    for result in results:
        if len(kept) == k:
            break
        count = per_source.get(result.url, 0)
        if max_per_source > 0 and count >= max_per_source:
            continue
        per_source[result.url] = count + 1
        kept.append(result)
    return [replace(result, rank=rank) for rank, result in enumerate(kept, start=1)]
