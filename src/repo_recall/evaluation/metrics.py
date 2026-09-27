"""Retrieval and latency metrics: pure functions over ranked URLs and timings.

Relevance is judged per source (a docs page or an issue), not per chunk: a
question's relevant sources are the indexed pages its reference answer
links to. A source's rank is the rank of its first chunk in the results,
so recall@5 describes exactly the chunks the answer model sees.
"""

from urllib.parse import urlsplit

import numpy as np


def normalize_url(url: str) -> str:
    """Compare-friendly URL: lowercase host without www., no query/anchor, no trailing slash."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}"


def relevant_sources(linked_urls: list[str], indexed_urls: set[str]) -> set[str]:
    """The linked URLs (normalized) that exist in the index; others can't be retrieved."""
    linked = {normalize_url(url) for url in linked_urls}
    return linked & indexed_urls


def source_ranks(retrieved_urls: list[str]) -> dict[str, int]:
    """1-based rank of each source's first chunk in a ranked list of chunk URLs."""
    ranks: dict[str, int] = {}
    for rank, url in enumerate(retrieved_urls, start=1):
        ranks.setdefault(normalize_url(url), rank)
    return ranks


def recall_at_k(ranks: dict[str, int], relevant: set[str], k: int) -> float:
    """Share of relevant sources with a chunk in the top k."""
    if not relevant:
        raise ValueError("recall is undefined without relevant sources")
    return sum(1 for url in relevant if ranks.get(url, k + 1) <= k) / len(relevant)


def reciprocal_rank(ranks: dict[str, int], relevant: set[str]) -> float:
    """1 / rank of the best-ranked relevant source, or 0 if none was retrieved."""
    hits = [ranks[url] for url in relevant if url in ranks]
    return 1 / min(hits) if hits else 0.0


def percentile(values: list[float], p: float) -> float | None:
    """The p-th percentile (0-100, linear interpolation), or None for no values."""
    return float(np.percentile(values, p)) if values else None


def mean(values: list[float]) -> float | None:
    """Arithmetic mean, or None for no values."""
    return sum(values) / len(values) if values else None
