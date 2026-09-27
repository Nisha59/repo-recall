"""The search modes and the result record every search returns."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class SearchMode(StrEnum):
    """How to find chunks for a question."""

    VECTOR = "vector"  # semantic similarity of embeddings (pgvector)
    KEYWORD = "keyword"  # Postgres full-text search
    HYBRID = "hybrid"  # both, combined with Reciprocal Rank Fusion


@dataclass
class SearchResult:
    """One retrieved chunk with its score, rank and citation metadata.

    `score` means different things per mode (cosine similarity, text rank,
    RRF score), so compare scores only within one mode. `component_ranks`
    is set for hybrid results: the chunk's rank in each underlying search.
    """

    chunk_id: str
    source_type: str
    title: str
    url: str
    date: datetime | None
    heading_path: list[str]
    labels: list[str]
    extra: dict
    text: str
    score: float
    rank: int
    mode: SearchMode
    component_ranks: dict[str, int] = field(default_factory=dict)
