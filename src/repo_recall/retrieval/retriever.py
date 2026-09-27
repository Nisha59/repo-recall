"""The retrieval entry point: question in, ranked chunks out."""

from typing import Protocol

from repo_recall.config import Settings
from repo_recall.retrieval.diversity import limit_per_source
from repo_recall.retrieval.fusion import reciprocal_rank_fusion
from repo_recall.retrieval.models import SearchMode, SearchResult
from repo_recall.retrieval.query_encoder import QueryEncoder


class SearchBackend(Protocol):
    """A store that supports both search types (Postgres today, LanceDB later)."""

    def vector_search(
        self, query_vector: list[float], k: int, source_type: str | None = None
    ) -> list[SearchResult]:
        """Top-k chunks by embedding similarity."""
        ...

    def keyword_search(
        self, question: str, k: int, source_type: str | None = None
    ) -> list[SearchResult]:
        """Top-k chunks by full-text relevance."""
        ...


class Retriever:
    """Finds the chunks most relevant to a question using the configured mode."""

    def __init__(self, backend: SearchBackend, encoder: QueryEncoder, settings: Settings) -> None:
        self._backend = backend
        self._encoder = encoder
        self._settings = settings

    def search(
        self,
        question: str,
        mode: SearchMode | str | None = None,
        top_k: int | None = None,
        source_type: str | None = None,
    ) -> list[SearchResult]:
        """Return the top-k chunks for `question`.

        `mode` and `top_k` default to the configured values; `source_type`
        ("doc" or "issue") restricts results to one kind of source. At most
        `retrieval_max_per_source` chunks come from any one page or issue.
        """
        mode = SearchMode(mode or self._settings.retrieval_mode)
        k = top_k or self._settings.retrieval_top_k
        max_per_source = self._settings.retrieval_max_per_source
        # With a per-source cap some rows get dropped, so fetch extra to still fill k.
        fetch = max(k, self._settings.retrieval_candidates) if max_per_source > 0 else k
        results = self._ranked(question, mode, fetch, source_type)
        return limit_per_source(results, max_per_source, k)

    def _ranked(
        self, question: str, mode: SearchMode, k: int, source_type: str | None
    ) -> list[SearchResult]:
        """The top-k chunks from one search mode, before the per-source cap."""
        if mode is SearchMode.VECTOR:
            return self._backend.vector_search(self._encoder.encode(question), k, source_type)
        if mode is SearchMode.KEYWORD:
            return self._backend.keyword_search(question, k, source_type)
        return self._hybrid(question, k, source_type)

    def _hybrid(self, question: str, k: int, source_type: str | None) -> list[SearchResult]:
        """Fuse a wide candidate list from each search with Reciprocal Rank Fusion."""
        candidates = max(k, self._settings.retrieval_candidates)
        vector_query = self._encoder.encode(question)
        return reciprocal_rank_fusion(
            {
                SearchMode.VECTOR.value: self._backend.vector_search(
                    vector_query, candidates, source_type
                ),
                SearchMode.KEYWORD.value: self._backend.keyword_search(
                    question, candidates, source_type
                ),
            },
            k=k,
            rrf_k=self._settings.rrf_k,
        )
