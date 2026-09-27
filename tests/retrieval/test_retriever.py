from dataclasses import replace

import pytest

from repo_recall.config import Settings
from repo_recall.retrieval.models import SearchMode
from repo_recall.retrieval.retriever import Retriever

from .conftest import ranked


class FakeBackend:
    """Records calls; returns canned ranked lists."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def vector_search(self, query_vector, k, source_type=None):
        self.calls.append(("vector", tuple(query_vector), k, source_type))
        return ranked(["v1", "both", "v3"])[:k]

    def keyword_search(self, question, k, source_type=None):
        self.calls.append(("keyword", question, k, source_type))
        return ranked(["k1", "both"], SearchMode.KEYWORD)[:k]


class FakeEncoder:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def encode(self, question: str) -> list[float]:
        self.questions.append(question)
        return [0.1, 0.2]


def make_retriever(mode: str = "hybrid", **overrides):
    settings = Settings(retrieval_mode=mode, retrieval_top_k=3, retrieval_candidates=20, **overrides)
    backend, encoder = FakeBackend(), FakeEncoder()
    return Retriever(backend, encoder, settings), backend, encoder


def test_vector_mode_embeds_question_and_calls_vector_search():
    retriever, backend, encoder = make_retriever()
    results = retriever.search("q", mode="vector", top_k=2, source_type="doc")
    assert encoder.questions == ["q"]
    assert backend.calls == [("vector", (0.1, 0.2), 20, "doc")]
    assert [r.chunk_id for r in results] == ["v1", "both"]


def test_keyword_mode_never_embeds():
    retriever, backend, encoder = make_retriever()
    retriever.search("q", mode=SearchMode.KEYWORD, source_type="issue")
    assert encoder.questions == []
    assert backend.calls == [("keyword", "q", 20, "issue")]


def test_default_mode_and_top_k_come_from_settings():
    retriever, backend, _ = make_retriever(mode="keyword", retrieval_max_per_source=0)
    results = retriever.search("q")
    assert backend.calls == [("keyword", "q", 3, None)]
    assert len(results) == 2


def test_hybrid_fetches_candidates_from_both_and_fuses():
    retriever, backend, encoder = make_retriever()
    results = retriever.search("q", source_type="doc")
    assert {c[0] for c in backend.calls} == {"vector", "keyword"}
    assert all(c[2] == 20 and c[3] == "doc" for c in backend.calls)
    assert encoder.questions == ["q"]
    assert results[0].chunk_id == "both"
    assert results[0].component_ranks == {"vector": 2, "keyword": 2}
    assert len(results) == 3


def test_results_are_capped_per_source():
    retriever, backend, _ = make_retriever(mode="vector", retrieval_max_per_source=1)
    backend.vector_search = lambda query_vector, k, source_type=None: [
        replace(r, url="same-issue" if r.chunk_id != "d" else "doc")
        for r in ranked(["a", "b", "c", "d"])
    ]
    results = retriever.search("q")
    assert [r.chunk_id for r in results] == ["a", "d"]
    assert [r.rank for r in results] == [1, 2]


def test_invalid_mode_raises():
    retriever, _, _ = make_retriever()
    with pytest.raises(ValueError):
        retriever.search("q", mode="fuzzy")

