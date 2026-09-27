from dataclasses import replace

from repo_recall.retrieval.diversity import limit_per_source

from .conftest import ranked


def from_sources(*urls: str):
    """Ranked results whose i-th chunk belongs to source urls[i]."""
    results = ranked([f"c{i}" for i in range(len(urls))])
    return [replace(r, url=url) for r, url in zip(results, urls)]


def test_caps_chunks_from_one_source_and_backfills_from_others():
    results = from_sources("issue/1", "issue/1", "issue/1", "doc/a", "issue/1", "doc/b")
    kept = limit_per_source(results, max_per_source=2, k=4)
    assert [r.url for r in kept] == ["issue/1", "issue/1", "doc/a", "doc/b"]
    assert [r.chunk_id for r in kept] == ["c0", "c1", "c3", "c5"]


def test_kept_results_are_reranked_in_order():
    results = from_sources("a", "a", "a", "b")
    assert [r.rank for r in limit_per_source(results, max_per_source=1, k=5)] == [1, 2]


def test_returns_fewer_than_k_when_sources_run_out():
    assert len(limit_per_source(from_sources("a", "a", "a"), max_per_source=2, k=5)) == 2


def test_zero_means_no_cap_just_top_k():
    results = from_sources("a", "a", "a", "a")
    assert [r.chunk_id for r in limit_per_source(results, max_per_source=0, k=3)] == [
        "c0",
        "c1",
        "c2",
    ]


def test_empty_input():
    assert limit_per_source([], max_per_source=2, k=5) == []
