import pytest

from repo_recall.evaluation.metrics import (
    mean,
    normalize_url,
    percentile,
    recall_at_k,
    reciprocal_rank,
    relevant_sources,
    source_ranks,
)

A = "https://kubernetes.io/docs/a/"
B = "https://kubernetes.io/docs/b/"
C = "https://kubernetes.io/docs/c/"


def test_normalize_url_ignores_scheme_www_anchor_query_and_trailing_slash():
    assert normalize_url("http://www.Kubernetes.io/docs/a/#probes?x=1") == "kubernetes.io/docs/a"
    assert normalize_url(A) == normalize_url("https://kubernetes.io/docs/a")


def test_relevant_sources_are_linked_urls_that_are_indexed():
    indexed = {normalize_url(A), normalize_url(B)}
    assert relevant_sources([A + "#x", C], indexed) == {normalize_url(A)}


def test_source_rank_is_the_rank_of_its_first_chunk():
    ranks = source_ranks([A, A, B, C, B])
    assert ranks == {normalize_url(A): 1, normalize_url(B): 3, normalize_url(C): 4}


def test_recall_and_rr_with_several_relevant_sources():
    ranks = source_ranks([C, C, A, "x", "y", "z", B])  # A at 3, B at 7
    relevant = {normalize_url(A), normalize_url(B)}
    assert recall_at_k(ranks, relevant, 5) == 0.5
    assert recall_at_k(ranks, relevant, 10) == 1.0
    assert reciprocal_rank(ranks, relevant) == pytest.approx(1 / 3)


def test_nothing_relevant_retrieved_scores_zero():
    ranks = source_ranks([C])
    assert recall_at_k(ranks, {normalize_url(A)}, 10) == 0.0
    assert reciprocal_rank(ranks, {normalize_url(A)}) == 0.0


def test_recall_requires_relevant_sources():
    with pytest.raises(ValueError):
        recall_at_k({}, set(), 5)


def test_percentile_and_mean():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == pytest.approx(80.8)
    assert percentile([], 50) is None
    assert mean([1.0, 2.0]) == 1.5 and mean([]) is None
