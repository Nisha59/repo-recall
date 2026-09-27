"""Integration tests against a real Postgres+pgvector (skipped if none is running)."""

import random
import uuid

import pytest

from repo_recall.indexing.backends.postgres import PostgresStore
from repo_recall.processing.chunk_model import Chunk
from repo_recall.retrieval.models import SearchMode
from repo_recall.retrieval.postgres_search import PostgresSearcher
from tests.indexing.backends.test_postgres import DSN, pytestmark  # noqa: F401  (skip if no DB)


def chunk(chunk_id: str, text: str, source_type: str = "doc") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_type=source_type,
        title=f"Title {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        date=None,
        text=text,
        token_count=len(text.split()),
    )


@pytest.fixture
def table():
    name = f"chunks_test_{uuid.uuid4().hex[:8]}"
    store = PostgresStore(DSN, table=name)
    store.ensure_schema(3)
    yield store, name
    store._connection().execute(f'DROP TABLE IF EXISTS "{name}"')
    store.close()


@pytest.fixture
def searcher(table):
    _, name = table
    with PostgresSearcher(DSN, table=name) as searcher:
        yield searcher


def test_vector_search_orders_by_similarity(table, searcher):
    store, _ = table
    store.upsert(
        [chunk("x", "a"), chunk("y", "b"), chunk("z", "c")],
        [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.0, 1.0]],
        ["h"] * 3,
        "m",
    )
    results = searcher.vector_search([1.0, 0.1, 0.0], k=3)
    assert [r.chunk_id for r in results] == ["x", "y", "z"]
    assert results[0].score > results[1].score > results[2].score
    assert results[0].mode is SearchMode.VECTOR and results[0].rank == 1


def test_vector_search_filter_returns_k_rare_rows_from_hnsw(table, searcher):
    store, _ = table
    rng = random.Random(0)
    docs = [chunk(f"doc{i}", "d") for i in range(300)]
    issues = [chunk(f"issue{i}", "i", "issue") for i in range(5)]
    doc_vecs = [[1.0, rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)] for _ in docs]
    issue_vecs = [[0.0, 1.0, rng.uniform(-0.1, 0.1)] for _ in issues]
    store.upsert(docs + issues, doc_vecs + issue_vecs, ["h"] * 305, "m")
    # Force the HNSW index so the test exercises post-filtering, not a seq scan.
    searcher._connection().execute("SET enable_seqscan = off")

    results = searcher.vector_search([1.0, 0.0, 0.0], k=5, source_type="issue")

    assert len(results) == 5
    assert all(r.source_type == "issue" for r in results)


def test_keyword_search_matches_any_term_and_ranks_more_matches_first(table, searcher):
    store, _ = table
    store.upsert(
        [
            chunk("both", "The kubelet talks to the scheduler about pods"),
            chunk("one", "The kubelet restarts containers"),
            chunk("none", "Completely unrelated gardening advice"),
        ],
        [[1.0, 0.0, 0.0]] * 3,
        ["h"] * 3,
        "m",
    )
    results = searcher.keyword_search("why does the kubelet ignore the scheduler", k=5)
    assert [r.chunk_id for r in results] == ["both", "one"]
    assert results[0].mode is SearchMode.KEYWORD


def test_keyword_search_filters_by_source_type(table, searcher):
    store, _ = table
    store.upsert(
        [chunk("d", "kubelet crash"), chunk("i", "kubelet crash", "issue")],
        [[1.0, 0.0, 0.0]] * 2,
        ["h"] * 2,
        "m",
    )
    results = searcher.keyword_search("kubelet crash", k=5, source_type="issue")
    assert [r.chunk_id for r in results] == ["i"]


def test_keyword_search_stopword_only_question_returns_nothing(table, searcher):
    store, _ = table
    store.upsert([chunk("a", "how do I")], [[1.0, 0.0, 0.0]], ["h"], "m")
    assert searcher.keyword_search("how do I", k=5) == []
