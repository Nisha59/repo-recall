"""Integration tests against a real Postgres+pgvector (skipped if none is running).

Start one with `docker compose up -d`. Each test uses its own throwaway
table, so real indexed data is never touched.
"""

import uuid

import psycopg
import pytest

from repo_recall.config import Settings
from repo_recall.indexing.backends.postgres import PostgresStore

DSN = Settings().postgres_dsn


def _postgres_available() -> bool:
    try:
        psycopg.connect(DSN, connect_timeout=2).close()
    except psycopg.OperationalError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable (run `docker compose up -d`)"
)


@pytest.fixture
def store():
    table = f"chunks_test_{uuid.uuid4().hex[:8]}"
    store = PostgresStore(DSN, table=table)
    store.ensure_schema(3)
    yield store
    store._connection().execute(f'DROP TABLE IF EXISTS "{table}"')
    store.close()


def test_ensure_schema_is_idempotent_and_creates_indexes(store):
    store.ensure_schema(3)
    rows = store._connection().execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = %s", (store._table,)
    ).fetchall()
    names = {r[0] for r in rows}
    assert f"{store._table}_embedding_hnsw" in names
    assert f"{store._table}_search_tsv_gin" in names


def test_ensure_schema_rejects_dimension_mismatch(store):
    with pytest.raises(ValueError, match="3-dim"):
        store.ensure_schema(384)


def test_upsert_round_trips_metadata_and_fills_search_tsv(store, chunk_factory):
    chunk = chunk_factory("issue:o/r#1:0", "The kubelet crashes on restart", source_type="issue")
    store.upsert([chunk], [[1.0, 0.0, 0.0]], ["hash1"], "m")

    row = store._connection().execute(
        f'SELECT title, labels, extra, date, search_tsv::text FROM "{store._table}"'
    ).fetchone()
    title, labels, extra, date, tsv = row
    assert title == "Title issue:o/r#1:0"
    assert labels == ["kind/bug"]
    assert extra == {"path": "concepts/pods.md"}
    assert date.year == 2026
    assert "'kubelet'" in tsv and "'crash'" in tsv
    assert store.indexed_hashes() == {"issue:o/r#1:0": "hash1"}


def test_upsert_replaces_existing_row(store, chunk_factory):
    store.upsert([chunk_factory("a", "old")], [[1.0, 0.0, 0.0]], ["h1"], "m")
    store.upsert([chunk_factory("a", "new")], [[0.0, 1.0, 0.0]], ["h2"], "m")

    rows = store._connection().execute(f'SELECT text FROM "{store._table}"').fetchall()
    assert rows == [("new",)]
    assert store.indexed_hashes() == {"a": "h2"}


def test_nearest_neighbour_query_returns_closest_row(store, chunk_factory):
    store.upsert(
        [chunk_factory("x"), chunk_factory("y"), chunk_factory("z")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ["hx", "hy", "hz"],
        "m",
    )
    nearest = store._connection().execute(
        f'SELECT chunk_id FROM "{store._table}" ORDER BY embedding <=> %s::vector LIMIT 1',
        ("[0.1, 0.9, 0.0]",),
    ).fetchone()
    assert nearest == ("y",)


def test_delete_by_urls_removes_every_chunk_of_those_sources(store, chunk_factory):
    chunks = [chunk_factory("a"), chunk_factory("b"), chunk_factory("c")]
    store.upsert(chunks, [[1.0, 0.0, 0.0]] * 3, ["h1", "h2", "h3"], "m")

    deleted = store.delete_by_urls({"https://example.com/a", "https://example.com/c"})

    assert deleted == 2
    assert set(store.indexed_hashes()) == {"b"}
    assert store.delete_by_urls(set()) == 0
