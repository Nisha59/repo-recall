"""Postgres + pgvector implementation of the VectorStore interface.

One table holds each chunk's text, citation metadata, its embedding (for
semantic search) and a generated tsvector (for keyword search), so
retrieval can do either — or both — against the same rows.
"""

import logging

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from repo_recall.processing.chunk_model import Chunk

logger = logging.getLogger(__name__)

# {table}/{*_index} are identifiers, {dim} is a trusted integer literal.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id        text PRIMARY KEY,
    source_type     text NOT NULL CHECK (source_type IN ('doc', 'issue')),
    title           text NOT NULL,
    url             text NOT NULL,
    date            timestamptz,
    heading_path    text[] NOT NULL DEFAULT '{{}}',
    labels          text[] NOT NULL DEFAULT '{{}}',
    extra           jsonb NOT NULL DEFAULT '{{}}',
    text            text NOT NULL,
    token_count     integer NOT NULL,
    content_hash    text NOT NULL,
    embedding_model text NOT NULL,
    embedding       vector({dim}) NOT NULL,
    search_tsv      tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', title), 'A') ||
                        setweight(to_tsvector('english', text), 'B')
                    ) STORED,
    indexed_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS {hnsw_index}
    ON {table} USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS {gin_index}
    ON {table} USING gin (search_tsv);
"""

_UPSERT_SQL = """
INSERT INTO {table} (
    chunk_id, source_type, title, url, date, heading_path, labels, extra,
    text, token_count, content_hash, embedding_model, embedding
) VALUES (
    %(chunk_id)s, %(source_type)s, %(title)s, %(url)s, %(date)s, %(heading_path)s,
    %(labels)s, %(extra)s, %(text)s, %(token_count)s, %(content_hash)s,
    %(embedding_model)s, %(embedding)s
)
ON CONFLICT (chunk_id) DO UPDATE SET
    source_type = EXCLUDED.source_type,
    title = EXCLUDED.title,
    url = EXCLUDED.url,
    date = EXCLUDED.date,
    heading_path = EXCLUDED.heading_path,
    labels = EXCLUDED.labels,
    extra = EXCLUDED.extra,
    text = EXCLUDED.text,
    token_count = EXCLUDED.token_count,
    content_hash = EXCLUDED.content_hash,
    embedding_model = EXCLUDED.embedding_model,
    embedding = EXCLUDED.embedding,
    indexed_at = now()
"""


def connect(dsn: str) -> psycopg.Connection:
    """Open a pgvector-ready connection to `dsn`.

    Makes sure the vector type exists before registering it with psycopg.
    Autocommit mode, so each `with conn.transaction()` block is a real
    BEGIN/COMMIT rather than a savepoint inside an implicit transaction.
    """
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


class PostgresStore:
    """Stores chunks and their embeddings in one Postgres table.

    Holds a single connection, opened on first use; call `close()` (or use
    it as a context manager) when done.
    """

    def __init__(self, dsn: str, table: str = "chunks") -> None:
        self._dsn = dsn
        self._table = table
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> "PostgresStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the database connection if one is open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> psycopg.Connection:
        """Return the open connection, connecting on first use."""
        if self._conn is None:
            self._conn = connect(self._dsn)
        return self._conn

    def _format(self, template: str, **literals: sql.Composable) -> sql.Composed:
        """Fill a SQL template with this store's table and index identifiers."""
        return sql.SQL(template).format(
            table=sql.Identifier(self._table),
            hnsw_index=sql.Identifier(f"{self._table}_embedding_hnsw"),
            gin_index=sql.Identifier(f"{self._table}_search_tsv_gin"),
            **literals,
        )

    def ensure_schema(self, dimension: int) -> None:
        """Create the table and its indexes if missing; fail on a vector-size mismatch."""
        conn = self._connection()
        with conn.transaction():
            conn.execute(self._format(_SCHEMA_DDL, dim=sql.Literal(dimension)))
        existing = self.stored_dimension()
        if existing != dimension:
            raise ValueError(
                f"Table {self._table!r} stores {existing}-dim vectors but the embedding "
                f"model produces {dimension}-dim ones; use a new table or drop the old one."
            )

    def stored_dimension(self) -> int | None:
        """The declared size of the table's `vector` column (None if the table is missing)."""
        row = self._connection().execute(
            """
            SELECT a.atttypmod FROM pg_attribute a
            WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding'
            """,
            (self._table,),
        ).fetchone()
        return row[0] if row else None

    def indexed_hashes(self) -> dict[str, str]:
        """Return `{chunk_id: content_hash}` for every stored chunk."""
        cursor = self._connection().execute(
            self._format("SELECT chunk_id, content_hash FROM {table}")
        )
        return dict(cursor.fetchall())

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        content_hashes: list[str],
        model_name: str,
    ) -> None:
        """Insert or replace a batch of chunks in one transaction."""
        rows = [
            _row(chunk, embedding, content_hash, model_name)
            for chunk, embedding, content_hash in zip(chunks, embeddings, content_hashes, strict=True)
        ]
        conn = self._connection()
        with conn.transaction(), conn.cursor() as cursor:
            cursor.executemany(self._format(_UPSERT_SQL), rows)

    def delete_by_urls(self, urls: set[str]) -> int:
        """Delete every chunk of the given sources in one transaction; return the row count."""
        if not urls:
            return 0
        conn = self._connection()
        with conn.transaction():
            cursor = conn.execute(
                self._format("DELETE FROM {table} WHERE url = ANY(%s)"), (sorted(urls),)
            )
        return cursor.rowcount


def _row(chunk: Chunk, embedding: list[float], content_hash: str, model_name: str) -> dict:
    """Map a chunk and its vector to the upsert statement's parameters."""
    return {
        "chunk_id": chunk.chunk_id,
        "source_type": chunk.source_type,
        "title": chunk.title,
        "url": chunk.url,
        "date": chunk.date,
        "heading_path": chunk.heading_path,
        "labels": chunk.labels,
        "extra": Jsonb(chunk.extra),
        "text": chunk.text,
        "token_count": chunk.token_count,
        "content_hash": content_hash,
        "embedding_model": model_name,
        "embedding": np.asarray(embedding, dtype=np.float32),
    }
