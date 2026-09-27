"""Vector and keyword search over the Postgres `chunks` table."""

import threading

import numpy as np
import psycopg
from psycopg import sql

from repo_recall.indexing.backends.postgres import connect
from repo_recall.retrieval.models import SearchMode, SearchResult

_COLUMNS = "chunk_id, source_type, title, url, date, heading_path, labels, extra, text"

# `%(source_type)s` NULL means "no filter".
_SOURCE_FILTER = "(%(source_type)s::text IS NULL OR source_type = %(source_type)s)"

# relaxed_order lets HNSW return rows slightly out of order, so the outer
# query re-sorts the (small) materialized result by exact distance.
_VECTOR_SQL = f"""
WITH nearest AS MATERIALIZED (
    SELECT {_COLUMNS}, embedding <=> %(query)s AS distance
    FROM {{table}}
    WHERE {_SOURCE_FILTER}
    ORDER BY embedding <=> %(query)s
    LIMIT %(k)s
)
SELECT {_COLUMNS}, 1 - distance AS score FROM nearest ORDER BY distance
"""

# plainto_tsquery ANDs every word; swapping to OR lets a question match
# chunks that contain only some of its terms, and ranking puts chunks
# matching more (and title-weighted) terms first.
_TSQUERY = "replace(plainto_tsquery('english', %(question)s)::text, ' & ', ' | ')::tsquery"

# ts_rank, not ts_rank_cd: real questions paste logs and run to 100+ terms,
# which OR-match nearly every chunk. ts_rank_cd's proximity scoring then
# took 50-80s per query on the 18k-chunk index; ts_rank takes ~1-3s.
_KEYWORD_SQL = f"""
SELECT {_COLUMNS}, ts_rank(search_tsv, {_TSQUERY}, 1) AS score
FROM {{table}}
WHERE search_tsv @@ {_TSQUERY} AND {_SOURCE_FILTER}
ORDER BY score DESC, chunk_id
LIMIT %(k)s
"""


class PostgresSearcher:
    """Runs vector and keyword searches against one chunks table.

    Holds one connection, opened on first use and reopened if it was lost
    (e.g. the database restarted). Safe to share between threads: queries
    take turns on the connection.
    """

    def __init__(self, dsn: str, table: str = "chunks") -> None:
        self._dsn = dsn
        self._table = sql.Identifier(table)
        self._conn: psycopg.Connection | None = None
        self._lock = threading.RLock()

    def __enter__(self) -> "PostgresSearcher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the database connection if one is open."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _connection(self) -> psycopg.Connection:
        """Return the open connection, (re)connecting if there is none or it was lost."""
        if self._conn is None or self._conn.closed or self._conn.broken:
            self._conn = connect(self._dsn)
        return self._conn

    def ping(self) -> bool:
        """True if the database answers a trivial query."""
        try:
            with self._lock:
                self._connection().execute("SELECT 1")
        except psycopg.Error:
            return False
        return True

    def vector_search(
        self, query_vector: list[float], k: int, source_type: str | None = None
    ) -> list[SearchResult]:
        """Top-k chunks by cosine similarity to `query_vector`.

        HNSW normally finds ~ef_search candidates and filters them afterwards,
        which returns too few rows when filtering to a rare source type.
        Iterative scanning keeps walking the graph until k rows pass the
        filter; both settings are transaction-local.
        """
        params = {
            "query": np.asarray(query_vector, dtype=np.float32),
            "k": k,
            "source_type": source_type,
        }
        with self._lock:
            conn = self._connection()
            with conn.transaction():
                conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
                conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(40, k)),))
                rows = conn.execute(sql.SQL(_VECTOR_SQL).format(table=self._table), params).fetchall()
        return _to_results(rows, SearchMode.VECTOR)

    def indexed_urls(self) -> set[str]:
        """Every distinct source URL in the table (used to tell what can be retrieved)."""
        with self._lock:
            rows = (
                self._connection()
                .execute(sql.SQL("SELECT DISTINCT url FROM {table}").format(table=self._table))
                .fetchall()
            )
        return {url for (url,) in rows}

    def keyword_search(
        self, question: str, k: int, source_type: str | None = None
    ) -> list[SearchResult]:
        """Top-k chunks by full-text rank for any of the question's terms."""
        params = {"question": question, "k": k, "source_type": source_type}
        with self._lock:
            rows = (
                self._connection()
                .execute(sql.SQL(_KEYWORD_SQL).format(table=self._table), params)
                .fetchall()
            )
        return _to_results(rows, SearchMode.KEYWORD)


def _to_results(rows: list[tuple], mode: SearchMode) -> list[SearchResult]:
    """Map result rows (in rank order) to SearchResults."""
    return [
        SearchResult(
            chunk_id=chunk_id,
            source_type=source_type,
            title=title,
            url=url,
            date=date,
            heading_path=heading_path,
            labels=labels,
            extra=extra,
            text=text,
            score=float(score),
            rank=rank,
            mode=mode,
        )
        for rank, (
            chunk_id, source_type, title, url, date, heading_path, labels, extra, text, score
        ) in enumerate(rows, start=1)
    ]
