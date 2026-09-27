"""The storage interface every vector store backend implements."""

from typing import Protocol

from repo_recall.processing.chunk_model import Chunk


class VectorStore(Protocol):
    """Where embedded chunks live. Backends: Postgres+pgvector (LanceDB later)."""

    def ensure_schema(self, dimension: int) -> None:
        """Create tables/indexes if missing; safe to call on every run."""
        ...

    def indexed_hashes(self) -> dict[str, str]:
        """Return `{chunk_id: content_hash}` for everything already stored."""
        ...

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        content_hashes: list[str],
        model_name: str,
    ) -> None:
        """Insert or replace chunks with their vectors, durably (one commit per call)."""
        ...

    def delete_by_urls(self, urls: set[str]) -> int:
        """Delete every chunk whose source URL is in `urls`; return how many were removed."""
        ...
