"""Embed chunks from `chunks.jsonl` and store them in the vector store.

Run as `python -m repo_recall.indexing.index_chunks [--batch-size N] [--limit N]`.
Re-runs are incremental: a chunk is skipped when a row with the same
`chunk_id` and the same content hash (text + model) is already stored, and
re-embedded when either changed. Each batch is committed on its own, so an
interrupted run resumes where it stopped.
"""

import argparse
import hashlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from repo_recall.config import Settings
from repo_recall.indexing.backends.postgres import PostgresStore
from repo_recall.indexing.embedder import Embedder, SentenceTransformerEmbedder
from repo_recall.indexing.store import VectorStore
from repo_recall.processing.chunk_all import read_jsonl
from repo_recall.processing.chunk_model import Chunk

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Outcome of one indexing run."""

    total: int = 0
    skipped: int = 0
    indexed: int = 0
    excluded: int = 0
    deleted: int = 0
    seconds: float = 0.0


def load_excluded_urls(eval_set_path: Path) -> set[str]:
    """Issue URLs in the evaluation set, which must never be searchable (empty if no set yet)."""
    if not eval_set_path.exists():
        return set()
    return {example["issue_url"] for example in read_jsonl(eval_set_path)}


def exclude_chunks(chunks: list[Chunk], excluded_urls: set[str]) -> list[Chunk]:
    """Chunks whose source isn't excluded."""
    return [c for c in chunks if c.url not in excluded_urls]


def content_hash(chunk: Chunk, model_name: str) -> str:
    """Fingerprint of what determines a chunk's vector: its text and the model."""
    return hashlib.sha256(f"{model_name}\n{chunk.text}".encode("utf-8")).hexdigest()


def select_pending(
    chunks: list[Chunk], hashes: dict[str, str], indexed: dict[str, str]
) -> list[Chunk]:
    """Chunks that are new, or whose text/model changed since they were stored."""
    return [c for c in chunks if indexed.get(c.chunk_id) != hashes[c.chunk_id]]


def batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Yield consecutive slices of `items` of length `size` (the last may be shorter)."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def index_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    store: VectorStore,
    batch_size: int,
    limit: int | None = None,
    excluded_urls: set[str] | None = None,
) -> IndexStats:
    """Embed and store every chunk not already indexed, logging progress per batch.

    Chunks from `excluded_urls` (the evaluation set's issues) are never
    embedded, and rows stored for them by earlier runs are deleted first,
    so evaluation questions can't retrieve their own answers.
    """
    started = time.perf_counter()
    excluded_urls = excluded_urls or set()
    deleted = store.delete_by_urls(excluded_urls) if excluded_urls else 0
    kept = exclude_chunks(chunks, excluded_urls)
    excluded = len(chunks) - len(kept)
    if excluded_urls:
        logger.info(
            "Excluding %d eval-set sources: skipped %d chunks, deleted %d stored rows",
            len(excluded_urls),
            excluded,
            deleted,
        )
    hashes = {c.chunk_id: content_hash(c, embedder.model_name) for c in kept}
    pending = select_pending(kept, hashes, store.indexed_hashes())
    stats = IndexStats(
        total=len(kept),
        skipped=len(kept) - len(pending),
        excluded=excluded,
        deleted=deleted,
    )
    if limit is not None:
        pending = pending[:limit]
    logger.info(
        "%d chunks total: %d already indexed, %d to embed", stats.total, stats.skipped, len(pending)
    )

    batch_count = -(-len(pending) // batch_size)
    for number, batch in enumerate(batched(pending, batch_size), start=1):
        embeddings = embedder.embed_documents([c.text for c in batch])
        store.upsert(batch, embeddings, [hashes[c.chunk_id] for c in batch], embedder.model_name)
        stats.indexed += len(batch)
        _log_progress(number, batch_count, stats.indexed, len(pending), started)

    stats.seconds = time.perf_counter() - started
    logger.info(
        "Indexed %d chunks, skipped %d, in %.1fs", stats.indexed, stats.skipped, stats.seconds
    )
    return stats


def _log_progress(batch: int, batches: int, done: int, todo: int, started: float) -> None:
    """Log one progress line: batch number, chunk count, throughput, elapsed time."""
    elapsed = time.perf_counter() - started
    logger.info(
        "Batch %d/%d · %d/%d chunks · %.1f chunks/s · %.0fs elapsed",
        batch,
        batches,
        done,
        todo,
        done / elapsed if elapsed else 0.0,
        elapsed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Embed chunks and store them for search.")
    parser.add_argument("--batch-size", type=int, default=None, help="chunks per embed/insert batch")
    parser.add_argument("--limit", type=int, default=None, help="max chunks to embed this run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: index `chunks.jsonl` into Postgres."""
    args = parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request HF Hub download logs

    batch_size = args.batch_size or settings.embed_batch_size
    chunks = [Chunk(**r) for r in read_jsonl(settings.data_processed_dir / "chunks.jsonl")]
    embedder = SentenceTransformerEmbedder(settings.embedding_model, batch_size)
    excluded_urls = load_excluded_urls(settings.eval_set_path)
    with PostgresStore(settings.postgres_dsn) as store:
        store.ensure_schema(embedder.dimension)
        index_chunks(
            chunks, embedder, store, batch_size, limit=args.limit, excluded_urls=excluded_urls
        )


if __name__ == "__main__":
    main()
