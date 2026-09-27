"""Chunk the collected docs and issues into `data/processed/chunks.jsonl`.

Run as `python -m repo_recall.processing.chunk_all`. Prints chunk counts and
the token-length distribution when done.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from repo_recall.collection.issues import output_paths
from repo_recall.config import Settings
from repo_recall.processing.chunk_model import Chunk, ChunkSizes
from repo_recall.processing.chunk_stats import compute_stats, format_stats
from repo_recall.processing.doc_chunker import build_breadcrumbs, chunk_doc
from repo_recall.processing.issue_chunker import chunk_issue
from repo_recall.processing.tokens import HFTokenCounter, TokenCounter

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    """Load every non-blank line of a JSONL file, or `[]` if the file is missing."""
    if not path.exists():
        logger.warning("%s not found; skipping", path)
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_docs(records: list[dict], sizes: ChunkSizes, counter: TokenCounter) -> Iterator[Chunk]:
    """Yield chunks for every doc record."""
    breadcrumbs = build_breadcrumbs(records)
    for record in records:
        yield from chunk_doc(record, breadcrumbs[record["path"]], sizes, counter)


def chunk_issues(
    records: list[dict],
    repo: str,
    sizes: ChunkSizes,
    max_comments: int,
    counter: TokenCounter,
) -> Iterator[Chunk]:
    """Yield chunks for every issue record."""
    for record in records:
        yield from chunk_issue(record, repo, sizes, max_comments, counter)


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks as JSONL, one chunk per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def build_all_chunks(settings: Settings, counter: TokenCounter) -> list[Chunk]:
    """Chunk the configured docs and issues sources."""
    sizes = ChunkSizes(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
    )
    docs = read_jsonl(settings.data_processed_dir / "docs.jsonl")
    issues = read_jsonl(output_paths(settings)[0])
    logger.info("Chunking %d doc pages and %d issues", len(docs), len(issues))
    return [
        *chunk_docs(docs, sizes, counter),
        *chunk_issues(issues, settings.github_repo, sizes, settings.issue_max_comments, counter),
    ]


def main() -> None:
    """CLI entry point: chunk everything, write JSONL, print stats."""
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request HF Hub download logs
    chunks = build_all_chunks(settings, HFTokenCounter(settings.embedding_model))
    output_path = settings.data_processed_dir / "chunks.jsonl"
    write_chunks(chunks, output_path)
    logger.info("Wrote %d chunks to %s", len(chunks), output_path)
    print(format_stats(compute_stats(chunks, settings.chunk_max_tokens)))


if __name__ == "__main__":
    main()
