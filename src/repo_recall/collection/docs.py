"""Collect a repo's Markdown docs into a clean, citable JSONL file."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from repo_recall.collection.repo import clone_or_update_repo, head_commit_date
from repo_recall.config import Settings
from repo_recall.processing.markdown import (
    clean_body,
    extract_title,
    parse_front_matter,
    path_to_url,
)

logger = logging.getLogger(__name__)


def iter_markdown_files(docs_dir: Path) -> Iterator[Path]:
    """Yield every `.md` file under `docs_dir`."""
    yield from docs_dir.rglob("*.md")


def build_doc_record(
    md_path: Path, docs_root: Path, base_url: str, snapshot_date: str
) -> dict | None:
    """Build a `{title, text, path, url, date}` record for one doc page.

    `date` is the docs snapshot's commit date, shared by every page.

    Returns `None` if the file can't be read/decoded or has no usable
    content once cleaned.
    """
    try:
        raw_text = md_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning("Skipping %s: could not read file", md_path)
        return None

    front_matter, body = parse_front_matter(raw_text)
    text = clean_body(body)
    if not text:
        logger.warning("Skipping %s: no content after cleaning", md_path)
        return None

    relative_path = PurePosixPath(md_path.relative_to(docs_root).as_posix())
    return {
        "title": extract_title(front_matter, relative_path),
        "text": text,
        "path": str(relative_path),
        "url": path_to_url(relative_path, base_url),
        "date": snapshot_date,
    }


def collect_docs(settings: Settings) -> tuple[int, int]:
    """Clone/update the configured docs repo and write cleaned pages to JSONL.

    Returns `(processed_count, skipped_count)`.
    """
    repo_name = settings.docs_repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    repo_dest = settings.data_raw_dir / repo_name
    clone_or_update_repo(settings.docs_repo_url, repo_dest)
    snapshot_date = head_commit_date(repo_dest)

    docs_dir = repo_dest / settings.docs_content_path
    output_path = settings.data_processed_dir / "docs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out_file:
        for md_path in iter_markdown_files(docs_dir):
            record = build_doc_record(
                md_path, docs_dir, settings.docs_public_base_url, snapshot_date
            )
            if record is None:
                skipped += 1
                continue
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

    logger.info("Processed %d pages, skipped %d, wrote %s", processed, skipped, output_path)
    return processed, skipped


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    collect_docs(settings)


if __name__ == "__main__":
    main()
