"""Chunk cleaned doc pages by Markdown heading, then by size.

Each chunk's embedded text starts with its full heading path (site
breadcrumb + page title + in-page headings), e.g.
"Concepts > Workloads > Pods > Pod lifecycle", so a section still makes
sense on its own once it's pulled out of the page.
"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from repo_recall.processing.chunk_model import Chunk, ChunkSizes
from repo_recall.processing.text_units import (
    BLOCK_SEPARATOR,
    is_fence_line,
    pack_units,
    split_blocks,
)
from repo_recall.processing.tokens import TokenCounter

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HEADING_ANCHOR_RE = re.compile(r"\s*\{#[^}]*\}")

HEADING_PATH_SEPARATOR = " > "


@dataclass
class Section:
    """A run of body text under one heading, with the headings above it."""

    headings: tuple[str, ...]
    body: str


def parse_heading(line: str) -> tuple[int, str] | None:
    """Return `(level, title)` if `line` is an ATX Markdown heading, else None."""
    match = _HEADING_RE.match(line)
    if not match:
        return None
    title = _HEADING_ANCHOR_RE.sub("", match.group(2)).strip()
    return (len(match.group(1)), title) if title else None


def split_sections(text: str) -> list[Section]:
    """Split a page body into sections at each heading.

    `#` lines inside fenced code blocks (e.g. shell comments) are not
    headings. Sections with no body text (a heading followed directly by a
    subheading) are dropped; their title lives on in the subsection's path.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append(Section(tuple(title for _, title in stack), body))
        lines.clear()

    for line in text.split("\n"):
        if is_fence_line(line):
            in_fence = not in_fence
        heading = None if in_fence else parse_heading(line)
        if heading is None:
            lines.append(line)
            continue
        flush()
        level, title = heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    flush()
    return sections


def _prettify_dir_name(name: str) -> str:
    """Turn a directory name like `cluster-administration` into a readable title."""
    return name.replace("-", " ").replace("_", " ").strip().title()


def build_breadcrumbs(records: list[dict]) -> dict[str, list[str]]:
    """Map each doc's path to the titles of the sections above it on the site.

    Section titles come from the `_index.md` page of each ancestor
    directory, falling back to a prettified directory name. For
    `concepts/workloads/pods/_index.md` this yields `["Concepts", "Workloads"]`;
    the page's own title is added by the chunker.
    """
    index_titles = {
        str(PurePosixPath(r["path"]).parent): r["title"]
        for r in records
        if PurePosixPath(r["path"]).name == "_index.md"
    }
    breadcrumbs: dict[str, list[str]] = {}
    for record in records:
        path = PurePosixPath(record["path"])
        page_dir = path.parent.parent if path.name == "_index.md" else path.parent
        ancestors = [] if str(page_dir) == "." else [*reversed(page_dir.parents), page_dir]
        breadcrumbs[record["path"]] = [
            index_titles.get(str(d), _prettify_dir_name(d.name)) for d in ancestors if str(d) != "."
        ]
    return breadcrumbs


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    """Longest shared leading run of two heading tuples."""
    prefix: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        prefix.append(x)
    return tuple(prefix)


def merge_sections(first: Section, second: Section) -> Section:
    """Merge two adjacent sections under their shared heading path.

    Headings that aren't shared move into the body as plain lines, so no
    context is lost.
    """
    common = _common_prefix(first.headings, second.headings)
    parts: list[str] = []
    for section in (first, second):
        own = section.headings[len(common) :]
        if own:
            parts.append(HEADING_PATH_SEPARATOR.join(own))
        parts.append(section.body)
    return Section(common, BLOCK_SEPARATOR.join(parts))


def merge_tiny_sections(
    sections: list[Section], counter: TokenCounter, min_tokens: int, max_tokens: int
) -> list[Section]:
    """Merge sections under `min_tokens` into their neighbour when the result fits.

    A tiny section merges forward into the next one (or, if it's last, back
    into the previous one) only if the merged body stays under `max_tokens`;
    otherwise it stays on its own rather than blurring a big section's path.
    """
    merged: list[Section] = []
    carry: Section | None = None
    for section in sections:
        if carry is not None:
            candidate = merge_sections(carry, section)
            if counter.count(candidate.body) <= max_tokens:
                section = candidate
            else:
                merged.append(carry)
            carry = None
        if counter.count(section.body) < min_tokens:
            carry = section
        else:
            merged.append(section)
    if carry is not None:
        if merged and counter.count(merge_sections(merged[-1], carry).body) <= max_tokens:
            merged[-1] = merge_sections(merged[-1], carry)
        else:
            merged.append(carry)
    return merged


def chunk_doc(
    record: dict, breadcrumb: list[str], sizes: ChunkSizes, counter: TokenCounter
) -> list[Chunk]:
    """Split one doc record into heading-scoped, size-limited chunks."""
    page_path = [*breadcrumb, record["title"]]
    sections = merge_tiny_sections(
        split_sections(record["text"]), counter, sizes.min_tokens, sizes.max_tokens
    )
    chunks: list[Chunk] = []
    for section in sections:
        heading_path = [*page_path, *section.headings]
        prefix = HEADING_PATH_SEPARATOR.join(heading_path) + BLOCK_SEPARATOR
        # The prefix is embedded too, so it comes out of the size budget.
        budget = max(sizes.max_tokens - counter.count(prefix), sizes.max_tokens // 4)
        for body in pack_units(split_blocks(section.body), counter, budget, sizes.overlap_tokens):
            text = prefix + body
            chunks.append(
                Chunk(
                    chunk_id=f"doc:{record['path']}#{len(chunks)}",
                    source_type="doc",
                    title=record["title"],
                    url=record["url"],
                    date=record.get("date"),
                    text=text,
                    token_count=counter.count(text),
                    heading_path=heading_path,
                    extra={"path": record["path"]},
                )
            )
    return chunks
