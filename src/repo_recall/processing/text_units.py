"""Split Markdown into paragraph-level blocks and pack them into sized chunks.

Shared by the doc and issue chunkers. Blocks are the unit of packing and of
overlap, so a chunk boundary never falls mid-paragraph or mid-code-block
unless a single block is itself bigger than the chunk budget.
"""

import re

from repo_recall.processing.tokens import TokenCounter

_FENCE_RE = re.compile(r"^\s*(```|~~~)")

BLOCK_SEPARATOR = "\n\n"


def is_fence_line(line: str) -> bool:
    """True if `line` opens or closes a fenced code block."""
    return bool(_FENCE_RE.match(line))


def split_blocks(markdown: str) -> list[str]:
    """Split Markdown on blank lines into blocks, keeping fenced code blocks whole."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in markdown.split("\n"):
        if is_fence_line(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def split_oversized(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Split one too-big block into pieces of at most `max_tokens`.

    Tries line boundaries first, then word boundaries, and only as a last
    resort cuts a single huge "word" (e.g. a base64 blob) by characters.
    """
    if counter.count(text) <= max_tokens:
        return [text]
    for separator in ("\n", " "):
        parts = text.split(separator)
        if len(parts) > 1:
            return _greedy_join(parts, separator, counter, max_tokens)
    middle = len(text) // 2
    return split_oversized(text[:middle], counter, max_tokens) + split_oversized(
        text[middle:], counter, max_tokens
    )


def _greedy_join(
    parts: list[str], separator: str, counter: TokenCounter, max_tokens: int
) -> list[str]:
    """Join consecutive parts while they fit; recursively split parts that never fit."""
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if counter.count(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if counter.count(part) <= max_tokens:
            current = part
        else:
            pieces.extend(split_oversized(part, counter, max_tokens))
            current = ""
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece.strip()]


def overlap_tail(units: list[str], counter: TokenCounter, overlap_tokens: int) -> list[str]:
    """Return the longest run of trailing whole units that fits in `overlap_tokens`."""
    tail: list[str] = []
    for unit in reversed(units):
        if counter.count(BLOCK_SEPARATOR.join([unit, *tail])) > overlap_tokens:
            break
        tail.insert(0, unit)
    return tail


def pack_units(
    units: list[str], counter: TokenCounter, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Greedily pack units into chunk bodies of at most `max_tokens` each.

    Each chunk after the first starts with the trailing whole units of the
    previous one (up to `overlap_tokens`), so context carries across a split.
    """
    expanded = [piece for unit in units for piece in split_oversized(unit, counter, max_tokens)]
    chunks: list[str] = []
    current: list[str] = []
    carried = 0  # how many units at the front of `current` are overlap
    for unit in expanded:
        if counter.count(BLOCK_SEPARATOR.join([*current, unit])) <= max_tokens:
            current.append(unit)
            continue
        if len(current) > carried:
            chunks.append(BLOCK_SEPARATOR.join(current))
            current = overlap_tail(current, counter, overlap_tokens)
        while current and counter.count(BLOCK_SEPARATOR.join([*current, unit])) > max_tokens:
            current.pop(0)
        carried = len(current)
        current.append(unit)
    if len(current) > carried:
        chunks.append(BLOCK_SEPARATOR.join(current))
    return chunks
