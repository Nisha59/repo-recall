"""Chunk closed GitHub issues: the problem statement plus the comments that matter.

An issue is useful for RAG because it pairs a problem with how it was
resolved. So chunks keep the title and description, and only the comments
most likely to carry diagnosis or resolution; pings and "+1"s are dropped.
Every chunk repeats the issue header so a split-off piece still says which
problem it belongs to.
"""

import re

from repo_recall.processing.chunk_model import Chunk, ChunkSizes
from repo_recall.processing.text_units import BLOCK_SEPARATOR, pack_units, split_blocks
from repo_recall.processing.tokens import TokenCounter

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# An issue-form section left empty by the reporter, e.g. "### Anything else?\n\n_No response_".
_EMPTY_TEMPLATE_SECTION_RE = re.compile(r"^#{1,6}[^\n]*\n\s*_No response_\s*$", re.MULTILINE)
_DETAILS_TAG_RE = re.compile(r"</?(details|summary)>", re.IGNORECASE)
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")

_MENTION_RE = re.compile(r"@[\w-]+(/[\w-]+)?")
_LOW_VALUE_RE = re.compile(
    r"^(\+1|me too|same here|same issue|any updates?|bump|ping|thanks|thank you|ptal|lgtm)"
    r"[\s!.?]*$",
    re.IGNORECASE,
)
_CODE_OR_TRACE_RE = re.compile(r"```|Traceback|panic:|goroutine \d+", re.IGNORECASE)
_PR_OR_COMMIT_RE = re.compile(r"/pull/\d+|/commit/[0-9a-f]{7,}|\bPR\s*#?\d+", re.IGNORECASE)
_RESOLUTION_RE = re.compile(
    r"\b(fixed|fixes|resolved|root cause|workaround|duplicate of|caused by|"
    r"the (fix|solution) is|works as intended|working as intended)\b",
    re.IGNORECASE,
)

MIN_SUBSTANTIVE_WORDS = 4


def clean_issue_body(text: str) -> str:
    """Strip issue-template noise: HTML comments, empty form sections, `<details>` tags."""
    text = _HTML_COMMENT_RE.sub("", text)
    text = _EMPTY_TEMPLATE_SECTION_RE.sub("", text)
    text = _DETAILS_TAG_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _EXCESS_BLANK_LINES_RE.sub("\n\n", text).strip()


def is_low_value_comment(body: str) -> bool:
    """True for comments with no content beyond pings, "+1", "any update?" and the like."""
    without_mentions = _MENTION_RE.sub("", body).strip()
    if len(without_mentions.split()) < MIN_SUBSTANTIVE_WORDS:
        return True
    return bool(_LOW_VALUE_RE.match(without_mentions))


def score_comment(body: str) -> float:
    """Heuristic relevance of a comment: code/traces, PR links, resolution wording, length."""
    score = 0.0
    if _CODE_OR_TRACE_RE.search(body):
        score += 3
    if _PR_OR_COMMIT_RE.search(body):
        score += 3
    if _RESOLUTION_RE.search(body):
        score += 2
    score += min(len(body.split()) / 50, 2)
    return score


def select_comments(comments: list[dict], max_comments: int) -> list[dict]:
    """Pick up to `max_comments` substantive comments, returned in original order.

    The last substantive comment is always kept, since it usually states
    the resolution; the rest are the highest-scoring ones.
    """
    if max_comments <= 0:
        return []
    useful = [(i, c) for i, c in enumerate(comments) if not is_low_value_comment(c["body"])]
    if not useful:
        return []
    *earlier, last = useful
    ranked = sorted(earlier, key=lambda item: score_comment(item[1]["body"]), reverse=True)
    chosen = [*ranked[: max_comments - 1], last]
    return [comment for _, comment in sorted(chosen, key=lambda item: item[0])]


def issue_header(record: dict) -> str:
    """The title/labels line repeated at the top of every chunk of an issue."""
    header = f"Issue #{record['number']}: {record['title']}"
    if record.get("labels"):
        header += f"\nLabels: {', '.join(record['labels'])}"
    return header


def issue_units(record: dict, max_comments: int) -> list[str]:
    """The packable pieces of an issue: description blocks, then one unit per kept comment."""
    units = split_blocks(clean_issue_body(record.get("body") or ""))
    for comment in select_comments(record.get("comments", []), max_comments):
        units.append(f"Comment by {comment['author']}:\n{comment['body']}")
    return units


def chunk_issue(
    record: dict, repo: str, sizes: ChunkSizes, max_comments: int, counter: TokenCounter
) -> list[Chunk]:
    """Split one issue record into size-limited chunks, each starting with the issue header."""
    prefix = issue_header(record) + BLOCK_SEPARATOR
    budget = max(sizes.max_tokens - counter.count(prefix), sizes.max_tokens // 4)
    bodies = pack_units(issue_units(record, max_comments), counter, budget, sizes.overlap_tokens)
    texts = [prefix + body for body in bodies] or [prefix.strip()]
    return [
        Chunk(
            chunk_id=f"issue:{repo}#{record['number']}:{i}",
            source_type="issue",
            title=record["title"],
            url=record["url"],
            date=record.get("closed_at"),
            text=text,
            token_count=counter.count(text),
            labels=list(record.get("labels", [])),
            extra={
                "number": record["number"],
                "state_reason": record.get("state_reason"),
                "created_at": record.get("created_at"),
            },
        )
        for i, text in enumerate(texts)
    ]
