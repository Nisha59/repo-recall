"""Build the evaluation set: real questions from closed issues, with their accepted answers.

Run as `python -m repo_recall.evaluation.eval_set` after collecting candidates
with `python -m repo_recall.collection.issue_search`.

Selection rules, in order (the first one that fails is the rejection reason):
1. not closed as a duplicate (the answer lives in another issue);
2. title doesn't look like CI or tracking work (flaky/failing tests, tracking issues);
3. not filed by a maintainer (OWNER/MEMBER/COLLABORATOR): those are work items, not user questions;
4. the question (title + essential body) has at least MIN_QUESTION_WORDS words;
5. at least one answer comment, meaning a comment that:
   - is by someone other than the asker (bots are dropped at collection),
   - has at least MIN_ANSWER_WORDS words once slash commands and quotes are removed,
   - isn't a non-answer (unless it links docs or a fix): a redirect to another channel,
     vendor or repo, a request for more information, or a "same issue here",
   - carries a trust signal: a maintainer reply (OWNER/MEMBER/COLLABORATOR/CONTRIBUTOR),
     a docs link, or a linked fix (PR or commit).
The best answer comment by score becomes the reference answer. If more than
`eval_max_examples` issues qualify, the highest-scoring ones are kept.
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from repo_recall.collection.issue_search import candidates_path
from repo_recall.config import Settings
from repo_recall.processing.chunk_all import read_jsonl
from repo_recall.processing.issue_chunker import clean_issue_body
from repo_recall.processing.text_units import BLOCK_SEPARATOR, is_fence_line, split_blocks

logger = logging.getLogger(__name__)

MIN_QUESTION_WORDS = 15
MIN_ANSWER_WORDS = 25
MAX_QUESTION_WORDS = 300
MAX_CODE_LINES = 30
MAINTAINER_ROLES = {"OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"}

_CI_OR_TRACKING_TITLE_RE = re.compile(
    r"\b(flak(e|y)|failing test|failing job|ci signal|tracking issue|umbrella issue)\b"
    r"|^\[?kep\b",
    re.IGNORECASE,
)
# A slash command on its own line: "/close", "/kind support", "/assign @x".
# Requires a space or end after the word, so paths like "/etc/kubernetes" survive.
_COMMAND_LINE_RE = re.compile(r"^/[a-z][a-z-]*(\s.*)?$")
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+")
_URL_TRAILING_PUNCT = ".,;:!?*_"
_FIX_URL_RE = re.compile(r"github\.com/[\w.-]+/[\w.-]+/(pull/\d+|commit/[0-9a-f]{7,})")
# Replies that aren't answers: pointing elsewhere, asking for more info, or "me too".
_NON_ANSWER_RE = re.compile(
    # redirects to other channels, vendors or repos
    r"stack\s*overflow|slack\.k8s\.io|kubernetes slack|discuss\.kubernetes\.io|"
    r"(not the right|wrong|better) place|support (channels?|forums?|questions|ticket)|"
    r"user support|please (ask|post|reach out) (on|in|to)|"
    r"(issue tracker|github issues|this repo)\b[^.]{0,20}\b(is|are) for\b|"
    r"(open|file|report) (an|the|this) issue (in|with|to|at)|should be filed with|"
    r"contact(ed)? (your|the) [\w ]{0,20}(vendor|provider|support)|"
    r"read-only (export|mirror)|out of scope for this|"
    # requests for more information
    r"please (provide|share|paste|fill|attach|add|answer|elaborate)|"
    r"(can|could) you (please )?(briefly )?(share|provide|tell|paste|post|attach|clarify|elaborate)|"
    r"can i (know|see|get) (your|the)|"
    r"more (details|information|info)\b|issue template|"
    # me-too reports and triage chatter
    r"(exact )?same (issue|problem|error)|we (also )?(see|hit|got) (this|the same)|"
    r"triage meeting|\bping @|^/?cc @",
    re.IGNORECASE | re.MULTILINE,
)
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")
# Issue authors in these roles usually file work items, not user questions.
_MAINTAINER_ASKER_ROLES = {"OWNER", "MEMBER", "COLLABORATOR"}
_CONFIRMATION_RE = re.compile(
    r"\b(thanks|thank you|that worked|it works|works now|solved|resolved|fixed it)\b",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(?:#{1,6}\s+(?P<md>.+?)|\*\*(?P<bold>[^*]+)\*\*:?.*)\s*$")
# Environment boilerplate from the Kubernetes bug template; rarely part of the question.
_BOILERPLATE_SECTION_RE = re.compile(
    r"kubernetes version|cloud provider|os version|operating system|install tools|"
    r"container runtime|related plugins|^environment",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerCandidate:
    """One comment that qualifies as an answer, with why and how strongly."""

    comment: dict
    text: str
    urls: list[str]
    signals: list[str]
    score: float


# --- Text helpers -----------------------------------------------------------


def strip_comment_noise(body: str) -> str:
    """Remove slash-command lines and quoted (`>`) lines, then tidy blank lines."""
    kept = [
        line
        for line in body.splitlines()
        if not _COMMAND_LINE_RE.match(line.strip()) and not line.lstrip().startswith(">")
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def word_count(text: str) -> int:
    """Whitespace-separated words in `text`."""
    return len(text.split())


def extract_urls(text: str) -> list[str]:
    """Every http(s) URL in `text`, trailing punctuation removed, de-duplicated in order."""
    urls: dict[str, None] = {}
    for match in _URL_RE.finditer(text):
        urls.setdefault(match.group().rstrip(_URL_TRAILING_PUNCT), None)
    return list(urls)


def _host_and_path(url: str) -> tuple[str, str]:
    """Lowercased host without `www.`, and the URL path."""
    parts = urlsplit(url)
    return parts.netloc.lower().removeprefix("www."), parts.path


def is_docs_url(url: str, settings: Settings) -> bool:
    """True for the project's docs site (any language) or its docs source repo."""
    host, path = _host_and_path(url)
    docs_host, _ = _host_and_path(settings.docs_public_base_url)
    repo_host, repo_path = _host_and_path(settings.docs_repo_url)
    if host == docs_host and "/docs" in path:
        return True
    return host == repo_host and path.startswith(repo_path.rstrip("/") + "/")


def normalize_doc_url(url: str, settings: Settings) -> str | None:
    """The index's form of an English docs URL (https, no query/anchor, trailing /), else None."""
    host, path = _host_and_path(url)
    docs_host, docs_path = _host_and_path(settings.docs_public_base_url)
    docs_path = docs_path.rstrip("/")
    if host != docs_host or not (path == docs_path or path.startswith(docs_path + "/")):
        return None
    return f"https://{docs_host}{path.rstrip('/')}/"


def is_fix_url(url: str) -> bool:
    """True for a GitHub pull request or commit link."""
    return bool(_FIX_URL_RE.search(url))


def is_mostly_questions(text: str) -> bool:
    """True if at least half the sentences are questions (a clarifying reply, not an answer)."""
    endings = _SENTENCE_END_RE.findall(text)
    return bool(endings) and sum("?" in e for e in endings) * 2 >= len(endings)


def is_non_answer(text: str) -> bool:
    """True if the comment redirects elsewhere, asks for more info, or says "me too"."""
    return bool(_NON_ANSWER_RE.search(text)) or is_mostly_questions(text)


# --- Question ---------------------------------------------------------------


def section_name(block: str) -> str | None:
    """The heading text if `block` starts with a Markdown or bold heading."""
    match = _HEADING_RE.match(block.split("\n", 1)[0].strip())
    if not match:
        return None
    return (match.group("md") or match.group("bold")).strip()


def cap_code_block(block: str, max_lines: int = MAX_CODE_LINES) -> str:
    """Shorten a fenced code block to `max_lines` lines of content."""
    lines = block.split("\n")
    if not is_fence_line(lines[0]) or len(lines) <= max_lines + 2:
        return block
    hidden = len(lines) - 2 - max_lines
    return "\n".join([*lines[: max_lines + 1], f"... ({hidden} more lines)", lines[-1]])


def essential_blocks(body: str) -> list[str]:
    """Body blocks minus environment boilerplate sections, code blocks capped."""
    blocks: list[str] = []
    skipping = False
    for block in split_blocks(clean_issue_body(body)):
        name = section_name(block)
        if name is not None:
            skipping = bool(_BOILERPLATE_SECTION_RE.search(name))
        if not skipping:
            blocks.append(cap_code_block(block))
    return blocks


def truncate_blocks(blocks: list[str], max_words: int = MAX_QUESTION_WORDS) -> str:
    """Join whole blocks while they fit in `max_words`; cut the first block if it alone is too long."""
    kept: list[str] = []
    total = 0
    for block in blocks:
        words = word_count(block)
        if total + words > max_words:
            break
        kept.append(block)
        total += words
    if not kept and blocks:
        return " ".join(blocks[0].split()[:max_words]) + " ..."
    return BLOCK_SEPARATOR.join(kept)


def build_question(record: dict) -> str:
    """The issue title plus the essential part of its body."""
    body = truncate_blocks(essential_blocks(record.get("body") or ""))
    return f"{record['title']}\n\n{body}" if body else record["title"]


# --- Answers ----------------------------------------------------------------


def asker_confirmed_after(comments: list[dict], index: int, asker: str | None) -> bool:
    """True if the asker's next comment after `index` thanks or confirms a fix.

    Only the asker's *next* comment counts, so a later "thanks" isn't
    credited to an earlier, unrelated reply.
    """
    if not asker:
        return False
    reply = next((c for c in comments[index + 1 :] if c["author"] == asker), None)
    return reply is not None and bool(_CONFIRMATION_RE.search(reply["body"]))


def answer_signals(comment: dict, urls: list[str], settings: Settings) -> list[str]:
    """The trust signals a comment carries."""
    signals = []
    if comment.get("author_association") in MAINTAINER_ROLES:
        signals.append("maintainer_reply")
    if any(is_docs_url(url, settings) for url in urls):
        signals.append("docs_link")
    if any(is_fix_url(url) for url in urls):
        signals.append("fix_link")
    return signals


def score_answer(signals: list[str], text: str) -> float:
    """How strong an answer looks: its signals plus a small length bonus.

    Asker confirmation weighs little: in practice it is mostly a polite
    "thanks", not proof the answer worked.
    """
    weights = {"maintainer_reply": 2, "docs_link": 2, "fix_link": 1, "asker_confirmed": 1}
    return sum(weights[s] for s in signals) + min(word_count(text) / 100, 1.0)


def answer_candidates(record: dict, settings: Settings) -> list[AnswerCandidate]:
    """Every comment that qualifies as an answer (rule 5 in the module docstring)."""
    asker = record.get("author")
    comments = record.get("comments", [])
    candidates = []
    for index, comment in enumerate(comments):
        if asker and comment["author"] == asker:
            continue
        text = strip_comment_noise(comment["body"])
        if word_count(text) < MIN_ANSWER_WORDS:
            continue
        urls = extract_urls(text)
        signals = answer_signals(comment, urls, settings)
        has_link = "docs_link" in signals or "fix_link" in signals
        if not signals or (is_non_answer(text) and not has_link):
            continue
        if asker_confirmed_after(comments, index, asker):
            signals.append("asker_confirmed")
        candidates.append(AnswerCandidate(comment, text, urls, signals, score_answer(signals, text)))
    return candidates


# --- Selection --------------------------------------------------------------


def rejection_reason(record: dict, question: str) -> str | None:
    """Why an issue fails rules 1-4, or None if it passes them."""
    if record.get("state_reason") == "duplicate":
        return "duplicate"
    if _CI_OR_TRACKING_TITLE_RE.search(record["title"]):
        return "ci_or_tracking_title"
    if record.get("author_association") in _MAINTAINER_ASKER_ROLES:
        return "asked_by_maintainer"
    if word_count(question) < MIN_QUESTION_WORDS:
        return "short_question"
    return None


def build_example(
    record: dict, question: str, answer: AnswerCandidate, settings: Settings
) -> dict:
    """The eval-set record for one issue and its chosen answer."""
    doc_urls = [normalize_doc_url(url, settings) for url in answer.urls]
    return {
        "id": f"{settings.github_repo}#{record['number']}",
        "issue_number": record["number"],
        "issue_url": record["url"],
        "title": record["title"],
        "question": question,
        "reference_answer": answer.text,
        "answer_author": answer.comment["author"],
        "answer_author_association": answer.comment.get("author_association"),
        "source_urls": answer.urls,
        "doc_urls": list(dict.fromkeys(url for url in doc_urls if url)),
        "labels": record.get("labels", []),
        "closed_at": record.get("closed_at"),
        "signals": answer.signals,
        "answer_score": round(answer.score, 3),
    }


def evaluate_issue(record: dict, settings: Settings) -> tuple[dict | None, str | None]:
    """`(example, None)` if the issue qualifies, else `(None, rejection_reason)`."""
    question = build_question(record)
    reason = rejection_reason(record, question)
    if reason:
        return None, reason
    candidates = answer_candidates(record, settings)
    if not candidates:
        return None, "no_qualifying_answer"
    best = max(candidates, key=lambda c: c.score)
    return build_example(record, question, best, settings), None


def build_eval_set(records: list[dict], settings: Settings) -> tuple[list[dict], Counter]:
    """Apply the rules to every record; keep the top `eval_max_examples` by answer score.

    Returns the examples and a count of rejections per reason.
    """
    rejections: Counter = Counter()
    qualified = []
    for record in records:
        example, reason = evaluate_issue(record, settings)
        if example is None:
            rejections[reason] += 1
        else:
            qualified.append(example)
    qualified.sort(key=lambda e: (-e["answer_score"], e["issue_number"]))
    kept = qualified[: settings.eval_max_examples]
    if len(qualified) > len(kept):
        rejections["over_max_examples"] = len(qualified) - len(kept)
    return kept, rejections


# --- Files and output -------------------------------------------------------


def write_eval_set(examples: list[dict], path: Path) -> None:
    """Write examples as JSONL, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def format_summary(examples: list[dict], rejections: Counter, total: int) -> str:
    """Counts of kept examples, rejections per reason, and signals among kept."""
    lines = [f"Candidates: {total}   kept: {len(examples)}", "Rejected:"]
    lines += [f"  {reason:<24} {count}" for reason, count in rejections.most_common()]
    signals = Counter(s for e in examples for s in e["signals"])
    lines.append("Signals among kept:")
    lines += [f"  {signal:<24} {count}" for signal, count in signals.most_common()]
    return "\n".join(lines)


def _shorten(text: str, max_chars: int) -> str:
    """Cut `text` to `max_chars` characters, marking the cut."""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " [...]"


def format_example(example: dict, max_chars: int = 800) -> str:
    """A readable view of one example for manual quality checks."""
    urls = "\n".join(f"  - {url}" for url in example["source_urls"]) or "  (none)"
    return "\n".join(
        [
            "=" * 80,
            f"{example['id']}  {example['issue_url']}",
            f"signals: {', '.join(example['signals'])}   score: {example['answer_score']}",
            "",
            "QUESTION:",
            _shorten(example["question"], max_chars),
            "",
            f"REFERENCE ANSWER ({example['answer_author']}, {example['answer_author_association']}):",
            _shorten(example["reference_answer"], max_chars),
            "",
            "SOURCE URLS:",
            urls,
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Build the evaluation set from candidate issues.")
    parser.add_argument("--show", type=int, default=5, help="random examples to print")
    parser.add_argument("--seed", type=int, default=None, help="seed for picking the examples shown")
    parser.add_argument(
        "--show-only", action="store_true", help="print examples from the existing file; don't rebuild"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: build and write the eval set, then print a summary and samples."""
    args = parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # issue text has emoji; Windows consoles default to cp1252

    if args.show_only:
        examples = read_jsonl(settings.eval_set_path)
    else:
        records = read_jsonl(candidates_path(settings))
        examples, rejections = build_eval_set(records, settings)
        write_eval_set(examples, settings.eval_set_path)
        logger.info("Wrote %d examples to %s", len(examples), settings.eval_set_path)
        if len(examples) < settings.eval_min_examples:
            logger.warning(
                "Only %d examples (target %d+); collect more candidates with "
                "`python -m repo_recall.collection.issue_search --limit N`",
                len(examples),
                settings.eval_min_examples,
            )
        print(format_summary(examples, rejections, len(records)))

    sample = random.Random(args.seed).sample(examples, min(args.show, len(examples)))
    for example in sample:
        print(format_example(example))


if __name__ == "__main__":
    main()
