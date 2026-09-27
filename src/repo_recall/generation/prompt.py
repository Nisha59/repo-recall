"""Turning retrieved chunks into numbered sources and a grounded-answer prompt."""

from dataclasses import dataclass

from repo_recall.generation.models import Source
from repo_recall.retrieval.models import SearchResult

NO_ANSWER = "The provided sources don't contain the answer to this question."

SYSTEM_TEMPLATE = """\
You answer developer questions about {project} using only the numbered sources \
in the user's message. The sources are excerpts from the project's documentation \
and its closed GitHub issues.

Rules:
- Use only information stated in the sources. Do not add facts from outside knowledge.
- Cite every claim with the number of the source it comes from, in square brackets, \
e.g. "Pods restart according to restartPolicy [2]." Use several numbers for a claim \
supported by several sources, e.g. [1][3]. Only cite numbers that appear in the sources.
- If the sources do not contain the answer, reply with exactly this sentence and \
nothing else: "{no_answer}"
- If the sources answer only part of the question, answer that part with citations \
and say plainly which part the sources don't cover.
- A closed issue describes one user's situation; say so when an answer comes from an issue.
- Be concise and practical. Keep commands, flags and config snippets exactly as written \
in the sources."""


@dataclass(frozen=True)
class Prompt:
    """The two messages sent to the model."""

    system: str
    user: str


def group_sources(results: list[SearchResult]) -> list[Source]:
    """Number sources 1..n in rank order, merging chunks that share a URL."""
    by_url: dict[str, list[SearchResult]] = {}
    for result in results:
        by_url.setdefault(result.url, []).append(result)
    return [
        Source(
            number=number,
            title=_source_title(chunks[0]),
            url=url,
            source_type=chunks[0].source_type,
            chunks=tuple(chunks),
        )
        for number, (url, chunks) in enumerate(by_url.items(), start=1)
    ]


def build_prompt(question: str, sources: list[Source], project: str) -> Prompt:
    """The system rules plus a user message holding the sources and the question."""
    system = SYSTEM_TEMPLATE.format(project=project, no_answer=NO_ANSWER)
    blocks = "\n\n".join(format_source(source) for source in sources)
    user = f"Sources:\n\n{blocks}\n\nQuestion: {question}"
    return Prompt(system=system, user=user)


def format_source(source: Source) -> str:
    """One source as the model sees it: number, kind, title, URL, then its excerpts."""
    kind = "Closed GitHub issue" if source.source_type == "issue" else "Documentation"
    excerpts = "\n\n".join(_excerpt(chunk) for chunk in source.chunks)
    return f"[{source.number}] {kind}: {source.title}\nURL: {source.url}\n{excerpts}"


def _source_title(chunk: SearchResult) -> str:
    """Issue number and title for issues, the page title for docs."""
    if chunk.source_type == "issue" and "number" in chunk.extra:
        return f"#{chunk.extra['number']} {chunk.title}"
    return chunk.title


def _excerpt(chunk: SearchResult) -> str:
    """A chunk's text, headed by its section path for docs."""
    if chunk.source_type == "doc" and chunk.heading_path:
        return f"Section: {' > '.join(chunk.heading_path)}\n{chunk.text}"
    return chunk.text
