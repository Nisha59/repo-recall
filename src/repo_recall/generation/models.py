"""The records answer generation returns."""

from dataclasses import dataclass

from repo_recall.retrieval.models import SearchResult


@dataclass(frozen=True)
class Source:
    """One numbered source shown to the model: a docs page or an issue.

    Several retrieved chunks from the same URL share one source number,
    so a citation points at a page or issue, not at a chunk.
    """

    number: int
    title: str
    url: str
    source_type: str
    chunks: tuple[SearchResult, ...]


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one LLM call and the estimated cost in USD.

    Counts are None when the server doesn't report them; cost is then None too.
    """

    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None


@dataclass(frozen=True)
class Timing:
    """Wall-clock seconds spent in each stage."""

    retrieval_seconds: float
    generation_seconds: float

    @property
    def total_seconds(self) -> float:
        """Retrieval plus generation."""
        return self.retrieval_seconds + self.generation_seconds


@dataclass(frozen=True)
class Answer:
    """A generated answer with its citations, timing and cost.

    `answered` is False when the model said the sources don't contain the
    answer, or when it cited nothing valid (an uncited answer isn't grounded).
    `retrieved` keeps every source the model was shown, for evaluation.
    """

    question: str
    text: str
    answered: bool
    cited_sources: list[Source]
    retrieved: list[Source]
    model: str
    usage: TokenUsage
    timing: Timing
