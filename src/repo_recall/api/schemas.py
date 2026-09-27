"""Request and response bodies of the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from repo_recall.generation.models import Answer, Source

# Real questions paste logs (the evaluation set's longest is ~5,300
# characters); this leaves room for that but stops runaway requests.
MAX_QUESTION_CHARS = 20_000


class AskRequest(BaseModel):
    """A question, with optional overrides for how many and which sources to use."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=20, description="chunks to retrieve")
    source_type: Literal["doc", "issue"] | None = Field(
        default=None, description="only search docs pages or only issues"
    )


class SourceOut(BaseModel):
    """A cited docs page or issue."""

    number: int
    title: str
    url: str
    source_type: str

    @classmethod
    def from_source(cls, source: Source) -> "SourceOut":
        """The public fields of a generation Source."""
        return cls(number=source.number, title=source.title, url=source.url, source_type=source.source_type)


class TimingOut(BaseModel):
    """Seconds spent searching, generating, and in total."""

    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


class UsageOut(BaseModel):
    """The model that answered and what the call used (None when the server doesn't report it)."""

    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None


class AskResponse(BaseModel):
    """The answer, the sources it cites, and timing and cost of producing it.

    `answered` is False when the sources didn't contain the answer (or the
    model cited nothing valid); `retrieved_count` is how many sources the
    model was shown.
    """

    question: str
    answer: str
    answered: bool
    sources: list[SourceOut]
    retrieved_count: int
    timing: TimingOut
    usage: UsageOut

    @classmethod
    def from_answer(cls, answer: Answer) -> "AskResponse":
        """Build the response body from a generated Answer."""
        return cls(
            question=answer.question,
            answer=answer.text,
            answered=answer.answered,
            sources=[SourceOut.from_source(s) for s in answer.cited_sources],
            retrieved_count=len(answer.retrieved),
            timing=TimingOut(
                retrieval_seconds=answer.timing.retrieval_seconds,
                generation_seconds=answer.timing.generation_seconds,
                total_seconds=answer.timing.total_seconds,
            ),
            usage=UsageOut(
                model=answer.model,
                prompt_tokens=answer.usage.prompt_tokens,
                completion_tokens=answer.usage.completion_tokens,
                estimated_cost_usd=answer.usage.estimated_cost_usd,
            ),
        )


class HealthResponse(BaseModel):
    """Whether the service can answer questions right now."""

    status: Literal["ok", "unavailable"]
    database: Literal["ok", "unreachable"]
