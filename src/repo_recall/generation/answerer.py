"""The answer entry point: question in, cited answer with timing and cost out."""

import logging
import time
from collections.abc import Callable
from typing import Protocol

from repo_recall.config import Settings
from repo_recall.generation.citations import cited_sources, is_no_answer
from repo_recall.generation.cost import estimate_usage
from repo_recall.generation.llm import LLMClient
from repo_recall.generation.models import Answer, Source, Timing, TokenUsage
from repo_recall.generation.prompt import NO_ANSWER, build_prompt, group_sources
from repo_recall.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class ChunkRetriever(Protocol):
    """What the answerer needs from retrieval (the Retriever class fits)."""

    def search(
        self, question: str, top_k: int | None = None, source_type: str | None = None
    ) -> list[SearchResult]:
        """The top chunks for `question`."""
        ...


class Answerer:
    """Retrieves sources for a question and has the LLM answer from them with citations."""

    def __init__(
        self,
        retriever: ChunkRetriever,
        llm: LLMClient,
        settings: Settings,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings
        self._clock = clock

    def answer(
        self, question: str, top_k: int | None = None, source_type: str | None = None
    ) -> Answer:
        """Answer `question` from the top retrieved chunks.

        If retrieval finds nothing, the LLM isn't called and the answer is
        the standard "sources don't contain the answer" reply.
        """
        start = self._clock()
        results = self._retriever.search(question, top_k=top_k, source_type=source_type)
        retrieval_seconds = self._clock() - start
        return self.answer_from_results(question, results, retrieval_seconds)

    def answer_from_results(
        self, question: str, results: list[SearchResult], retrieval_seconds: float
    ) -> Answer:
        """Answer `question` from chunks that were already retrieved.

        For callers that retrieve separately (the evaluation runs one search
        per question and reuses it); `retrieval_seconds` is reported as given.
        """
        sources = group_sources(results)
        if not sources:
            logger.info("No sources retrieved; skipping the LLM call")
            no_usage = TokenUsage(0, 0, 0.0)
            return self._build_answer(question, NO_ANSWER, sources, no_usage, retrieval_seconds, 0.0)

        prompt = build_prompt(question, sources, project=self._settings.github_repo)
        start = self._clock()
        completion = self._llm.complete(prompt.system, prompt.user)
        generation_seconds = self._clock() - start

        usage = estimate_usage(
            completion.prompt_tokens,
            completion.completion_tokens,
            self._settings.llm_input_cost_per_mtok,
            self._settings.llm_output_cost_per_mtok,
        )
        answer = self._build_answer(
            question, completion.text.strip(), sources, usage, retrieval_seconds, generation_seconds
        )
        log_usage(answer)
        return answer

    def _build_answer(
        self,
        question: str,
        text: str,
        sources: list[Source],
        usage: TokenUsage,
        retrieval_seconds: float,
        generation_seconds: float,
    ) -> Answer:
        """Assemble the result, working out which sources the text cites."""
        cited = [] if is_no_answer(text) else cited_sources(text, sources)
        return Answer(
            question=question,
            text=text,
            answered=bool(cited),
            cited_sources=cited,
            retrieved=sources,
            model=self._llm.model,
            usage=usage,
            timing=Timing(retrieval_seconds, generation_seconds),
        )


def log_usage(answer: Answer) -> None:
    """One log line per query with tokens, estimated cost and timings."""
    usage = answer.usage
    if usage.estimated_cost_usd is None:
        tokens = "token usage not reported by the server"
    else:
        tokens = (
            f"{usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens, "
            f"est. ${usage.estimated_cost_usd:.6f}"
        )
    logger.info(
        "%s: %s; retrieval %.2fs, generation %.2fs; cited %d of %d sources",
        answer.model,
        tokens,
        answer.timing.retrieval_seconds,
        answer.timing.generation_seconds,
        len(answer.cited_sources),
        len(answer.retrieved),
    )
