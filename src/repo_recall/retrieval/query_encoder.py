"""Turning a question into a query vector."""

from typing import Protocol

from repo_recall.indexing.embedder import Embedder


class QueryEncoder(Protocol):
    """Anything that can embed a search query."""

    def encode(self, question: str) -> list[float]:
        """Return the query vector for `question`."""
        ...


class InstructedQueryEncoder:
    """Embeds queries with the document embedder plus a query-only instruction.

    bge models are trained with an instruction prefix on the query side
    only; documents were indexed without it. Pass `""` for models that
    don't use one.
    """

    def __init__(self, embedder: Embedder, instruction: str) -> None:
        self._embedder = embedder
        self._instruction = instruction

    def encode(self, question: str) -> list[float]:
        """Embed `question` with the instruction prepended."""
        return self._embedder.embed_documents([self._instruction + question])[0]
