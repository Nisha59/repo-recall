import pytest

from repo_recall.processing.chunk_model import ChunkSizes


class WordCounter:
    """Fake TokenCounter: one token per whitespace-separated word."""

    def count(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def counter() -> WordCounter:
    return WordCounter()


@pytest.fixture
def sizes() -> ChunkSizes:
    return ChunkSizes(max_tokens=40, overlap_tokens=10, min_tokens=5)
