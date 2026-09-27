import pytest

from repo_recall.retrieval.models import SearchMode, SearchResult


def make_result(chunk_id: str, rank: int, mode: SearchMode = SearchMode.VECTOR) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source_type="issue" if chunk_id.startswith("issue") else "doc",
        title=f"Title {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        date=None,
        heading_path=["Concepts", "Pods", chunk_id],
        labels=[],
        extra={"number": 7},
        text="text",
        score=1.0 / rank,
        rank=rank,
        mode=mode,
    )


def ranked(ids: list[str], mode: SearchMode = SearchMode.VECTOR) -> list[SearchResult]:
    return [make_result(chunk_id, i, mode) for i, chunk_id in enumerate(ids, start=1)]


@pytest.fixture
def ranked_list():
    return ranked
