import pytest

from repo_recall.processing.chunk_model import Chunk


def make_chunk(chunk_id: str, text: str = "some text", source_type: str = "doc") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_type=source_type,
        title=f"Title {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        date="2026-09-24T19:00:16Z",
        text=text,
        token_count=len(text.split()),
        heading_path=["Concepts", "Pods"],
        labels=["kind/bug"] if source_type == "issue" else [],
        extra={"path": "concepts/pods.md"},
    )


@pytest.fixture
def chunk_factory():
    return make_chunk
