from repo_recall.retrieval.models import SearchMode, SearchResult


def chunk(chunk_id: str, url: str, source_type: str = "doc", rank: int = 1, **extra) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source_type=source_type,
        title=f"Title {url}",
        url=url,
        date=None,
        heading_path=["Concepts", "Pods"] if source_type == "doc" else [],
        labels=[],
        extra=extra,
        text=f"text of {chunk_id}",
        score=1.0,
        rank=rank,
        mode=SearchMode.HYBRID,
    )
