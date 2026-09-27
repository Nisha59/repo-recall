"""The chunk record shared by the doc and issue chunkers."""

from dataclasses import asdict, dataclass, field
from typing import Literal

SourceType = Literal["doc", "issue"]


@dataclass(frozen=True)
class ChunkSizes:
    """Chunk size limits, in embedding-model tokens."""

    max_tokens: int
    overlap_tokens: int
    min_tokens: int


@dataclass
class Chunk:
    """One retrievable unit of text plus the metadata needed to cite it.

    `text` is exactly what gets embedded (context prefix included), and
    `token_count` is measured on that text.
    """

    chunk_id: str
    source_type: SourceType
    title: str
    url: str
    date: str | None
    text: str
    token_count: int
    heading_path: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the chunk as a JSON-serializable dict."""
        return asdict(self)
