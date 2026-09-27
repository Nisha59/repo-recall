"""Turning chunk text into vectors, behind an interface so tests can use a fake."""

from typing import Protocol


class Embedder(Protocol):
    """Anything that can embed documents into fixed-size vectors."""

    model_name: str

    @property
    def dimension(self) -> int:
        """Length of every vector this embedder returns."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts, one vector per text, in order."""
        ...


class SentenceTransformerEmbedder:
    """Embeds documents with a sentence-transformers model (loaded on first use).

    Vectors are L2-normalized, so cosine similarity equals dot product and
    scores are comparable across queries. Documents are embedded as-is:
    bge's "Represent this sentence for searching..." instruction is only
    for queries, which is retrieval's job.
    """

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _load(self):
        """Return the underlying model, loading it the first time."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Vector size reported by the model (384 for bge-small-en-v1.5)."""
        return self._load().get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into normalized vectors."""
        vectors = self._load().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
