import json
from dataclasses import replace

from repo_recall.indexing.index_chunks import (
    batched,
    content_hash,
    exclude_chunks,
    index_chunks,
    load_excluded_urls,
)
from repo_recall.processing.chunk_model import Chunk


class FakeEmbedder:
    """Records every batch it's asked to embed; returns 3-dim vectors."""

    def __init__(self, model_name: str = "fake-model") -> None:
        self.model_name = model_name
        self.batches: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class InMemoryStore:
    """VectorStore fake: a dict of chunk_id -> (hash, chunk, vector, model)."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple] = {}

    def ensure_schema(self, dimension: int) -> None:
        pass

    def indexed_hashes(self) -> dict[str, str]:
        return {chunk_id: row[0] for chunk_id, row in self.rows.items()}

    def upsert(self, chunks, embeddings, content_hashes, model_name) -> None:
        for chunk, vector, digest in zip(chunks, embeddings, content_hashes, strict=True):
            self.rows[chunk.chunk_id] = (digest, chunk, vector, model_name)

    def delete_by_urls(self, urls: set[str]) -> int:
        doomed = [cid for cid, row in self.rows.items() if row[1].url in urls]
        for chunk_id in doomed:
            del self.rows[chunk_id]
        return len(doomed)


def test_index_chunks_embeds_and_stores_new_chunks(chunk_factory):
    chunks = [chunk_factory(f"c{i}", f"text {i}") for i in range(5)]
    store = InMemoryStore()

    stats = index_chunks(chunks, FakeEmbedder(), store, batch_size=2)

    assert set(store.rows) == {f"c{i}" for i in range(5)}
    assert (stats.total, stats.indexed, stats.skipped) == (5, 5, 0)
    assert store.rows["c0"][3] == "fake-model"


def test_index_chunks_rerun_skips_everything(chunk_factory):
    chunks = [chunk_factory(f"c{i}") for i in range(4)]
    store = InMemoryStore()
    index_chunks(chunks, FakeEmbedder(), store, batch_size=2)

    embedder = FakeEmbedder()
    stats = index_chunks(chunks, embedder, store, batch_size=2)

    assert embedder.batches == []
    assert (stats.indexed, stats.skipped) == (0, 4)


def test_index_chunks_reembeds_chunk_whose_text_changed(chunk_factory):
    store = InMemoryStore()
    index_chunks([chunk_factory("a", "old"), chunk_factory("b", "same")], FakeEmbedder(), store, 10)

    embedder = FakeEmbedder()
    stats = index_chunks(
        [chunk_factory("a", "new text"), chunk_factory("b", "same")], embedder, store, 10
    )

    assert embedder.batches == [["new text"]]
    assert stats.indexed == 1
    assert store.rows["a"][1].text == "new text"


def test_index_chunks_reembeds_everything_when_model_changes(chunk_factory):
    chunks = [chunk_factory("a"), chunk_factory("b")]
    store = InMemoryStore()
    index_chunks(chunks, FakeEmbedder("model-v1"), store, 10)

    stats = index_chunks(chunks, FakeEmbedder("model-v2"), store, 10)

    assert stats.indexed == 2
    assert {row[3] for row in store.rows.values()} == {"model-v2"}


def test_index_chunks_uses_requested_batch_size(chunk_factory):
    chunks = [chunk_factory(f"c{i}") for i in range(7)]
    embedder = FakeEmbedder()

    index_chunks(chunks, embedder, InMemoryStore(), batch_size=3)

    assert [len(b) for b in embedder.batches] == [3, 3, 1]


def test_index_chunks_limit_caps_chunks_embedded(chunk_factory):
    chunks = [chunk_factory(f"c{i}") for i in range(10)]
    store = InMemoryStore()

    stats = index_chunks(chunks, FakeEmbedder(), store, batch_size=4, limit=6)

    assert stats.indexed == 6
    assert len(store.rows) == 6


def test_content_hash_depends_on_text_and_model(chunk_factory):
    chunk: Chunk = chunk_factory("a", "hello")
    assert content_hash(chunk, "m1") == content_hash(chunk_factory("a", "hello"), "m1")
    assert content_hash(chunk, "m1") != content_hash(chunk, "m2")
    assert content_hash(chunk, "m1") != content_hash(chunk_factory("a", "bye"), "m1")


def test_batched_splits_with_short_last_batch():
    assert list(batched(list(range(5)), 2)) == [[0, 1], [2, 3], [4]]


EVAL_ISSUE = "https://github.com/kubernetes/kubernetes/issues/129372"


def issue_chunks(chunk_factory, url: str, count: int) -> list[Chunk]:
    number = url.rsplit("/", 1)[1]
    return [
        replace(chunk_factory(f"issue#{number}:{i}", f"issue text {i}", "issue"), url=url)
        for i in range(count)
    ]


def test_eval_set_issues_are_never_embedded_or_stored(chunk_factory):
    eval_chunks = [
        replace(c, text=f"the eval answer {i}")
        for i, c in enumerate(issue_chunks(chunk_factory, EVAL_ISSUE, 3))
    ]
    other = [chunk_factory("doc-a"), *issue_chunks(chunk_factory, "https://x/issues/1", 1)]
    embedder, store = FakeEmbedder(), InMemoryStore()

    stats = index_chunks([*eval_chunks, *other], embedder, store, 10, excluded_urls={EVAL_ISSUE})

    embedded = [text for batch in embedder.batches for text in batch]
    assert not any("eval answer" in text for text in embedded)
    assert set(store.rows) == {c.chunk_id for c in other}
    assert all(row[1].url != EVAL_ISSUE for row in store.rows.values())
    assert (stats.excluded, stats.indexed) == (3, 2)


def test_eval_issue_indexed_before_it_joined_the_set_is_deleted(chunk_factory):
    eval_chunks = issue_chunks(chunk_factory, EVAL_ISSUE, 2)
    keep = chunk_factory("doc-a")
    store = InMemoryStore()
    index_chunks([*eval_chunks, keep], FakeEmbedder(), store, 10)
    assert len(store.rows) == 3

    stats = index_chunks([*eval_chunks, keep], FakeEmbedder(), store, 10, excluded_urls={EVAL_ISSUE})

    assert set(store.rows) == {"doc-a"}
    assert stats.deleted == 2


def test_exclusion_uses_the_eval_set_file(tmp_path, chunk_factory):
    eval_path = tmp_path / "eval_set.jsonl"
    eval_path.write_text(
        json.dumps({"id": "kubernetes/kubernetes#129372", "issue_url": EVAL_ISSUE}) + "\n",
        encoding="utf-8",
    )
    chunks = [*issue_chunks(chunk_factory, EVAL_ISSUE, 2), chunk_factory("doc-a")]

    urls = load_excluded_urls(eval_path)

    assert urls == {EVAL_ISSUE}
    assert [c.chunk_id for c in exclude_chunks(chunks, urls)] == ["doc-a"]


def test_no_eval_set_file_excludes_nothing(tmp_path):
    assert load_excluded_urls(tmp_path / "missing.jsonl") == set()
