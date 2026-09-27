from repo_recall.retrieval.query_encoder import InstructedQueryEncoder


class RecordingEmbedder:
    model_name = "m"
    dimension = 2

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def test_query_encoder_prefixes_instruction():
    embedder = RecordingEmbedder()
    vector = InstructedQueryEncoder(embedder, "Query: ").encode("pods")
    assert embedder.texts == ["Query: pods"]
    assert vector == [1.0, 0.0]


def test_query_encoder_with_empty_instruction_passes_question_unchanged():
    embedder = RecordingEmbedder()
    InstructedQueryEncoder(embedder, "").encode("pods")
    assert embedder.texts == ["pods"]
