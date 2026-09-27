from repo_recall.generation.models import Answer, Source, Timing, TokenUsage

DOC_URL = "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/"
ISSUE_URL = "https://github.com/kubernetes/kubernetes/issues/123"


def source(number: int, url: str, title: str, source_type: str = "doc") -> Source:
    return Source(number=number, title=title, url=url, source_type=source_type, chunks=())


def make_answer(
    question: str = "why does my pod restart?",
    cited: list[Source] | None = None,
    retrieved: int = 3,
) -> Answer:
    if cited is None:
        cited = [source(1, DOC_URL, "Pod Lifecycle"), source(3, ISSUE_URL, "Pod restarts", "issue")]
    return Answer(
        question=question,
        text="Check the liveness probe [1]; see also [3].",
        answered=bool(cited),
        cited_sources=cited,
        retrieved=[source(n, f"https://example.com/{n}", f"S{n}") for n in range(1, retrieved + 1)],
        model="test-model",
        usage=TokenUsage(1_000, 100, 0.00042),
        timing=Timing(retrieval_seconds=0.5, generation_seconds=2.0),
    )
