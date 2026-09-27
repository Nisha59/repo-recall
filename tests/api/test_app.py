import psycopg
import pytest
from fastapi.testclient import TestClient

from repo_recall.api.app import WARM_UP_QUESTION, Services, create_app, warm_up
from repo_recall.config import Settings
from repo_recall.generation.llm import LLMError
from repo_recall.retrieval.models import SearchMode

from .conftest import DOC_URL, ISSUE_URL, make_answer


class FakeAnswerer:
    """Returns a canned answer (or raises `error`) and records what it was asked."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, int | None, str | None]] = []

    def answer(self, question, top_k=None, source_type=None):
        self.calls.append((question, top_k, source_type))
        if self.error is not None:
            raise self.error
        return make_answer(question)


class World:
    """What the fake services report, adjustable per test."""

    def __init__(self) -> None:
        self.answerer = FakeAnswerer()
        self.database_up = True
        self.closed = False

    def services(self, settings: Settings) -> Services:
        def close() -> None:
            self.closed = True

        return Services(answerer=self.answerer, database_ok=lambda: self.database_up, close=close)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def client(world):
    settings = Settings(_env_file=None, github_repo="o/r")
    with TestClient(create_app(settings, world.services)) as client:
        yield client


def test_health_is_ok_when_the_database_answers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_is_503_when_the_database_is_down(client, world):
    world.database_up = False
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unreachable"}


def test_ask_returns_answer_sources_timing_and_usage(client, world):
    response = client.post("/ask", json={"question": "  why does my pod restart?  "})

    assert response.status_code == 200
    body = response.json()
    assert world.answerer.calls == [("why does my pod restart?", None, None)]  # stripped
    assert body["answer"] == "Check the liveness probe [1]; see also [3]."
    assert body["answered"] is True
    assert body["sources"] == [
        {"number": 1, "title": "Pod Lifecycle", "url": DOC_URL, "source_type": "doc"},
        {"number": 3, "title": "Pod restarts", "url": ISSUE_URL, "source_type": "issue"},
    ]
    assert body["retrieved_count"] == 3
    assert body["timing"] == {"retrieval_seconds": 0.5, "generation_seconds": 2.0, "total_seconds": 2.5}
    assert body["usage"] == {
        "model": "test-model", "prompt_tokens": 1000, "completion_tokens": 100, "estimated_cost_usd": 0.00042,
    }


def test_ask_passes_top_k_and_source_type_through(client, world):
    client.post("/ask", json={"question": "q", "top_k": 8, "source_type": "issue"})
    assert world.answerer.calls == [("q", 8, "issue")]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 20_001},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 21},
        {"question": "q", "source_type": "wiki"},
    ],
)
def test_invalid_requests_are_rejected_without_calling_the_model(client, world, body):
    assert client.post("/ask", json=body).status_code == 422
    assert world.answerer.calls == []


def test_llm_failure_is_a_502(client, world):
    world.answerer.error = LLMError("upstream exploded with secret details")
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "secret details" not in response.text  # internals stay in the log


def test_unreachable_index_is_a_503(client, world):
    world.answerer.error = psycopg.OperationalError("connection refused")
    assert client.post("/ask", json={"question": "q"}).status_code == 503


def test_root_redirects_to_the_ui_which_is_served(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == "/ui"
    assert client.get("/ui/").status_code == 200


def test_services_are_closed_on_shutdown(world):
    with TestClient(create_app(Settings(_env_file=None), world.services)):
        assert not world.closed
    assert world.closed


class RecordingRetriever:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def search(self, question, mode=None, top_k=None, source_type=None):
        self.calls.append((question, mode))
        if self.error is not None:
            raise self.error
        return []


def test_warm_up_runs_one_hybrid_search():
    retriever = RecordingRetriever()
    warm_up(retriever)
    assert retriever.calls == [(WARM_UP_QUESTION, SearchMode.HYBRID)]  # hybrid touches both indexes


def test_warm_up_does_not_stop_startup_when_the_database_is_down():
    warm_up(RecordingRetriever(psycopg.OperationalError("connection refused")))  # no exception
