"""The HTTP app: POST /ask, GET /health, and the Gradio UI at /ui.

    uvicorn repo_recall.api.app:create_app --factory      # or: python -m repo_recall.api

The embedding model, database connection and LLM client are built once at
startup (see `build_services`); tests pass their own factory with fakes.
"""

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

import gradio as gr
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from repo_recall.api.schemas import AskRequest, AskResponse, HealthResponse
from repo_recall.api.ui import build_ui
from repo_recall.config import Settings
from repo_recall.generation.answerer import Answerer
from repo_recall.generation.llm import LLMError, make_llm_client
from repo_recall.generation.models import Answer
from repo_recall.indexing.embedder import SentenceTransformerEmbedder
from repo_recall.retrieval.models import SearchMode
from repo_recall.retrieval.postgres_search import PostgresSearcher
from repo_recall.retrieval.query_encoder import InstructedQueryEncoder
from repo_recall.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

UI_PATH = "/ui"


class QuestionAnswerer(Protocol):
    """What the app needs to answer a question (the Answerer class fits)."""

    def answer(self, question: str, top_k: int | None = None, source_type: str | None = None) -> Answer:
        """A cited answer to `question`."""
        ...


@dataclass
class Services:
    """The long-lived pieces the endpoints use, and how to shut them down."""

    answerer: QuestionAnswerer
    database_ok: Callable[[], bool]
    close: Callable[[], None]


WARM_UP_QUESTION = "warm-up query"


def warm_up(retriever: Retriever) -> None:
    """Run one throwaway search so the first real question isn't slow.

    The first search after startup took ~20s (model and vector index loading
    into memory); later ones take ~0.1s. Failure is only logged: the app
    still starts, and /health reports the database as unreachable.
    """
    start = time.perf_counter()
    try:
        retriever.search(WARM_UP_QUESTION, mode=SearchMode.HYBRID)
    except psycopg.Error as exc:
        logger.warning("Warm-up search failed; is the database up? %s", exc)
        return
    logger.info("Warm-up search took %.1fs", time.perf_counter() - start)


def build_services(settings: Settings) -> Services:
    """Load the embedding model, connect the retriever and LLM client, and warm up search."""
    encoder = InstructedQueryEncoder(
        SentenceTransformerEmbedder(settings.embedding_model), settings.query_instruction
    )
    searcher = PostgresSearcher(settings.postgres_dsn)
    retriever = Retriever(searcher, encoder, settings)
    warm_up(retriever)
    answerer = Answerer(retriever, make_llm_client(settings), settings)
    return Services(answerer=answerer, database_ok=searcher.ping, close=searcher.close)


def get_services(request: Request) -> Services:
    """The services built at startup (a FastAPI dependency)."""
    return request.app.state.services


def create_app(
    settings: Settings | None = None,
    services_factory: Callable[[Settings], Services] = build_services,
) -> FastAPI:
    """Build the FastAPI app with the Gradio UI mounted at /ui."""
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = services_factory(settings)
        app.state.services = services
        logger.info("RepoRecall is ready for questions about %s", settings.github_repo)
        try:
            yield
        finally:
            services.close()

    app = FastAPI(
        title="RepoRecall",
        description=f"Answers questions about {settings.github_repo} from its docs and closed issues.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(response: Response, services: Services = Depends(get_services)) -> HealthResponse:
        """200 if the search index is reachable, 503 if not."""
        if services.database_ok():
            return HealthResponse(status="ok", database="ok")
        response.status_code = 503
        return HealthResponse(status="unavailable", database="unreachable")

    @app.post("/ask", responses={502: {"description": "LLM failed"}, 503: {"description": "Index unreachable"}})
    def ask(body: AskRequest, services: Services = Depends(get_services)) -> AskResponse:
        """Answer a question with cited sources, timing and cost."""
        try:
            answer = services.answerer.answer(body.question, top_k=body.top_k, source_type=body.source_type)
        except LLMError as exc:
            logger.error("LLM request failed: %s", exc)
            raise HTTPException(status_code=502, detail="The language model request failed.") from exc
        except psycopg.OperationalError as exc:
            logger.error("Search index unreachable: %s", exc)
            raise HTTPException(status_code=503, detail="The search index is unreachable.") from exc
        return AskResponse.from_answer(answer)

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        """Send browsers to the UI."""
        return RedirectResponse(UI_PATH)

    ui = build_ui(lambda question: app.state.services.answerer.answer(question), settings.github_repo)
    return gr.mount_gradio_app(app, ui, path=UI_PATH)
