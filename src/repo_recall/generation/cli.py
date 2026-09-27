"""Ask a question from the terminal and get a cited answer.

    python -m repo_recall.generation.cli "why does my pod keep restarting"
    python -m repo_recall.generation.cli --source-type issue -k 8 "..."

With no question given, prompts for questions until an empty line.
"""

import argparse
import logging

from repo_recall.config import Settings
from repo_recall.generation.answerer import Answerer
from repo_recall.generation.llm import LLMError, make_llm_client
from repo_recall.generation.models import Answer
from repo_recall.indexing.embedder import SentenceTransformerEmbedder
from repo_recall.retrieval.postgres_search import PostgresSearcher
from repo_recall.retrieval.query_encoder import InstructedQueryEncoder
from repo_recall.retrieval.retriever import Retriever


def render_answer(answer: Answer) -> str:
    """The answer, its cited sources, and a one-line timing/cost footer."""
    lines = [answer.text, ""]
    if answer.cited_sources:
        lines.append("Sources:")
        lines += [f"  [{s.number}] {s.title}  {s.url}" for s in answer.cited_sources]
    else:
        lines.append(f"(no sources cited; {len(answer.retrieved)} were retrieved)")
    usage, timing = answer.usage, answer.timing
    cost = "cost n/a" if usage.estimated_cost_usd is None else f"~${usage.estimated_cost_usd:.4f}"
    lines.append("")
    lines.append(
        f"{answer.model} | retrieval {timing.retrieval_seconds:.2f}s, "
        f"generation {timing.generation_seconds:.2f}s | "
        f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens, {cost}"
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Answer a question from docs and issues.")
    parser.add_argument("question", nargs="?", help="question to answer (omit for a prompt loop)")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="chunks to retrieve")
    parser.add_argument("--source-type", choices=["doc", "issue"], default=None)
    return parser.parse_args(argv)


def ask(answerer: Answerer, question: str, args: argparse.Namespace) -> None:
    """Answer one question and print it, reporting LLM failures without crashing the loop."""
    try:
        answer = answerer.answer(question, top_k=args.top_k, source_type=args.source_type)
    except LLMError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return
    print(render_answer(answer))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    encoder = InstructedQueryEncoder(
        SentenceTransformerEmbedder(settings.embedding_model), settings.query_instruction
    )
    with PostgresSearcher(settings.postgres_dsn) as searcher:
        answerer = Answerer(Retriever(searcher, encoder, settings), make_llm_client(settings), settings)
        if args.question:
            ask(answerer, args.question, args)
            return
        while True:
            try:
                question = input("\nquestion (empty to quit)> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            ask(answerer, question, args)


if __name__ == "__main__":
    main()
