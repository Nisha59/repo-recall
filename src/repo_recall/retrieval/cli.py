"""Compare search modes side by side from the terminal.

    python -m repo_recall.retrieval.cli "why does my pod keep restarting"
    python -m repo_recall.retrieval.cli --source-type issue -k 8
    python -m repo_recall.retrieval.cli --modes vector hybrid

With no question given, prompts for questions until an empty line.
"""

import argparse
import logging
import shutil

from repo_recall.config import Settings
from repo_recall.indexing.embedder import SentenceTransformerEmbedder
from repo_recall.retrieval.models import SearchMode, SearchResult
from repo_recall.retrieval.postgres_search import PostgresSearcher
from repo_recall.retrieval.query_encoder import InstructedQueryEncoder
from repo_recall.retrieval.retriever import Retriever

RANK_WIDTH = 3
COLUMN_GAP = 2
MIN_COLUMN_WIDTH = 24


def short_label(result: SearchResult) -> str:
    """A compact name for a result: `#123 title` for issues, the last two headings for docs."""
    if result.source_type == "issue":
        return f"#{result.extra.get('number', '?')} {result.title}"
    return " > ".join(result.heading_path[-2:]) or result.title


def truncate(text: str, width: int) -> str:
    """Cut `text` to `width` characters, marking the cut with '...'."""
    return text if len(text) <= width else text[: max(width - 3, 0)] + "..."


def format_score(result: SearchResult) -> str:
    """Score with enough precision for its mode (RRF scores are small)."""
    return f"{result.score:.4f}" if result.mode is SearchMode.HYBRID else f"{result.score:.3f}"


def cell_lines(result: SearchResult | None, ref: int | None, width: int) -> list[str]:
    """The two lines shown for one result in one mode's column."""
    if result is None:
        return ["-", ""]
    header = f"[{ref}] {format_score(result)} {result.source_type}"
    if result.component_ranks:
        header += " " + " ".join(f"{name[0]}{r}" for name, r in result.component_ranks.items())
    return [truncate(header, width), truncate(short_label(result), width)]


def assign_refs(results_by_mode: dict[SearchMode, list[SearchResult]]) -> dict[str, int]:
    """Number each distinct chunk once, in order of first appearance across modes."""
    refs: dict[str, int] = {}
    for results in results_by_mode.values():
        for result in results:
            refs.setdefault(result.chunk_id, len(refs) + 1)
    return refs


def render_side_by_side(
    results_by_mode: dict[SearchMode, list[SearchResult]], top_k: int, total_width: int
) -> str:
    """Lay out results as one row per rank and one column per mode, plus a source list."""
    modes = list(results_by_mode)
    width = max(
        MIN_COLUMN_WIDTH,
        (total_width - RANK_WIDTH - COLUMN_GAP * len(modes)) // max(len(modes), 1),
    )
    refs = assign_refs(results_by_mode)
    gap = " " * COLUMN_GAP

    def row(label: str, cells: list[str]) -> str:
        return (label.ljust(RANK_WIDTH) + gap + gap.join(c.ljust(width) for c in cells)).rstrip()

    lines = [row("#", [m.value.upper() for m in modes]), row("", ["-" * width for _ in modes])]
    for rank in range(top_k):
        per_mode = [
            results_by_mode[m][rank] if rank < len(results_by_mode[m]) else None for m in modes
        ]
        cells = [cell_lines(r, refs.get(r.chunk_id) if r else None, width) for r in per_mode]
        lines.append(row(str(rank + 1), [c[0] for c in cells]))
        lines.append(row("", [c[1] for c in cells]))

    shown = {r.chunk_id: r for results in results_by_mode.values() for r in results}
    lines.append("")
    lines.append("Sources:")
    lines += [f"  [{ref}] {chunk_id}  {shown[chunk_id].url}" for chunk_id, ref in refs.items()]
    if any(m is SearchMode.HYBRID for m in modes):
        lines.append("  (hybrid: v/k = rank in vector/keyword results)")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Compare retrieval modes side by side.")
    parser.add_argument("question", nargs="?", help="question to search (omit for a prompt loop)")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="results per mode")
    parser.add_argument("--source-type", choices=["doc", "issue"], default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        type=SearchMode,
        default=list(SearchMode),
        help="modes to show (default: all three)",
    )
    return parser.parse_args(argv)


def run_query(retriever: Retriever, question: str, args: argparse.Namespace, top_k: int) -> None:
    """Search every requested mode and print the comparison table."""
    results = {
        mode: retriever.search(question, mode=mode, top_k=top_k, source_type=args.source_type)
        for mode in args.modes
    }
    width = shutil.get_terminal_size((140, 40)).columns
    print(render_side_by_side(results, top_k, width))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    top_k = args.top_k or settings.retrieval_top_k
    encoder = InstructedQueryEncoder(
        SentenceTransformerEmbedder(settings.embedding_model), settings.query_instruction
    )
    with PostgresSearcher(settings.postgres_dsn) as searcher:
        retriever = Retriever(searcher, encoder, settings)
        if args.question:
            run_query(retriever, args.question, args, top_k)
            return
        while True:
            try:
                question = input("\nquestion (empty to quit)> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            run_query(retriever, question, args, top_k)


if __name__ == "__main__":
    main()
