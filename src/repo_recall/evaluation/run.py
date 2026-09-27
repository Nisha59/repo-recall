"""Evaluate retrieval, answers, latency and cost for each retrieval mode.

    python -m repo_recall.evaluation.run                      # all questions, all modes
    python -m repo_recall.evaluation.run --sample 30          # quick subset
    python -m repo_recall.evaluation.run --retrieval-only     # no LLM calls, free
    python -m repo_recall.evaluation.run --max-cost 3         # stop before spending > $3
    python -m repo_recall.evaluation.run --resume data/eval/results/<timestamp>.json

For each mode, every question is retrieved once (k=10, sequentially, so
retrieval latency isn't distorted by concurrency). The top `retrieval_top_k`
chunks of that search are what a normal k=5 search returns, and they go to
the answer model; answers are generated and judged in a small thread pool.
Results go to `<eval_results_dir>/<timestamp>.json` and `.md`, saved after
each mode. Every answered question is also appended to
`<timestamp>.progress.jsonl` as soon as it finishes, so an interrupted run
loses at most the questions in flight: `--resume` keeps the finished ones
(and their spend) and only retrieves, answers and judges the rest.
"""

import argparse
import json
import logging
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from repo_recall.config import Settings
from repo_recall.evaluation.judge import Judge, make_judge
from repo_recall.evaluation.metrics import (
    mean,
    normalize_url,
    percentile,
    recall_at_k,
    reciprocal_rank,
    relevant_sources,
    source_ranks,
)
from repo_recall.evaluation.progress import ProgressLog, progress_path
from repo_recall.generation.answerer import Answerer
from repo_recall.generation.llm import LLMError, make_llm_client
from repo_recall.generation.models import Answer
from repo_recall.indexing.embedder import SentenceTransformerEmbedder
from repo_recall.processing.chunk_all import read_jsonl
from repo_recall.retrieval.models import SearchMode, SearchResult
from repo_recall.retrieval.postgres_search import PostgresSearcher
from repo_recall.retrieval.query_encoder import InstructedQueryEncoder
from repo_recall.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

RETRIEVAL_K = 10
RECALL_KS = (5, 10)


class ModeRetriever(Protocol):
    """What the runner needs from retrieval (the Retriever class fits)."""

    def search(
        self, question: str, mode: SearchMode | str | None = None, top_k: int | None = None
    ) -> list[SearchResult]:
        """The top chunks for `question` in one mode."""
        ...


@dataclass
class Budget:
    """Tracks spend across threads and refuses new work that would exceed the limit.

    A question is only started if the spend so far plus the average cost of
    the questions finished so far stays within `max_cost` (the first ones,
    with no cost data yet, start while anything is left). Questions already
    running finish, so the total can overshoot by at most `workers` questions.
    """

    max_cost: float | None
    spent: float = 0.0
    questions: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def allows_next(self) -> bool:
        """True if one more question fits in the budget."""
        with self._lock:
            if self.max_cost is None:
                return True
            per_question = self.spent / self.questions if self.questions else 0.0
            return self.spent + per_question <= self.max_cost and self.spent < self.max_cost

    def record(self, cost: float) -> None:
        """Add one finished question's cost."""
        with self._lock:
            self.spent += cost
            self.questions += 1


@dataclass(frozen=True)
class RunOptions:
    """Choices for one evaluation run."""

    modes: list[SearchMode]
    top_k: int
    retrieval_only: bool = False
    workers: int = 4
    max_cost: float | None = None


# --- Retrieval --------------------------------------------------------------


def score_retrieval(results: list[SearchResult], example: dict, indexed: set[str]) -> dict:
    """Recall@k and reciprocal rank for one question, or None metrics if nothing is relevant."""
    retrieved = [r.url for r in results]
    relevant = relevant_sources(example["source_urls"], indexed)
    ranks = source_ranks(retrieved)
    scores: dict[str, Any] = {
        "retrieved_urls": list(dict.fromkeys(retrieved)),
        "relevant_urls": sorted(relevant),
        "relevant_ranks": {url: ranks.get(url) for url in sorted(relevant)},
    }
    for k in RECALL_KS:
        scores[f"recall@{k}"] = recall_at_k(ranks, relevant, k) if relevant else None
    scores["reciprocal_rank"] = reciprocal_rank(ranks, relevant) if relevant else None
    return scores


def retrieve_all(
    retriever: ModeRetriever,
    examples: list[dict],
    mode: SearchMode,
    indexed: set[str],
    clock: Callable[[], float],
) -> list[tuple[dict, list[SearchResult]]]:
    """Search every question once in `mode`; return per-question records and raw results."""
    out = []
    for example in examples:
        start = clock()
        results = retriever.search(example["question"], mode=mode, top_k=RETRIEVAL_K)
        seconds = clock() - start
        record = {"id": example["id"], "mode": mode.value, "retrieval_seconds": seconds}
        record.update(score_retrieval(results, example, indexed))
        out.append((record, results))
    return out


# --- Generation and judging -------------------------------------------------


def answer_fields(answer: Answer) -> dict:
    """The parts of an answer worth keeping in the results file."""
    return {
        "answer": answer.text,
        "answered": answer.answered,
        "cited_urls": [s.url for s in answer.cited_sources],
        "generation_seconds": answer.timing.generation_seconds,
        "end_to_end_seconds": answer.timing.total_seconds,
        "prompt_tokens": answer.usage.prompt_tokens,
        "completion_tokens": answer.usage.completion_tokens,
        "generation_cost_usd": answer.usage.estimated_cost_usd,
    }


def answer_and_judge(
    record: dict,
    results: list[SearchResult],
    example: dict,
    answerer: Answerer,
    judge: Judge,
    top_k: int,
    budget: Budget,
) -> None:
    """Generate and judge one answer, filling `record` in place (skipped if over budget)."""
    if not budget.allows_next():
        record["skipped"] = "budget"
        return
    try:
        answer = answerer.answer_from_results(
            example["question"], results[:top_k], record["retrieval_seconds"]
        )
    except LLMError as exc:
        logger.warning("Generation failed for %s (%s): %s", example["id"], record["mode"], exc)
        record["error"] = str(exc)
        budget.record(0.0)
        return
    record.update(answer_fields(answer))
    verdict = judge.grade(example, answer)
    record.update(
        {
            "correctness": verdict.correctness,
            "faithfulness": verdict.faithfulness,
            "judge_reasoning": verdict.reasoning,
            "judge_cost_usd": verdict.usage.estimated_cost_usd,
        }
    )
    budget.record((answer.usage.estimated_cost_usd or 0.0) + (verdict.usage.estimated_cost_usd or 0.0))


def answer_all(
    pairs: list[tuple[dict, list[SearchResult]]],
    examples: dict[str, dict],
    answerer: Answerer,
    judge: Judge,
    options: RunOptions,
    budget: Budget,
    on_question_done: Callable[[dict], None] | None = None,
) -> None:
    """Answer and judge every question of one mode in a thread pool.

    `on_question_done` gets each answered record as soon as it finishes (on
    this thread, so it needn't be thread-safe); skipped and failed questions
    aren't reported, so a resumed run tries them again.
    """
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = {
            pool.submit(
                answer_and_judge, record, results, examples[record["id"]], answerer, judge,
                options.top_k, budget,
            ): record
            for record, results in pairs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            record = futures[future]
            if on_question_done is not None and "answer" in record:
                on_question_done(record)
            if done % 10 == 0 or done == len(futures):
                logger.info("  answered %d/%d (spent $%.4f)", done, len(futures), budget.spent)


# --- Summaries --------------------------------------------------------------


def _values(records: list[dict], key: str) -> list[float]:
    """Non-null values of `key` across records."""
    return [r[key] for r in records if r.get(key) is not None]


def summarize_mode(records: list[dict]) -> dict:
    """Retrieval, answer-quality, latency and cost figures for one mode."""
    answered = [r for r in records if "answer" in r]
    judged = [r for r in answered if r.get("correctness") is not None]
    return {
        "questions": len(records),
        "retrieval": {
            "scored": len(_values(records, "reciprocal_rank")),
            "recall@5": mean(_values(records, "recall@5")),
            "recall@10": mean(_values(records, "recall@10")),
            "mrr": mean(_values(records, "reciprocal_rank")),
        },
        "answers": {
            "generated": len(answered),
            "answered_rate": mean([float(r["answered"]) for r in answered]),
            "judged": len(judged),
            "judge_failures": len(answered) - len(judged),
            "correctness": mean(_values(judged, "correctness")),
            "faithfulness": mean(_values(judged, "faithfulness")),
        },
        "latency_seconds": {
            "retrieval_p50": percentile(_values(records, "retrieval_seconds"), 50),
            "retrieval_p95": percentile(_values(records, "retrieval_seconds"), 95),
            "end_to_end_p50": percentile(_values(answered, "end_to_end_seconds"), 50),
            "end_to_end_p95": percentile(_values(answered, "end_to_end_seconds"), 95),
        },
        "cost": {
            "avg_generation_usd_per_query": mean(_values(answered, "generation_cost_usd")),
            "avg_prompt_tokens": mean(_values(answered, "prompt_tokens")),
            "avg_completion_tokens": mean(_values(answered, "completion_tokens")),
            "generation_total_usd": sum(_values(answered, "generation_cost_usd")),
            "judge_total_usd": sum(_values(answered, "judge_cost_usd")),
        },
        "skipped_for_budget": sum(1 for r in records if r.get("skipped") == "budget"),
        "generation_errors": sum(1 for r in records if "error" in r),
    }


def _fmt(value: float | None, pattern: str) -> str:
    """Format a metric, showing missing values as a dash."""
    return "–" if value is None else pattern.format(value)


def format_markdown(run: dict) -> str:
    """A Markdown report: one summary row per mode, then run details."""
    header = (
        "| Mode | Recall@5 | Recall@10 | MRR | Correctness (1-5) | Faithfulness (1-5) "
        "| Answered | Latency p50 | Latency p95 | Cost / query |"
    )
    lines = [f"# Evaluation run {run['timestamp']}", "", header, "|" + "---|" * 10]
    for mode, s in run["summary"].items():
        r, a, lat, c = s["retrieval"], s["answers"], s["latency_seconds"], s["cost"]
        p50 = lat["end_to_end_p50"] if lat["end_to_end_p50"] is not None else lat["retrieval_p50"]
        p95 = lat["end_to_end_p95"] if lat["end_to_end_p95"] is not None else lat["retrieval_p95"]
        lines.append(
            f"| {mode} | {_fmt(r['recall@5'], '{:.3f}')} | {_fmt(r['recall@10'], '{:.3f}')} "
            f"| {_fmt(r['mrr'], '{:.3f}')} | {_fmt(a['correctness'], '{:.2f}')} "
            f"| {_fmt(a['faithfulness'], '{:.2f}')} | {_fmt(a['answered_rate'], '{:.0%}')} "
            f"| {_fmt(p50, '{:.2f}s')} | {_fmt(p95, '{:.2f}s')} "
            f"| {_fmt(c['avg_generation_usd_per_query'], '${:.5f}')} |"
        )
    cfg = run["config"]
    first = next(iter(run["summary"].values()), None)
    lines += [
        "",
        f"- Questions: {cfg['questions']} ({'sample of ' + str(cfg['sample']) if cfg['sample'] else 'full set'}"
        f"{', seed ' + str(cfg['seed']) if cfg['sample'] else ''})",
        f"- Retrieval metrics over {first['retrieval']['scored'] if first else 0} questions whose "
        "reference answer links an indexed page; ranks are chunk ranks in the top 10.",
        f"- Answer model: {cfg['answer_model']} (top {cfg['top_k']} chunks); judge: {cfg['judge_model'] or 'none'}",
        "- Latency: end-to-end = retrieval + generation"
        + (" (retrieval only in this run)" if cfg["retrieval_only"] else ""),
        f"- Spend: ${run['spent_usd']:.4f} total (generation + judging)"
        + (f", limit ${cfg['max_cost']:.2f}" if cfg["max_cost"] is not None else "")
        + (" — stopped early at the limit" if run["stopped_early"] else ""),
    ]
    return "\n".join(lines) + "\n"


# --- Running ----------------------------------------------------------------


def select_examples(examples: list[dict], sample: int | None, seed: int) -> list[dict]:
    """All examples, or a seeded random subset of `sample` of them (same for every mode)."""
    if sample is None or sample >= len(examples):
        return examples
    return random.Random(seed).sample(examples, sample)


def run_evaluation(
    examples: list[dict],
    retriever: ModeRetriever,
    indexed_urls: set[str],
    answerer: Answerer | None,
    judge: Judge | None,
    options: RunOptions,
    clock: Callable[[], float],
    budget: Budget | None = None,
    completed: dict[str, list[dict]] | None = None,
    partial: dict[str, dict[str, dict]] | None = None,
    on_mode_done: Callable[[dict[str, list[dict]], Budget], None] | None = None,
    on_question_done: Callable[[dict], None] | None = None,
) -> tuple[dict[str, list[dict]], Budget]:
    """Run every mode; return per-mode question records and the spend tracker.

    Modes already in `completed` (from an interrupted run) are kept and not
    re-run. Questions in `partial` ({mode: {id: record}}, from the progress
    log) are kept and only the rest of their mode is run. `on_mode_done` is
    called after each mode with everything so far, and `on_question_done`
    after each answered question, so the caller can save progress.
    """
    indexed = {normalize_url(url) for url in indexed_urls}
    by_id = {e["id"]: e for e in examples}
    budget = budget or Budget(options.max_cost)
    records: dict[str, list[dict]] = dict(completed or {})
    for mode in options.modes:
        if mode.value in records:
            logger.info("Mode %s: already done, skipping", mode.value)
            continue
        done = (partial or {}).get(mode.value, {})
        todo = [e for e in examples if e["id"] not in done]
        logger.info("Mode %s: retrieving %d questions (%d already done)", mode.value, len(todo), len(done))
        pairs = retrieve_all(retriever, todo, mode, indexed, clock)
        if not options.retrieval_only and answerer is not None and judge is not None:
            answer_all(pairs, by_id, answerer, judge, options, budget, on_question_done)
        new = {record["id"]: record for record, _ in pairs}
        records[mode.value] = [done.get(e["id"]) or new[e["id"]] for e in examples]
        if on_mode_done is not None:
            on_mode_done(records, budget)
    return records, budget


def completed_modes(previous: dict, examples: list[dict]) -> dict[str, list[dict]]:
    """Modes of an earlier run that finished for exactly these questions.

    A mode counts as finished if it has a record for every question and none
    were skipped for budget; anything else is re-run.
    """
    ids = sorted(e["id"] for e in examples)
    done = {}
    for mode, records in previous.get("questions", {}).items():
        same_questions = sorted(r["id"] for r in records) == ids
        if same_questions and not any(r.get("skipped") for r in records):
            done[mode] = records
    return done


def unfinished_progress(
    logged: dict[str, dict[str, dict]], completed: dict[str, list[dict]]
) -> dict[str, dict[str, dict]]:
    """Logged questions of modes that didn't finish (finished modes come from the results JSON)."""
    return {mode: rs for mode, rs in logged.items() if mode not in completed}


def record_cost(record: dict) -> float:
    """What generating and judging one question cost."""
    return (record.get("generation_cost_usd") or 0.0) + (record.get("judge_cost_usd") or 0.0)


def budget_after(
    previous: dict | None,
    max_cost: float | None,
    partial: dict[str, dict[str, dict]] | None = None,
) -> Budget:
    """A spend tracker that continues from an earlier run's spend, if any.

    `previous["spent_usd"]` covers the modes saved to the results JSON;
    questions from `partial` (unfinished modes) are added on top.
    """
    spent, answered = 0.0, 0
    if previous is not None:
        spent = previous["spent_usd"]
        answered = sum(1 for rs in previous["questions"].values() for r in rs if "answer" in r)
    for records in (partial or {}).values():
        spent += sum(record_cost(r) for r in records.values())
        answered += len(records)
    return Budget(max_cost, spent=spent, questions=answered)


def build_run(
    records: dict[str, list[dict]],
    budget: Budget,
    config: dict,
    timestamp: str,
) -> dict:
    """The full results document: config, per-mode summaries, per-question records."""
    return {
        "timestamp": timestamp,
        "config": config,
        "summary": {mode: summarize_mode(rs) for mode, rs in records.items()},
        "spent_usd": budget.spent,
        "stopped_early": any(r.get("skipped") == "budget" for rs in records.values() for r in rs),
        "questions": records,
    }


def write_results(run: dict, results_dir: Path) -> tuple[Path, Path]:
    """Write `<timestamp>.json` and `<timestamp>.md`; return both paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{run['timestamp']}.json"
    md_path = results_dir / f"{run['timestamp']}.md"
    json_path.write_text(json.dumps(run, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path.write_text(format_markdown(run), encoding="utf-8")
    return json_path, md_path


def git_commit() -> str | None:
    """The current commit hash, if this is a git checkout with commits."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers per mode.")
    parser.add_argument("--sample", type=int, default=None, help="evaluate a random subset of N questions")
    parser.add_argument("--seed", type=int, default=0, help="seed for --sample")
    parser.add_argument(
        "--modes", nargs="+", type=SearchMode, default=list(SearchMode), help="modes to evaluate"
    )
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation and judging")
    parser.add_argument("--max-cost", type=float, default=None, help="stop starting questions past this USD spend")
    parser.add_argument("--workers", type=int, default=4, help="parallel answer/judge calls")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue an interrupted run from its results JSON (keeps its questions and settings)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "repo_recall.generation.answerer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")

    previous = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
    if previous is not None:
        saved = previous["config"]
        args.sample, args.seed = saved["sample"], saved["seed"]
        args.modes = [SearchMode(m) for m in saved["modes"]]
        args.retrieval_only = saved["retrieval_only"]
    examples = select_examples(read_jsonl(settings.eval_set_path), args.sample, args.seed)
    completed = completed_modes(previous, examples) if previous else {}
    timestamp = previous["timestamp"] if previous else datetime.now().strftime("%Y%m%d-%H%M%S")
    progress = ProgressLog(progress_path(settings.eval_results_dir, timestamp))
    partial = unfinished_progress(progress.load(), completed) if previous else {}
    if previous is not None:
        logger.info(
            "Resuming %s: modes already done: %s; questions already done in unfinished modes: %s",
            args.resume.name,
            ", ".join(completed) or "none",
            ", ".join(f"{mode} {len(rs)}" for mode, rs in partial.items()) or "none",
        )
    options = RunOptions(
        modes=args.modes,
        top_k=settings.retrieval_top_k,
        retrieval_only=args.retrieval_only,
        workers=args.workers,
        max_cost=args.max_cost,
    )
    config = {
        "questions": len(examples),
        "sample": args.sample,
        "seed": args.seed,
        "modes": [m.value for m in args.modes],
        "top_k": settings.retrieval_top_k,
        "retrieval_k": RETRIEVAL_K,
        "max_per_source": settings.retrieval_max_per_source,
        "embedding_model": settings.embedding_model,
        "answer_model": None if args.retrieval_only else settings.llm_model,
        "judge_model": None if args.retrieval_only else settings.eval_judge_model,
        "retrieval_only": args.retrieval_only,
        "max_cost": args.max_cost,
        "git_commit": git_commit(),
    }

    def save_progress(records: dict[str, list[dict]], budget: Budget) -> None:
        json_path, _ = write_results(build_run(records, budget, config, timestamp), settings.eval_results_dir)
        logger.info("Saved progress (%s) to %s", ", ".join(records), json_path)

    if previous is None:
        # Written up front so a run killed during its first mode can still be resumed.
        save_progress({}, Budget(args.max_cost))

    answerer = judge = None
    encoder = InstructedQueryEncoder(
        SentenceTransformerEmbedder(settings.embedding_model), settings.query_instruction
    )
    with PostgresSearcher(settings.postgres_dsn) as searcher:
        retriever = Retriever(searcher, encoder, settings)
        if not args.retrieval_only:
            answerer = Answerer(retriever, make_llm_client(settings), settings)
            judge = make_judge(settings)
        records, budget = run_evaluation(
            examples,
            retriever,
            searcher.indexed_urls(),
            answerer,
            judge,
            options,
            time.perf_counter,
            budget=budget_after(previous, args.max_cost, partial),
            completed=completed,
            partial=partial,
            on_mode_done=save_progress,
            on_question_done=progress.append,
        )

    run = build_run(records, budget, config, timestamp)
    json_path, md_path = write_results(run, settings.eval_results_dir)
    logger.info("Wrote %s and %s", json_path, md_path)
    print(format_markdown(run))


if __name__ == "__main__":
    main()
