import itertools
import json

import pytest

from repo_recall.config import Settings
from repo_recall.evaluation.judge import Judge, load_judge_prompt
from repo_recall.evaluation.progress import ProgressLog
from repo_recall.evaluation.run import (
    RunOptions,
    budget_after,
    build_run,
    completed_modes,
    run_evaluation,
    select_examples,
    unfinished_progress,
    write_results,
)
from repo_recall.generation.answerer import Answerer
from repo_recall.generation.llm import Completion
from repo_recall.retrieval.models import SearchMode

from ..generation.conftest import chunk

DOC_A = "https://kubernetes.io/docs/a/"
DOC_B = "https://kubernetes.io/docs/b/"
UNRELATED = "https://kubernetes.io/docs/other/"


def example(n: int, source_urls: list[str]) -> dict:
    return {
        "id": f"o/r#{n}",
        "question": f"question {n}",
        "reference_answer": f"reference {n}",
        "source_urls": source_urls,
    }


EXAMPLES = [
    example(1, [DOC_A + "#probes"]),  # relevant, retrieved first
    example(2, [DOC_B]),  # relevant, retrieved at rank 7
    example(3, ["https://stackoverflow.com/q/1"]),  # nothing relevant is indexed
]


class FakeRetriever:
    """Ranked chunk URLs per question; records the modes and k it was asked for."""

    RANKINGS = {
        "question 1": [DOC_A, DOC_A, UNRELATED],
        "question 2": [UNRELATED] * 6 + [DOC_B],
        "question 3": [UNRELATED],
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def search(self, question, mode=None, top_k=None, source_type=None):
        self.calls.append((question, SearchMode(mode).value, top_k))
        return [
            chunk(f"{question}:{i}", url, rank=i + 1)
            for i, url in enumerate(self.RANKINGS[question])
        ]


class Killed(Exception):
    """Stands in for the process dying mid-run."""


class FakeLLM:
    def __init__(self, model: str, reply: str, tokens: tuple[int, int], die_on_call: int | None = None) -> None:
        self.model = model
        self.reply = reply
        self.tokens = tokens
        self.die_on_call = die_on_call
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.calls == self.die_on_call:
            raise Killed
        return Completion(self.reply, self.model, *self.tokens)


def ticking_clock():
    counter = itertools.count()
    return lambda: next(counter) * 0.5


QUESTION_COST = (1_100 + 550) / 1e6  # generation + judging, per question


def run(options: RunOptions, examples=EXAMPLES, die_on_call=None, **extra):
    settings = Settings(_env_file=None, llm_input_cost_per_mtok=1.0, llm_output_cost_per_mtok=1.0)
    retriever = FakeRetriever()
    gen = FakeLLM("gen-model", "Use a probe [1].", (1_000, 100), die_on_call)
    judge_llm = FakeLLM("judge-model", '{"correctness": 4, "faithfulness": 5, "reasoning": "ok"}', (500, 50))
    answerer = Answerer(retriever, gen, settings)
    judge = Judge(judge_llm, load_judge_prompt(), 1.0, 1.0)
    records, budget = run_evaluation(
        examples, retriever, {DOC_A, DOC_B, UNRELATED}, answerer, judge, options, ticking_clock(), **extra
    )
    return records, budget, retriever, gen


def options(**overrides) -> RunOptions:
    defaults = {"modes": [SearchMode.VECTOR, SearchMode.HYBRID], "top_k": 5, "workers": 2}
    return RunOptions(**{**defaults, **overrides})


def test_retrieval_metrics_per_question_and_summary():
    records, budget, retriever, _ = run(options())
    summary = build_run(records, budget, {}, "t")["summary"]["vector"]

    by_id = {r["id"]: r for r in records["vector"]}
    assert by_id["o/r#1"]["recall@5"] == 1.0 and by_id["o/r#1"]["reciprocal_rank"] == 1.0
    assert by_id["o/r#2"]["recall@5"] == 0.0 and by_id["o/r#2"]["recall@10"] == 1.0
    assert by_id["o/r#2"]["reciprocal_rank"] == pytest.approx(1 / 7)
    assert by_id["o/r#3"]["recall@5"] is None  # nothing relevant is indexed: not scored
    assert summary["retrieval"]["scored"] == 2
    assert summary["retrieval"]["recall@5"] == 0.5
    assert summary["retrieval"]["mrr"] == pytest.approx((1 + 1 / 7) / 2)
    assert {k for _, _, k in retriever.calls} == {10}
    assert {m for _, m, _ in retriever.calls} == {"vector", "hybrid"}


def test_answers_judging_latency_and_cost_are_summarized():
    records, budget, _, gen = run(options())
    summary = build_run(records, budget, {}, "t")["summary"]["hybrid"]

    assert gen.calls == 6  # 3 questions x 2 modes, retrieved once each
    assert summary["answers"]["generated"] == 3
    assert summary["answers"]["correctness"] == 4 and summary["answers"]["faithfulness"] == 5
    assert summary["answers"]["answered_rate"] == 1.0
    assert summary["cost"]["avg_generation_usd_per_query"] == pytest.approx(1_100 / 1e6)
    assert summary["cost"]["judge_total_usd"] == pytest.approx(3 * 550 / 1e6)
    assert summary["latency_seconds"]["retrieval_p50"] == 0.5
    assert summary["latency_seconds"]["end_to_end_p50"] is not None
    assert budget.spent == pytest.approx(6 * (1_100 + 550) / 1e6)


def test_generation_gets_only_the_top_k_chunks():
    records, _, _, _ = run(options(modes=[SearchMode.VECTOR], top_k=1))
    record = next(r for r in records["vector"] if r["id"] == "o/r#2")
    assert record["answer"] == "Use a probe [1]."
    assert record["cited_urls"] == [UNRELATED]  # only rank 1 was shown to the model


def test_retrieval_only_makes_no_llm_calls():
    records, budget, _, gen = run(options(retrieval_only=True))
    assert gen.calls == 0 and budget.spent == 0
    assert all("answer" not in r for rs in records.values() for r in rs)


def test_max_cost_stops_starting_new_questions():
    records, budget, _, gen = run(options(modes=[SearchMode.VECTOR], workers=1, max_cost=0.002))
    skipped = [r for r in records["vector"] if r.get("skipped") == "budget"]
    assert gen.calls == 1 and len(skipped) == 2
    assert build_run(records, budget, {}, "t")["stopped_early"] is True


def test_sample_is_seeded_and_shared():
    many = [example(n, []) for n in range(50)]
    first = select_examples(many, 10, seed=3)
    assert first == select_examples(many, 10, seed=3)
    assert len(first) == 10
    assert select_examples(many, None, seed=3) == many


def test_writes_timestamped_json_and_markdown(tmp_path):
    records, budget, _, _ = run(options())
    config = {
        "questions": 3, "sample": None, "seed": 0, "top_k": 5, "answer_model": "gen-model",
        "judge_model": "judge-model", "retrieval_only": False, "max_cost": 3.0,
    }
    json_path, md_path = write_results(build_run(records, budget, config, "20260926-120000"), tmp_path)

    assert json_path.name == "20260926-120000.json" and md_path.name == "20260926-120000.md"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(saved["summary"]) == {"vector", "hybrid"}
    assert len(saved["questions"]["vector"]) == 3
    table = md_path.read_text(encoding="utf-8")
    assert "| vector | 0.500 | 1.000 |" in table
    assert "| hybrid |" in table and "limit $3.00" in table


def test_progress_is_reported_after_each_mode():
    saved: list[list[str]] = []
    run(options(), on_mode_done=lambda records, budget: saved.append(list(records)))
    assert saved == [["vector"], ["vector", "hybrid"]]


def test_resume_skips_finished_modes_and_continues_the_spend():
    first, budget, _, _ = run(options(modes=[SearchMode.VECTOR]))
    previous = build_run(first, budget, {}, "t")

    done = completed_modes(previous, EXAMPLES)
    records, resumed_budget, retriever, gen = run(
        options(), completed=done, budget=budget_after(previous, None)
    )

    assert {m for _, m, _ in retriever.calls} == {"hybrid"}  # vector not searched again
    assert gen.calls == 3
    assert records["vector"] == first["vector"]
    assert resumed_budget.spent == pytest.approx(2 * budget.spent)


def test_each_answered_question_is_reported_but_not_budget_skipped_ones():
    reported: list[str] = []
    run(
        options(modes=[SearchMode.VECTOR], workers=1, max_cost=0.002),
        on_question_done=lambda r: reported.append(r["id"]),
    )
    assert reported == ["o/r#1"]  # the other two were skipped for budget, so a resume runs them


def test_a_run_killed_mid_mode_resumes_from_the_progress_log(tmp_path):
    log = ProgressLog(tmp_path / "t.progress.jsonl")
    # Vector finishes and is saved to the results JSON; hybrid dies on its 3rd question.
    vector_records, vector_budget, _, _ = run(options(modes=[SearchMode.VECTOR]))
    previous = build_run(vector_records, vector_budget, {}, "t")
    with pytest.raises(Killed):
        run(
            options(workers=1),
            die_on_call=3,
            completed=completed_modes(previous, EXAMPLES),
            budget=budget_after(previous, None),
            on_question_done=log.append,
        )

    partial = unfinished_progress(log.load(), completed_modes(previous, EXAMPLES))
    assert {m: sorted(rs) for m, rs in partial.items()} == {"hybrid": ["o/r#1", "o/r#2"]}

    records, budget, retriever, gen = run(
        options(),
        completed=completed_modes(previous, EXAMPLES),
        partial=partial,
        budget=budget_after(previous, None, partial),
    )
    assert retriever.calls == [("question 3", "hybrid", 10)]  # only the unfinished question
    assert gen.calls == 1
    assert [r["id"] for r in records["hybrid"]] == ["o/r#1", "o/r#2", "o/r#3"]
    assert all("answer" in r for r in records["hybrid"])
    assert budget.spent == pytest.approx(6 * QUESTION_COST)  # each question paid for exactly once


def test_logged_questions_of_finished_modes_are_not_counted_twice():
    logged = {"vector": {"o/r#1": {"id": "o/r#1", "mode": "vector"}}, "hybrid": {}}
    assert unfinished_progress(logged, {"vector": []}) == {"hybrid": {}}


def test_modes_with_missing_or_skipped_questions_are_not_counted_as_done():
    records, budget, _, _ = run(options(modes=[SearchMode.VECTOR], workers=1, max_cost=0.002))
    previous = build_run(records, budget, {}, "t")
    assert completed_modes(previous, EXAMPLES) == {}  # stopped early at the budget
    finished = build_run(run(options(modes=[SearchMode.VECTOR]))[0], budget, {}, "t")
    assert completed_modes(finished, EXAMPLES[:2]) == {}  # different question set
