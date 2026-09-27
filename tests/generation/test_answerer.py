import itertools
import logging

import pytest

from repo_recall.config import Settings
from repo_recall.generation.answerer import Answerer
from repo_recall.generation.llm import Completion
from repo_recall.generation.prompt import NO_ANSWER

from .conftest import chunk


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question, top_k=None, source_type=None):
        self.calls.append((question, top_k, source_type))
        return self.results


class FakeLLM:
    model = "fake-model"

    def __init__(self, text: str, prompt_tokens=1_000, completion_tokens=100):
        self.completion = Completion(text, self.model, prompt_tokens, completion_tokens)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return self.completion


def ticking_clock(step: float = 0.5):
    """perf_counter stand-in: every call is `step` seconds after the last."""
    counter = itertools.count()
    return lambda: next(counter) * step


def make_answerer(results, llm):
    settings = Settings(_env_file=None, llm_input_cost_per_mtok=1.0, llm_output_cost_per_mtok=4.0)
    return Answerer(FakeRetriever(results), llm, settings, clock=ticking_clock())


RESULTS = [chunk("a1", "page/a"), chunk("b1", "page/b"), chunk("a2", "page/a")]


def test_answer_has_text_cited_sources_timing_and_cost():
    answer = make_answerer(RESULTS, FakeLLM("Pods restart per policy [2], see also [1].")).answer(
        "why?"
    )

    assert answer.answered
    assert answer.text == "Pods restart per policy [2], see also [1]."
    assert [(s.number, s.url) for s in answer.cited_sources] == [(2, "page/b"), (1, "page/a")]
    assert len(answer.retrieved) == 2
    assert answer.timing.retrieval_seconds == 0.5
    assert answer.timing.generation_seconds == 0.5
    assert answer.usage.estimated_cost_usd == pytest.approx((1_000 * 1.0 + 100 * 4.0) / 1e6)
    assert answer.model == "fake-model"


def test_prompt_sent_to_llm_contains_sources_and_question():
    llm = FakeLLM("x [1]")
    make_answerer(RESULTS, llm).answer("why do pods restart?")
    system, user = llm.prompts[0]
    assert NO_ANSWER in system
    assert "[1] Documentation" in user and "[2] Documentation" in user
    assert user.endswith("Question: why do pods restart?")


def test_retrieval_options_are_passed_through():
    answerer = make_answerer(RESULTS, FakeLLM("x [1]"))
    answerer.answer("q", top_k=8, source_type="issue")
    assert answerer._retriever.calls == [("q", 8, "issue")]


def test_no_answer_reply_is_not_answered_and_cites_nothing():
    answer = make_answerer(RESULTS, FakeLLM(NO_ANSWER)).answer("q")
    assert not answer.answered
    assert answer.cited_sources == []


def test_uncited_answer_is_not_counted_as_answered():
    assert not make_answerer(RESULTS, FakeLLM("Pods restart a lot.")).answer("q").answered


def test_nothing_retrieved_skips_the_llm():
    llm = FakeLLM("should not be used")
    answer = make_answerer([], llm).answer("q")
    assert llm.prompts == []
    assert answer.text == NO_ANSWER
    assert not answer.answered
    assert answer.timing.generation_seconds == 0.0
    assert answer.usage.estimated_cost_usd == 0.0


def test_logs_tokens_cost_and_timing(caplog):
    caplog.set_level(logging.INFO, logger="repo_recall.generation.answerer")
    make_answerer(RESULTS, FakeLLM("x [1]")).answer("q")
    [message] = [r.getMessage() for r in caplog.records]
    assert "1000 prompt + 100 completion tokens" in message
    assert "est. $0.001400" in message
    assert "retrieval 0.50s, generation 0.50s" in message


def test_logs_when_server_reports_no_usage(caplog):
    caplog.set_level(logging.INFO, logger="repo_recall.generation.answerer")
    make_answerer(RESULTS, FakeLLM("x [1]", prompt_tokens=None)).answer("q")
    assert "token usage not reported" in caplog.records[0].getMessage()


def test_answer_from_results_skips_retrieval_and_keeps_given_timing():
    retriever = FakeRetriever(RESULTS)
    llm = FakeLLM("x [1]")
    answerer = Answerer(retriever, llm, Settings(_env_file=None), clock=ticking_clock())

    answer = answerer.answer_from_results("q", RESULTS[:1], retrieval_seconds=0.25)

    assert retriever.calls == []
    assert [s.url for s in answer.retrieved] == ["page/a"]
    assert answer.timing.retrieval_seconds == 0.25
    assert answer.timing.generation_seconds == 0.5
