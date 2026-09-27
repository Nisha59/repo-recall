import pytest

from repo_recall.evaluation.judge import (
    Judge,
    load_judge_prompt,
    parse_scores,
    render_judge_prompt,
)
from repo_recall.generation.llm import Completion, LLMError
from repo_recall.generation.models import Answer, Timing, TokenUsage
from repo_recall.generation.prompt import group_sources

from ..generation.conftest import chunk

EXAMPLE = {
    "id": "o/r#1",
    "question": "Why is my $HOME pod stuck?",
    "reference_answer": "The CNI plugin isn't ready; check flannel.",
}


def make_answer(text: str = "Check flannel [1].") -> Answer:
    sources = group_sources([chunk("a1", "https://kubernetes.io/docs/a/")])
    return Answer(
        question=EXAMPLE["question"],
        text=text,
        answered=True,
        cited_sources=sources,
        retrieved=sources,
        model="gen",
        usage=TokenUsage(100, 10, 0.0),
        timing=Timing(0.1, 1.0),
    )


class FakeJudgeLLM:
    model = "judge-model"

    def __init__(self, reply: str | Exception):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> Completion:
        self.prompts.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return Completion(self.reply, self.model, 2_000, 300)


def make_judge(reply: str | Exception) -> tuple[Judge, FakeJudgeLLM]:
    llm = FakeJudgeLLM(reply)
    return Judge(llm, load_judge_prompt(), 0.25, 2.0), llm


@pytest.mark.parametrize(
    "reply",
    [
        '{"correctness": 4, "faithfulness": 5, "reasoning": "ok"}',
        'Here is my grade:\n```json\n{"correctness": 4, "faithfulness": 5, "reasoning": "ok"}\n```',
        '{"correctness": "4", "faithfulness": 5.0, "reasoning": "ok"}',
    ],
)
def test_parse_scores_accepts_bare_wrapped_and_stringy_json(reply):
    assert parse_scores(reply) == (4, 5, "ok")


@pytest.mark.parametrize(
    "reply",
    ["no json here", '{"correctness": 7, "faithfulness": 5}', '{"faithfulness": 5}', "{broken"],
)
def test_parse_scores_rejects_unusable_replies(reply):
    assert parse_scores(reply) is None


def test_prompt_file_fills_every_placeholder_and_keeps_dollar_signs_in_values():
    prompt = render_judge_prompt(load_judge_prompt(), EXAMPLE, make_answer())
    assert "$question" not in prompt and "$answer" not in prompt
    assert "Why is my $HOME pod stuck?" in prompt
    assert "The CNI plugin isn't ready" in prompt
    assert "[1] Documentation: Title https://kubernetes.io/docs/a/" in prompt
    assert prompt.rstrip().endswith("Check flannel [1].")


def test_grade_returns_scores_and_prices_the_judge_call():
    judge, llm = make_judge('{"correctness": 3, "faithfulness": 4, "reasoning": "partial"}')
    result = judge.grade(EXAMPLE, make_answer())
    assert (result.correctness, result.faithfulness, result.reasoning) == (3, 4, "partial")
    assert result.usage.estimated_cost_usd == pytest.approx((2_000 * 0.25 + 300 * 2.0) / 1e6)
    assert len(llm.prompts) == 1


def test_unusable_reply_or_llm_error_gives_a_failed_result_not_an_exception():
    bad, _ = make_judge("I think it's fine")
    broken, _ = make_judge(LLMError("HTTP 500"))
    assert bad.grade(EXAMPLE, make_answer()).failed
    assert broken.grade(EXAMPLE, make_answer()).failed
