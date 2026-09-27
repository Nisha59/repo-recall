"""LLM-as-judge: score a generated answer for correctness and faithfulness (1-5).

The grading instructions live in `prompts/judge.md` so they can be read and
changed without touching code. The judge sees the same numbered sources the
answering model saw, so faithfulness is checked against what it was given.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

from repo_recall.config import Settings
from repo_recall.generation.cost import estimate_usage
from repo_recall.generation.llm import LLMClient, LLMError, make_llm_client
from repo_recall.generation.models import Answer, TokenUsage
from repo_recall.generation.prompt import format_source

logger = logging.getLogger(__name__)

JUDGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"
JUDGE_SYSTEM = "You are a strict, consistent grader. Reply with JSON only."
SCORE_RANGE = range(1, 6)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeResult:
    """Scores for one answer; both None when the judge's reply couldn't be used."""

    correctness: int | None
    faithfulness: int | None
    reasoning: str
    usage: TokenUsage

    @property
    def failed(self) -> bool:
        """True if no scores could be extracted."""
        return self.correctness is None


def load_judge_prompt(path: Path = JUDGE_PROMPT_PATH) -> Template:
    """The judge prompt as a template with $question, $reference_answer, $sources, $answer."""
    return Template(path.read_text(encoding="utf-8"))


def render_judge_prompt(template: Template, example: dict, answer: Answer) -> str:
    """Fill the judge prompt for one eval example and the generated answer."""
    sources = "\n\n".join(format_source(source) for source in answer.retrieved) or "(none)"
    return template.substitute(
        question=example["question"],
        reference_answer=example["reference_answer"],
        sources=sources,
        answer=answer.text,
    )


def parse_scores(reply: str) -> tuple[int, int, str] | None:
    """`(correctness, faithfulness, reasoning)` from the judge's reply, or None if unusable.

    Accepts the JSON object on its own or wrapped in prose / a code fence.
    """
    match = _JSON_OBJECT_RE.search(reply)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        correctness, faithfulness = int(data["correctness"]), int(data["faithfulness"])
    except (ValueError, KeyError, TypeError):
        return None
    if correctness not in SCORE_RANGE or faithfulness not in SCORE_RANGE:
        return None
    return correctness, faithfulness, str(data.get("reasoning", ""))


class Judge:
    """Grades answers with an LLM, pricing its own calls separately from the system's."""

    def __init__(
        self,
        llm: LLMClient,
        template: Template,
        input_cost_per_mtok: float,
        output_cost_per_mtok: float,
    ) -> None:
        self.model = llm.model
        self._llm = llm
        self._template = template
        self._prices = (input_cost_per_mtok, output_cost_per_mtok)

    def grade(self, example: dict, answer: Answer) -> JudgeResult:
        """Score one answer. LLM failures and bad replies give a failed result, not an error."""
        prompt = render_judge_prompt(self._template, example, answer)
        try:
            completion = self._llm.complete(JUDGE_SYSTEM, prompt)
        except LLMError as exc:
            logger.warning("Judge call failed for %s: %s", example["id"], exc)
            return JudgeResult(None, None, f"judge call failed: {exc}", TokenUsage(0, 0, 0.0))
        usage = estimate_usage(completion.prompt_tokens, completion.completion_tokens, *self._prices)
        scores = parse_scores(completion.text)
        if scores is None:
            logger.warning("Unusable judge reply for %s: %r", example["id"], completion.text[:200])
            return JudgeResult(None, None, f"unparseable reply: {completion.text[:300]}", usage)
        return JudgeResult(*scores, usage)


def make_judge(settings: Settings) -> Judge:
    """The configured judge: same provider and key as generation, its own model and prices."""
    judge_settings = settings.model_copy(
        update={
            "llm_model": settings.eval_judge_model,
            "llm_max_tokens": settings.eval_judge_max_tokens,
        }
    )
    return Judge(
        make_llm_client(judge_settings),
        load_judge_prompt(),
        settings.eval_judge_input_cost_per_mtok,
        settings.eval_judge_output_cost_per_mtok,
    )
