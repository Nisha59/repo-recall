"""The Gradio page: type a question, read the answer with clickable source links."""

import logging
from collections.abc import Callable

import gradio as gr

from repo_recall.generation.llm import LLMError
from repo_recall.generation.models import Answer

logger = logging.getLogger(__name__)

EMPTY_QUESTION = "Please type a question first."
LLM_FAILED = "**Couldn't get an answer:** the language model request failed. Please try again in a moment."


def _link_text(title: str) -> str:
    """A title that can't break out of Markdown link text."""
    return title.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def format_sources(answer: Answer) -> str:
    """A Markdown list of cited sources as links, labelled with the [n] markers used in the answer.

    Bullets, not a numbered list: sources are listed in citation order
    (e.g. [3] before [1]), which Markdown would renumber 1, 2, ...
    """
    if not answer.cited_sources:
        return f"_No sources cited ({len(answer.retrieved)} were searched)._"
    lines = ["**Sources**", ""]
    lines += [
        f"- **[{s.number}]** [{_link_text(s.title)}]({s.url}) · {s.source_type}"
        for s in answer.cited_sources
    ]
    return "\n".join(lines)


def format_footer(answer: Answer) -> str:
    """One line with the model, timings and estimated cost."""
    timing, usage = answer.timing, answer.usage
    cost = "" if usage.estimated_cost_usd is None else f" · ~${usage.estimated_cost_usd:.4f}"
    return (
        f"<sub>{answer.model} · search {timing.retrieval_seconds:.1f}s · "
        f"answer {timing.generation_seconds:.1f}s{cost}</sub>"
    )


def format_answer(answer: Answer) -> str:
    """The whole result as Markdown: answer text, source links, footer."""
    return "\n\n".join([answer.text, format_sources(answer), format_footer(answer)])


def respond(ask: Callable[[str], Answer], question: str) -> str:
    """Answer one question for the page, turning failures into a readable message."""
    question = (question or "").strip()
    if not question:
        return EMPTY_QUESTION
    try:
        return format_answer(ask(question))
    except LLMError:
        logger.exception("LLM request failed for a UI question")
        return LLM_FAILED


def build_ui(ask: Callable[[str], Answer], project: str) -> gr.Blocks:
    """The question page; `ask` answers a question (the app passes its Answerer)."""
    with gr.Blocks(title="RepoRecall", analytics_enabled=False) as ui:
        gr.Markdown(
            f"# RepoRecall\nAsk a question about **{project}**. Answers come from its docs "
            "and closed GitHub issues, with links to the sources they cite."
        )
        question = gr.Textbox(
            label="Question",
            lines=3,
            placeholder="e.g. Why does my pod keep restarting with CrashLoopBackOff?",
        )
        ask_button = gr.Button("Ask", variant="primary")
        answer = gr.Markdown()
        ask_button.click(lambda q: respond(ask, q), inputs=question, outputs=answer)
    return ui
