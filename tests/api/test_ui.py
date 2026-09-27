from repo_recall.api.ui import EMPTY_QUESTION, LLM_FAILED, format_answer, format_sources, respond
from repo_recall.generation.llm import LLMError

from .conftest import DOC_URL, ISSUE_URL, make_answer, source


def test_sources_are_links_labelled_with_their_citation_numbers():
    text = format_sources(make_answer())
    assert f"- **[1]** [Pod Lifecycle]({DOC_URL}) · doc" in text
    assert f"- **[3]** [Pod restarts]({ISSUE_URL}) · issue" in text


def test_sources_keep_their_numbers_when_cited_out_of_order():
    answer = make_answer(cited=[source(3, ISSUE_URL, "Later"), source(1, DOC_URL, "Earlier")])
    lines = format_sources(answer).splitlines()[2:]
    assert [line[:9] for line in lines] == ["- **[3]**", "- **[1]**"]  # not renumbered 1, 2


def test_brackets_in_titles_cannot_break_the_link():
    answer = make_answer(cited=[source(1, DOC_URL, "kubectl [beta] ] (x)")])
    assert f"[kubectl \\[beta\\] \\] (x)]({DOC_URL})" in format_sources(answer)


def test_no_citations_says_how_many_sources_were_searched():
    assert format_sources(make_answer(cited=[], retrieved=4)) == "_No sources cited (4 were searched)._"


def test_answer_page_has_text_sources_and_footer():
    page = format_answer(make_answer())
    assert page.startswith("Check the liveness probe [1]; see also [3].")
    assert "**Sources**" in page
    assert "test-model · search 0.5s · answer 2.0s · ~$0.0004" in page


def test_respond_strips_the_question_and_rejects_empty_ones():
    asked: list[str] = []

    def ask(question):
        asked.append(question)
        return make_answer(question)

    assert respond(ask, "   ") == EMPTY_QUESTION
    assert respond(ask, None) == EMPTY_QUESTION
    respond(ask, "  why?  ")
    assert asked == ["why?"]


def test_respond_shows_a_friendly_message_when_the_llm_fails():
    def ask(question):
        raise LLMError("boom")

    assert respond(ask, "q") == LLM_FAILED
