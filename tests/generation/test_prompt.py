from repo_recall.generation.prompt import NO_ANSWER, build_prompt, group_sources

from .conftest import chunk


def test_chunks_sharing_a_url_become_one_numbered_source_in_rank_order():
    sources = group_sources(
        [chunk("a1", "page/a"), chunk("b1", "page/b"), chunk("a2", "page/a")]
    )
    assert [(s.number, s.url) for s in sources] == [(1, "page/a"), (2, "page/b")]
    assert [c.chunk_id for c in sources[0].chunks] == ["a1", "a2"]


def test_issue_titles_include_the_issue_number():
    [source] = group_sources([chunk("i", "issues/42", source_type="issue", number=42)])
    assert source.title == "#42 Title issues/42"


def test_prompt_holds_rules_numbered_sources_and_question():
    sources = group_sources(
        [chunk("a1", "page/a"), chunk("i", "issues/7", source_type="issue", number=7)]
    )
    prompt = build_prompt("why?", sources, project="kubernetes/kubernetes")

    assert "kubernetes/kubernetes" in prompt.system
    assert NO_ANSWER in prompt.system
    assert "only" in prompt.system.lower()
    assert "[1] Documentation: Title page/a\nURL: page/a" in prompt.user
    assert "Section: Concepts > Pods\ntext of a1" in prompt.user
    assert "[2] Closed GitHub issue: #7 Title issues/7" in prompt.user
    assert prompt.user.endswith("Question: why?")


def test_no_chunks_gives_no_sources():
    assert group_sources([]) == []
