from repo_recall.generation.citations import cited_numbers, cited_sources, is_no_answer
from repo_recall.generation.prompt import NO_ANSWER, group_sources

from .conftest import chunk


def test_parses_single_grouped_and_adjacent_citations_in_first_seen_order():
    text = "Restarts follow the policy [2]. Probes matter [1, 3][2]. Year [2024 is not a list."
    assert cited_numbers(text) == [2, 1, 3]


def test_unknown_numbers_are_dropped():
    sources = group_sources([chunk("a", "page/a"), chunk("b", "page/b")])
    cited = cited_sources("See [2] and [9].", sources)
    assert [s.url for s in cited] == ["page/b"]


def test_no_answer_reply_is_recognized_despite_small_formatting_changes():
    curly = NO_ANSWER.replace("'", "’").rstrip(".")
    assert is_no_answer(NO_ANSWER)
    assert is_no_answer(f'  "{curly}"  ')
    assert not is_no_answer(f"Restarts follow the policy [1]. {NO_ANSWER}")
