"""Finding which numbered sources an answer cites."""

import logging
import re

from repo_recall.generation.models import Source
from repo_recall.generation.prompt import NO_ANSWER

logger = logging.getLogger(__name__)

# [1], [1, 3] and [1][3] all count; the last is just two matches.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(text: str) -> list[int]:
    """Every source number cited in `text`, in order of first citation."""
    numbers: dict[int, None] = {}
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            numbers.setdefault(int(part), None)
    return list(numbers)


def cited_sources(text: str, sources: list[Source]) -> list[Source]:
    """The sources `text` cites, in order of first citation.

    Numbers that don't match a source are logged and dropped, since the
    model has invented them.
    """
    by_number = {source.number: source for source in sources}
    cited = []
    for number in cited_numbers(text):
        if number in by_number:
            cited.append(by_number[number])
        else:
            logger.warning("Answer cites [%d], but only %d sources were given", number, len(sources))
    return cited


def is_no_answer(text: str) -> bool:
    """True if the model gave the "sources don't contain the answer" reply."""
    return _normalize(text) == _normalize(NO_ANSWER)


def _normalize(text: str) -> str:
    """Lowercase, straight apostrophes, no surrounding quotes/space/final period."""
    return text.strip().strip("\"'").strip().rstrip(".").replace("’", "'").lower()
