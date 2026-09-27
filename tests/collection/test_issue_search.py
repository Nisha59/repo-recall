import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from repo_recall.collection.issue_search import (
    candidates_path,
    collect_search_candidates,
    date_windows,
    search_url,
)
from repo_recall.config import Settings


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_raw_dir=tmp_path, eval_candidate_query="label:kind/support")


def search_item(number: int) -> dict:
    return {
        "number": number,
        "title": f"Question {number}",
        "user": {"login": "asker"},
        "author_association": "NONE",
        "body": "How do I ...?",
        "comments": 0,
        "labels": [{"name": "kind/support"}],
        "created_at": "2024-01-01T00:00:00Z",
        "closed_at": "2024-02-01T00:00:00Z",
        "state_reason": "completed",
        "html_url": f"https://github.com/o/r/issues/{number}",
    }


class FakeClient:
    """Returns `pages_by_window[i]` for the i-th search; records URLs requested."""

    def __init__(self, pages_by_window: list[list[dict]]) -> None:
        self.pages_by_window = pages_by_window
        self.urls: list[str] = []

    def paginate(self, url, params=None):
        self.urls.append(url)
        index = len(self.urls) - 1
        pages = self.pages_by_window[index] if index < len(self.pages_by_window) else []
        for page in pages:
            yield {"total_count": len(page), "items": page}, None


def test_date_windows_step_back_a_year_without_gaps_and_stop_at_oldest():
    windows = list(date_windows(date(2024, 6, 30), date(2022, 1, 1), days=365))
    assert windows[0] == (date(2023, 7, 2), date(2024, 6, 30))
    assert windows[1][1] == date(2023, 7, 1)
    assert windows[-1][0] == date(2022, 1, 1)


def test_search_url_scopes_query_to_closed_issues_in_the_window():
    url = search_url("o/r", "label:kind/support", date(2024, 1, 1), date(2024, 12, 31))
    params = parse_qs(urlsplit(url).query)
    assert urlsplit(url).path == "/search/issues"
    assert params["q"] == [
        "repo:o/r is:issue is:closed label:kind/support created:2024-01-01..2024-12-31"
    ]
    assert params["sort"] == ["created"] and params["order"] == ["desc"]


def test_collects_records_across_windows_up_to_limit(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient([[[search_item(3), search_item(2)]], [[search_item(1)]]])

    stats = collect_search_candidates(
        client, settings, limit=3, newest=date(2024, 6, 30), oldest=date(2020, 1, 1)
    )

    lines = candidates_path(settings).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert stats.written == 3
    assert [r["number"] for r in records] == [3, 2, 1]
    assert records[0]["author"] == "asker" and records[0]["url"].endswith("/issues/3")
    assert len(client.urls) == 2


def test_rerun_skips_issues_already_collected(tmp_path):
    settings = make_settings(tmp_path)
    window = [[[search_item(2), search_item(1)]]]
    collect_search_candidates(FakeClient(window), settings, newest=date(2024, 6, 30), oldest=date(2024, 1, 1))

    stats = collect_search_candidates(
        FakeClient(window), settings, newest=date(2024, 6, 30), oldest=date(2024, 1, 1)
    )

    assert (stats.written, stats.skipped_existing) == (0, 2)
    assert len(candidates_path(settings).read_text(encoding="utf-8").splitlines()) == 2
