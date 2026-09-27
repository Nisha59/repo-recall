"""Collect closed issues matching a GitHub search query, as evaluation candidates.

The main collector walks the newest closed issues, which are mostly bug
reports. Evaluation needs questions, so this one searches for them instead
(for Kubernetes, `label:kind/support`) and writes records in the same format
to a separate file, so they never enter the search index by accident.

GitHub search returns at most 1,000 results per query, so the query is run
over one-year `created:` windows, newest first. Re-runs skip issue numbers
already in the output.
"""

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

from repo_recall.collection.github_client import API_BASE_URL, GitHubClient
from repo_recall.collection.issues import (
    PAGE_SIZE,
    build_issue_record,
    drop_partial_last_line,
    fetch_useful_comments,
    load_done_numbers,
)
from repo_recall.config import Settings

logger = logging.getLogger(__name__)

SEARCH_RESULT_CAP = 1000
WINDOW_DAYS = 365
# Kubernetes' first public commit; older windows can't contain issues.
DEFAULT_OLDEST = date(2014, 6, 1)


@dataclass
class SearchStats:
    """Counters for one search collection run."""

    written: int = 0
    skipped_existing: int = 0
    windows: int = 0


def candidates_path(settings: Settings) -> Path:
    """Where search-collected candidate issues for the configured repo are stored."""
    stem = settings.github_repo.replace("/", "__")
    return settings.data_raw_dir / "eval_candidates" / f"{stem}.jsonl"


def date_windows(newest: date, oldest: date, days: int = WINDOW_DAYS) -> Iterator[tuple[date, date]]:
    """Yield inclusive `(start, end)` windows of `days` days, from `newest` back to `oldest`."""
    end = newest
    while end >= oldest:
        start = max(end - timedelta(days=days - 1), oldest)
        yield start, end
        end = start - timedelta(days=1)


def search_url(repo: str, query: str, start: date, end: date) -> str:
    """First results page for closed issues matching `query`, created in [start, end]."""
    q = f"repo:{repo} is:issue is:closed {query} created:{start.isoformat()}..{end.isoformat()}"
    params = {"q": q, "sort": "created", "order": "desc", "per_page": PAGE_SIZE}
    return f"{API_BASE_URL}/search/issues?{urlencode(params)}"


def collect_search_candidates(
    client: GitHubClient,
    settings: Settings,
    limit: int | None = None,
    newest: date | None = None,
    oldest: date = DEFAULT_OLDEST,
) -> SearchStats:
    """Append new issues matching `settings.eval_candidate_query` to the candidates file.

    Stops after writing `limit` new records (if given).
    """
    path = candidates_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_partial_last_line(path)
    done = load_done_numbers(path)
    stats = SearchStats()

    with path.open("a", encoding="utf-8") as out_file:
        for start, end in date_windows(newest or date.today(), oldest):
            stats.windows += 1
            url = search_url(settings.github_repo, settings.eval_candidate_query, start, end)
            for page, _ in client.paginate(url):
                _warn_if_capped(page, start, end)
                for item in page["items"]:
                    if item["number"] in done:
                        stats.skipped_existing += 1
                        continue
                    comments = fetch_useful_comments(
                        client, settings.github_repo, item, settings.github_bot_logins
                    )
                    record = build_issue_record(item, comments)
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_file.flush()
                    done.add(item["number"])
                    stats.written += 1
                    if limit is not None and stats.written >= limit:
                        return _finish(stats, path)
    return _finish(stats, path)


def _warn_if_capped(page: dict, start: date, end: date) -> None:
    """Log when a window has more matches than search will return."""
    total = page.get("total_count", 0)
    if total > SEARCH_RESULT_CAP:
        logger.warning(
            "%s..%s matches %d issues; search returns only the first %d",
            start,
            end,
            total,
            SEARCH_RESULT_CAP,
        )


def _finish(stats: SearchStats, path: Path) -> SearchStats:
    """Log a run summary and return the stats."""
    logger.info(
        "Wrote %d new candidate issues (skipped %d already collected, %d windows searched) to %s",
        stats.written,
        stats.skipped_existing,
        stats.windows,
        path,
    )
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Collect closed issues matching a search query.")
    parser.add_argument("--limit", type=int, default=None, help="max new issues to write this run")
    parser.add_argument(
        "--oldest",
        type=date.fromisoformat,
        default=DEFAULT_OLDEST,
        help="stop at issues created before this date (YYYY-MM-DD)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: collect evaluation candidates for the configured repo."""
    args = parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    if not settings.github_token:
        raise SystemExit("GITHUB_TOKEN is not set; add it to .env (unauthenticated limit is 60/hour)")

    client = GitHubClient(settings.github_token)
    collect_search_candidates(client, settings, limit=args.limit, oldest=args.oldest)
    logger.info("GitHub API requests remaining this hour: %s", client.last_remaining)


if __name__ == "__main__":
    main()
