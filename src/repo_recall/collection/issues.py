"""Collect a repo's closed GitHub issues (with useful comments) into a JSONL file.

The collector is resumable: records are appended one per line as soon as an
issue is complete, and a small checkpoint file remembers which listing page
to continue from. Re-running picks up where the last run stopped, skipping
issues already in the output.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TextIO
from urllib.parse import urlencode

from repo_recall.collection.github_client import API_BASE_URL, GitHubClient
from repo_recall.config import Settings

logger = logging.getLogger(__name__)

PAGE_SIZE = 100


@dataclass
class CollectionStats:
    """Counters for one collection run."""

    written: int = 0
    skipped_prs: int = 0
    skipped_existing: int = 0
    skipped_before_since: int = 0


def is_pull_request(item: dict) -> bool:
    """True if an item from the issues endpoint is actually a pull request."""
    return "pull_request" in item


def is_bot(user: dict | None, bot_logins: list[str]) -> bool:
    """True if a GitHub user is an app bot or one of the configured bot accounts."""
    if not user:
        return True  # deleted ("ghost") accounts carry no useful identity
    login = user.get("login", "")
    return user.get("type") == "Bot" or login.endswith("[bot]") or login in bot_logins


def is_command_only(text: str) -> bool:
    """True if every non-empty line is a slash command such as `/close` or `/assign`."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("/") for line in lines)


def is_useful_comment(comment: dict, bot_logins: list[str]) -> bool:
    """True if a comment was written by a human and says more than a bot command."""
    body = (comment.get("body") or "").strip()
    return bool(body) and not is_bot(comment.get("user"), bot_logins) and not is_command_only(body)


def closed_on_or_after(issue: dict, since: date | None) -> bool:
    """True if the issue was closed on or after `since` (always true if `since` is None)."""
    if since is None:
        return True
    closed_at = issue.get("closed_at") or ""
    return closed_at[:10] >= since.isoformat()


def build_comment(comment: dict) -> dict:
    """Reduce a raw API comment to author, role, body and time.

    `author_association` is the author's relation to the repo (OWNER,
    MEMBER, COLLABORATOR, CONTRIBUTOR, NONE, ...); evaluation uses it to
    tell maintainer answers from drive-by replies.
    """
    return {
        "author": comment["user"]["login"],
        "author_association": comment.get("author_association"),
        "body": comment["body"].strip(),
        "created_at": comment["created_at"],
    }


def build_issue_record(issue: dict, comments: list[dict]) -> dict:
    """Build the JSONL record for one issue from its raw API data and cleaned comments."""
    return {
        "number": issue["number"],
        "title": issue["title"],
        "author": (issue.get("user") or {}).get("login"),
        "author_association": issue.get("author_association"),
        "body": (issue.get("body") or "").strip(),
        "comments": comments,
        "labels": [label["name"] for label in issue.get("labels", [])],
        "created_at": issue["created_at"],
        "closed_at": issue["closed_at"],
        "state_reason": issue.get("state_reason"),
        "url": issue["html_url"],
    }


def fetch_useful_comments(
    client: GitHubClient, repo: str, issue: dict, bot_logins: list[str]
) -> list[dict]:
    """Fetch all comments on an issue, keeping only useful human ones."""
    if issue.get("comments", 0) == 0:
        return []
    url = f"repos/{repo}/issues/{issue['number']}/comments"
    comments: list[dict] = []
    for page, _ in client.paginate(url, {"per_page": PAGE_SIZE}):
        comments.extend(build_comment(c) for c in page if is_useful_comment(c, bot_logins))
    return comments


def first_page_url(repo: str, since: date | None) -> str:
    """URL of the first page of closed issues, newest first.

    GitHub's `since` filters on *updated* time; closing an issue updates it,
    so this is a superset of "closed since" and is narrowed client-side.
    """
    params: dict[str, str | int] = {
        "state": "closed",
        "sort": "created",
        "direction": "desc",
        "per_page": PAGE_SIZE,
    }
    if since is not None:
        params["since"] = f"{since.isoformat()}T00:00:00Z"
    return f"{API_BASE_URL}/repos/{repo}/issues?{urlencode(params)}"


def output_paths(settings: Settings) -> tuple[Path, Path]:
    """Return `(records_path, checkpoint_path)` for the configured repo."""
    stem = settings.github_repo.replace("/", "__")
    issues_dir = settings.data_raw_dir / "issues"
    return issues_dir / f"{stem}.jsonl", issues_dir / f"{stem}.state.json"


def drop_partial_last_line(path: Path) -> None:
    """Remove a truncated final line left behind by an interrupted write."""
    if not path.exists():
        return
    content = path.read_bytes()
    if not content or content.endswith(b"\n"):
        return
    last_newline = content.rfind(b"\n")
    path.write_bytes(content[: last_newline + 1])
    logger.warning("Removed a partial trailing record from %s", path)


def load_done_numbers(path: Path) -> set[int]:
    """Return the issue numbers already written to the JSONL output."""
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["number"])
    return done


def load_start_url(checkpoint_path: Path, since: date | None) -> str | None:
    """Return the page URL to resume from, if a matching unfinished checkpoint exists."""
    if not checkpoint_path.exists():
        return None
    state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if state.get("since") != (since.isoformat() if since else None):
        logger.info("Checkpoint was for a different --since; starting from page 1")
        return None
    return state.get("page_url")


def save_checkpoint(checkpoint_path: Path, since: date | None, page_url: str | None) -> None:
    """Record the next listing page still to process (`None` once the listing is done)."""
    state = {"since": since.isoformat() if since else None, "page_url": page_url}
    checkpoint_path.write_text(json.dumps(state), encoding="utf-8")


def collect_issues(
    client: GitHubClient, settings: Settings, limit: int | None = None, since: date | None = None
) -> CollectionStats:
    """Fetch closed issues and append new ones to the JSONL output, resuming if possible.

    Stops after writing `limit` new records (if given). The checkpoint only
    advances past a page once every issue on it has been handled, so an
    interrupted page is revisited and its already-written issues skipped.
    """
    records_path, checkpoint_path = output_paths(settings)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    drop_partial_last_line(records_path)
    done = load_done_numbers(records_path)

    page_url = load_start_url(checkpoint_path, since)
    if page_url:
        logger.info("Resuming from checkpoint (%d issues already collected)", len(done))
    else:
        page_url = first_page_url(settings.github_repo, since)

    stats = CollectionStats()
    with records_path.open("a", encoding="utf-8") as out_file:
        for page, next_url in client.paginate(page_url):
            for item in page:
                _process_item(client, settings, item, since, done, out_file, stats)
                if limit is not None and stats.written >= limit:
                    save_checkpoint(checkpoint_path, since, page_url)
                    return _finish(stats, records_path)
            page_url = next_url
            save_checkpoint(checkpoint_path, since, page_url)

    logger.info("Reached the end of the issue listing")
    return _finish(stats, records_path)


def _process_item(
    client: GitHubClient,
    settings: Settings,
    item: dict,
    since: date | None,
    done: set[int],
    out_file: TextIO,
    stats: CollectionStats,
) -> None:
    """Filter one listing item and, if it's a new in-range issue, write its record."""
    if is_pull_request(item):
        stats.skipped_prs += 1
    elif item["number"] in done:
        stats.skipped_existing += 1
    elif not closed_on_or_after(item, since):
        stats.skipped_before_since += 1
    else:
        comments = fetch_useful_comments(
            client, settings.github_repo, item, settings.github_bot_logins
        )
        record = build_issue_record(item, comments)
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        done.add(item["number"])
        stats.written += 1


def _finish(stats: CollectionStats, records_path: Path) -> CollectionStats:
    """Log a run summary and return the stats."""
    logger.info(
        "Wrote %d new issues (skipped %d PRs, %d already collected, %d closed before --since) to %s",
        stats.written,
        stats.skipped_prs,
        stats.skipped_existing,
        stats.skipped_before_since,
        records_path,
    )
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Collect closed GitHub issues into JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="max new issues to write this run")
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        help="only issues closed on or after this date (YYYY-MM-DD)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: collect issues for the configured repo."""
    args = parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    if not settings.github_token:
        raise SystemExit("GITHUB_TOKEN is not set; add it to .env (unauthenticated limit is 60/hour)")

    client = GitHubClient(settings.github_token)
    collect_issues(client, settings, limit=args.limit, since=args.since)
    logger.info("GitHub API requests remaining this hour: %s", client.last_remaining)


if __name__ == "__main__":
    main()
