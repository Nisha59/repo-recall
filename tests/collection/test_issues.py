from datetime import date
from pathlib import Path

from repo_recall.collection.issues import (
    build_comment,
    build_issue_record,
    closed_on_or_after,
    drop_partial_last_line,
    is_bot,
    is_command_only,
    is_pull_request,
    is_useful_comment,
    load_done_numbers,
    load_start_url,
    save_checkpoint,
)

BOTS = ["k8s-ci-robot", "k8s-triage-robot"]


def make_comment(login: str, body: str, user_type: str = "User") -> dict:
    return {"user": {"login": login, "type": user_type}, "body": body, "created_at": "2024-01-01T00:00:00Z"}


def test_is_pull_request() -> None:
    assert is_pull_request({"number": 1, "pull_request": {"url": "x"}})
    assert not is_pull_request({"number": 1})


def test_is_bot_detects_app_bots_suffix_and_configured_logins() -> None:
    assert is_bot({"login": "dependabot", "type": "Bot"}, BOTS)
    assert is_bot({"login": "renovate[bot]", "type": "User"}, BOTS)
    assert is_bot({"login": "k8s-ci-robot", "type": "User"}, BOTS)
    assert is_bot(None, BOTS)
    assert not is_bot({"login": "alice", "type": "User"}, BOTS)


def test_is_command_only() -> None:
    assert is_command_only("/close")
    assert is_command_only("/lifecycle stale\n\n/assign @bob")
    assert not is_command_only("/close\nFixed in #123, see the release notes.")
    assert not is_command_only("")


def test_is_useful_comment_filters_bots_commands_and_empty() -> None:
    assert is_useful_comment(make_comment("alice", "Try restarting kubelet."), BOTS)
    assert not is_useful_comment(make_comment("k8s-triage-robot", "This issue is stale."), BOTS)
    assert not is_useful_comment(make_comment("alice", "/remove-lifecycle stale"), BOTS)
    assert not is_useful_comment(make_comment("alice", "   "), BOTS)


def test_build_comment_keeps_author_association() -> None:
    raw = {**make_comment("alice", " Use a readiness probe. "), "author_association": "MEMBER"}
    assert build_comment(raw) == {
        "author": "alice",
        "author_association": "MEMBER",
        "body": "Use a readiness probe.",
        "created_at": "2024-01-01T00:00:00Z",
    }


def test_closed_on_or_after() -> None:
    issue = {"closed_at": "2024-03-15T10:00:00Z"}
    assert closed_on_or_after(issue, None)
    assert closed_on_or_after(issue, date(2024, 3, 15))
    assert not closed_on_or_after(issue, date(2024, 3, 16))


def test_build_issue_record() -> None:
    issue = {
        "number": 42,
        "title": "Pod stuck terminating",
        "body": None,
        "labels": [{"name": "kind/bug"}, {"name": "sig/node"}],
        "created_at": "2024-01-01T00:00:00Z",
        "closed_at": "2024-02-01T00:00:00Z",
        "state_reason": "completed",
        "html_url": "https://github.com/kubernetes/kubernetes/issues/42",
        "user": {"login": "bob"},
        "author_association": "NONE",
    }
    comments = [{"author": "alice", "body": "hi", "created_at": "2024-01-02T00:00:00Z"}]

    record = build_issue_record(issue, comments)

    assert record == {
        "number": 42,
        "title": "Pod stuck terminating",
        "author": "bob",
        "author_association": "NONE",
        "body": "",
        "comments": comments,
        "labels": ["kind/bug", "sig/node"],
        "created_at": "2024-01-01T00:00:00Z",
        "closed_at": "2024-02-01T00:00:00Z",
        "state_reason": "completed",
        "url": "https://github.com/kubernetes/kubernetes/issues/42",
    }


def test_partial_last_line_is_dropped_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    path.write_text('{"number": 1}\n{"number": 2}\n{"numb', encoding="utf-8")

    drop_partial_last_line(path)

    assert path.read_text(encoding="utf-8") == '{"number": 1}\n{"number": 2}\n'
    assert load_done_numbers(path) == {1, 2}


def test_load_done_numbers_missing_file(tmp_path: Path) -> None:
    assert load_done_numbers(tmp_path / "missing.jsonl") == set()


def test_checkpoint_round_trip_and_since_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    save_checkpoint(path, date(2024, 1, 1), "https://api.github.com/page2")

    assert load_start_url(path, date(2024, 1, 1)) == "https://api.github.com/page2"
    assert load_start_url(path, None) is None
    assert load_start_url(tmp_path / "missing.json", None) is None
