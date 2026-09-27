from repo_recall.processing.chunk_model import ChunkSizes
from repo_recall.processing.issue_chunker import (
    chunk_issue,
    clean_issue_body,
    is_low_value_comment,
    score_comment,
    select_comments,
)


def comment(body: str, author: str = "dev") -> dict:
    return {"author": author, "body": body, "created_at": "2026-09-01T00:00:00Z"}


def issue_record(body: str = "The pod crashes on startup.", comments: list | None = None) -> dict:
    return {
        "number": 123,
        "title": "Pod crashloops after upgrade",
        "body": body,
        "comments": comments or [],
        "labels": ["kind/bug", "sig/node"],
        "created_at": "2026-08-01T00:00:00Z",
        "closed_at": "2026-09-02T00:00:00Z",
        "state_reason": "completed",
        "url": "https://github.com/kubernetes/kubernetes/issues/123",
    }


def test_clean_issue_body_strips_template_noise():
    body = (
        "<!-- Please describe the bug -->\n### What happened?\n\nIt broke.\n\n"
        "### Anything else we need to know?\n\n_No response_\n\n"
        "### Kubernetes version\n\n<details>\n\nv1.35\n</details>"
    )
    cleaned = clean_issue_body(body)
    assert "Please describe" not in cleaned
    assert "Anything else" not in cleaned
    assert "_No response_" not in cleaned
    assert "<details>" not in cleaned
    assert "It broke." in cleaned and "v1.35" in cleaned


def test_is_low_value_comment_flags_pings_and_plus_ones():
    assert is_low_value_comment("+1")
    assert is_low_value_comment("@alice @bob PTAL")
    assert is_low_value_comment("FYI @carol")
    assert is_low_value_comment("any update?")
    assert not is_low_value_comment("The kubelet restarts because the cgroup driver mismatches.")


def test_score_comment_rewards_code_links_and_resolution_words():
    plain = score_comment("I also see this happening on my cluster sometimes.")
    fix = score_comment("Fixed by https://github.com/kubernetes/kubernetes/pull/999")
    code = score_comment("Here is the log:\n```\npanic: nil pointer\n```")
    assert fix > plain
    assert code > plain


def test_select_comments_always_keeps_last_and_preserves_order():
    comments = [
        comment("Workaround: set the cgroup driver to systemd, see ```config```", "a"),
        comment("I think this might be related to the new release somehow", "b"),
        comment("+1", "c"),
        comment("Root cause found, fixed in /pull/42 and backported", "d"),
        comment("Closing now since the fix was released in the patch", "e"),
    ]
    selected = select_comments(comments, max_comments=3)
    assert [c["author"] for c in selected] == ["a", "d", "e"]


def test_select_comments_returns_empty_when_nothing_substantive():
    assert select_comments([comment("+1"), comment("@x ping")], max_comments=5) == []


def test_chunk_issue_single_chunk_has_header_and_metadata(counter, sizes):
    record = issue_record(comments=[comment("The fix was to restart the kubelet service.", "e")])
    chunks = chunk_issue(record, "kubernetes/kubernetes", sizes, 5, counter)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text.startswith("Issue #123: Pod crashloops after upgrade\nLabels: kind/bug, sig/node")
    assert "Comment by e:" in chunk.text
    assert chunk.chunk_id == "issue:kubernetes/kubernetes#123:0"
    assert chunk.source_type == "issue"
    assert chunk.labels == ["kind/bug", "sig/node"]
    assert chunk.date == "2026-09-02T00:00:00Z"
    assert chunk.url == "https://github.com/kubernetes/kubernetes/issues/123"
    assert chunk.extra["state_reason"] == "completed"


def test_chunk_issue_long_issue_splits_with_header_in_every_chunk(counter):
    sizes = ChunkSizes(max_tokens=40, overlap_tokens=0, min_tokens=5)
    body = "\n\n".join(" ".join(f"b{i}w{j}" for j in range(10)) for i in range(6))
    comments = [comment(" ".join(f"c{i}w{j}" for j in range(10)), f"u{i}") for i in range(4)]
    chunks = chunk_issue(issue_record(body, comments), "o/r", sizes, 5, counter)
    assert len(chunks) > 1
    assert all(c.text.startswith("Issue #123: Pod crashloops after upgrade") for c in chunks)
    assert all(c.token_count <= 40 for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"issue:o/r#123:{i}" for i in range(len(chunks))]


def test_chunk_issue_with_empty_body_still_yields_title_chunk(counter, sizes):
    chunks = chunk_issue(issue_record(body=""), "o/r", sizes, 5, counter)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Issue #123")
