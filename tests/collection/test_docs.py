from pathlib import Path

from repo_recall.collection.docs import build_doc_record


def test_build_doc_record_includes_snapshot_date(tmp_path: Path):
    page = tmp_path / "concepts" / "pods.md"
    page.parent.mkdir()
    page.write_text("---\ntitle: Pods\n---\nPods are the smallest unit.\n", encoding="utf-8")

    record = build_doc_record(page, tmp_path, "https://kubernetes.io/docs", "2026-09-20T10:00:00Z")

    assert record == {
        "title": "Pods",
        "text": "Pods are the smallest unit.",
        "path": "concepts/pods.md",
        "url": "https://kubernetes.io/docs/concepts/pods/",
        "date": "2026-09-20T10:00:00Z",
    }
