from repo_recall.processing.chunk_model import ChunkSizes
from repo_recall.processing.doc_chunker import (
    Section,
    build_breadcrumbs,
    chunk_doc,
    merge_tiny_sections,
    parse_heading,
    split_sections,
)


def doc_record(text: str, path: str = "concepts/workloads/pods/_index.md") -> dict:
    return {
        "title": "Pods",
        "text": text,
        "path": path,
        "url": "https://kubernetes.io/docs/concepts/workloads/pods/",
        "date": "2026-09-20T10:00:00+00:00",
    }


def test_parse_heading_strips_hugo_anchor():
    assert parse_heading("## Pod lifecycle {#pod-lifecycle}") == (2, "Pod lifecycle")
    assert parse_heading("not a heading") is None
    assert parse_heading("#hashtag") is None


def test_split_sections_tracks_nesting_and_resets_on_higher_level():
    text = (
        "Intro text.\n\n## A\n\nBody A.\n\n### A1\n\nBody A1.\n\n## B\n\nBody B."
    )
    sections = split_sections(text)
    assert [s.headings for s in sections] == [(), ("A",), ("A", "A1"), ("B",)]
    assert sections[2].body == "Body A1."


def test_split_sections_ignores_hash_lines_inside_code_fences():
    text = "## Install\n\n```bash\n# install the tool\nkubectl apply -f x.yaml\n```"
    sections = split_sections(text)
    assert len(sections) == 1
    assert sections[0].headings == ("Install",)
    assert "# install the tool" in sections[0].body


def test_split_sections_drops_heading_with_no_body():
    sections = split_sections("## Parent\n\n### Child\n\nText.")
    assert sections == [Section(("Parent", "Child"), "Text.")]


def test_build_breadcrumbs_uses_index_titles_of_ancestor_dirs():
    records = [
        {"path": "concepts/_index.md", "title": "Concepts"},
        {"path": "concepts/workloads/_index.md", "title": "Workloads"},
        {"path": "concepts/workloads/pods/_index.md", "title": "Pods"},
        {"path": "concepts/workloads/pods/pod-lifecycle.md", "title": "Pod Lifecycle"},
        {"path": "concepts/no-index-dir/page.md", "title": "Page"},
    ]
    crumbs = build_breadcrumbs(records)
    assert crumbs["concepts/_index.md"] == []
    assert crumbs["concepts/workloads/pods/_index.md"] == ["Concepts", "Workloads"]
    assert crumbs["concepts/workloads/pods/pod-lifecycle.md"] == ["Concepts", "Workloads", "Pods"]
    assert crumbs["concepts/no-index-dir/page.md"] == ["Concepts", "No Index Dir"]


def test_merge_tiny_sections_merges_forward_when_it_fits(counter):
    sections = [
        Section(("A",), "tiny"),
        Section(("B",), "a body that is long enough to stand alone here"),
    ]
    merged = merge_tiny_sections(sections, counter, min_tokens=5, max_tokens=40)
    assert len(merged) == 1
    assert merged[0].headings == ()
    assert merged[0].body.startswith("A\n\ntiny\n\nB\n\n")


def test_merge_tiny_sections_keeps_tiny_section_when_merge_would_overflow(counter):
    big = " ".join(["word"] * 39)
    sections = [Section(("A",), "tiny"), Section(("B",), big)]
    merged = merge_tiny_sections(sections, counter, min_tokens=5, max_tokens=40)
    assert [s.headings for s in merged] == [("A",), ("B",)]


def test_merge_tiny_sections_merges_trailing_tiny_section_backward(counter):
    sections = [
        Section(("A",), "a body that is long enough to stand alone here"),
        Section(("A", "See also"), "link"),
    ]
    merged = merge_tiny_sections(sections, counter, min_tokens=5, max_tokens=40)
    assert len(merged) == 1
    assert merged[0].headings == ("A",)
    assert merged[0].body.endswith("See also\n\nlink")


def test_chunk_doc_prefixes_heading_path_and_sets_metadata(counter, sizes):
    text = "## Pod lifecycle\n\nPods follow a defined lifecycle starting in Pending phase."
    chunks = chunk_doc(doc_record(text), ["Concepts", "Workloads"], sizes, counter)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.heading_path == ["Concepts", "Workloads", "Pods", "Pod lifecycle"]
    assert chunk.text.startswith("Concepts > Workloads > Pods > Pod lifecycle\n\n")
    assert chunk.chunk_id == "doc:concepts/workloads/pods/_index.md#0"
    assert chunk.source_type == "doc"
    assert chunk.title == "Pods"
    assert chunk.url == "https://kubernetes.io/docs/concepts/workloads/pods/"
    assert chunk.date == "2026-09-20T10:00:00+00:00"
    assert chunk.labels == []


def test_chunk_doc_splits_long_section_within_budget(counter):
    sizes = ChunkSizes(max_tokens=30, overlap_tokens=0, min_tokens=5)
    paragraphs = "\n\n".join(" ".join(f"p{i}w{j}" for j in range(8)) for i in range(10))
    chunks = chunk_doc(doc_record(f"## Long\n\n{paragraphs}"), ["Concepts"], sizes, counter)
    assert len(chunks) > 1
    assert all(c.text.startswith("Concepts > Pods > Long\n\n") for c in chunks)
    assert all(c.token_count <= 30 for c in chunks)
    assert [c.chunk_id[-2:] for c in chunks] == [f"#{i}" for i in range(len(chunks))]
    joined = "\n".join(c.text for c in chunks)
    assert all(f"p{i}w0" in joined for i in range(10))
