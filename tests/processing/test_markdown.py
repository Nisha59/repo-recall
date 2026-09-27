from pathlib import PurePosixPath

from repo_recall.processing.markdown import (
    clean_body,
    extract_title,
    parse_front_matter,
    path_to_url,
    strip_shortcodes,
)


def test_parse_front_matter_extracts_title_and_body():
    raw = "---\ntitle: Pods\nweight: 10\n---\nPods are the smallest unit.\n"
    front_matter, body = parse_front_matter(raw)
    assert front_matter == {"title": "Pods", "weight": 10}
    assert body == "Pods are the smallest unit.\n"


def test_parse_front_matter_missing_block_returns_original_text():
    raw = "Just a plain page with no front matter.\n"
    front_matter, body = parse_front_matter(raw)
    assert front_matter == {}
    assert body == raw


def test_parse_front_matter_invalid_yaml_returns_original_text():
    raw = "---\ntitle: [unclosed\n---\nBody text.\n"
    front_matter, body = parse_front_matter(raw)
    assert front_matter == {}
    assert body == raw


def test_strip_shortcodes_keeps_inner_text_of_paired_shortcode():
    text = "Before.\n{{< note >}}\nThis matters.\n{{< /note >}}\nAfter."
    result = strip_shortcodes(text)
    assert "This matters." in result
    assert "{{<" not in result


def test_strip_shortcodes_keeps_text_attribute_of_glossary_tooltip():
    text = 'A group of one or more {{< glossary_tooltip text="containers" term_id="container" >}}, sharing storage.'
    result = strip_shortcodes(text)
    assert "{{<" not in result
    assert "A group of one or more containers, sharing storage." == result


def test_strip_shortcodes_removes_self_closing_shortcode_without_text():
    text = "See the {{< some_shortcode term_id=\"pod\" >}} concept."
    result = strip_shortcodes(text)
    assert "{{<" not in result
    assert "See the" in result and "concept." in result


def test_clean_body_collapses_excess_blank_lines():
    text = "Line one.\n\n\n\nLine two.   \n"
    result = clean_body(text)
    assert result == "Line one.\n\nLine two."


def test_clean_body_removes_html_comments():
    text = "<!-- overview -->\nIntro text.\n<!-- body -->\nMore text."
    result = clean_body(text)
    assert "<!--" not in result
    assert "Intro text." in result and "More text." in result


def test_path_to_url_for_index_page():
    path = PurePosixPath("concepts/workloads/pods/_index.md")
    assert path_to_url(path, "https://kubernetes.io/docs") == "https://kubernetes.io/docs/concepts/workloads/pods/"


def test_path_to_url_for_regular_page():
    path = PurePosixPath("concepts/workloads/pods/pod-lifecycle.md")
    url = path_to_url(path, "https://kubernetes.io/docs")
    assert url == "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/"


def test_extract_title_prefers_front_matter():
    assert extract_title({"title": "Pods"}, PurePosixPath("pods.md")) == "Pods"


def test_extract_title_falls_back_to_filename():
    title = extract_title({}, PurePosixPath("pod-lifecycle.md"))
    assert title == "Pod Lifecycle"
