"""Pure text-processing functions for Hugo-flavored Markdown docs.

No file I/O here on purpose — everything takes and returns strings so it's
trivial to unit test.
"""

import re
from pathlib import PurePosixPath

import yaml

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n?", re.DOTALL)

# Paired shortcode: {{< note >}}...{{< /note >}} or {{% tab %}}...{{% /tab %}}
_PAIRED_SHORTCODE_RE = re.compile(
    r"\{\{[%<]\s*(\w+)[^%>]*?[%>]\}\}(.*?)\{\{[%<]\s*/\1\s*[%>]\}\}",
    re.DOTALL,
)

# Self-closing shortcode carrying a text="..." attribute (e.g. glossary_tooltip) —
# that attribute is the word actually meant to be read, so keep it.
_SELF_CLOSING_WITH_TEXT_RE = re.compile(r"""\{\{[%<][^%>]*?\btext=["']([^"']*)["'][^%>]*?[%>]\}\}""")

# Any remaining (self-closing) shortcode tag.
_SELF_CLOSING_SHORTCODE_RE = re.compile(r"\{\{[%<].*?[%>]\}\}", re.DOTALL)

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def parse_front_matter(raw_text: str) -> tuple[dict, str]:
    """Split a leading YAML front matter block off of `raw_text`.

    Returns `(front_matter_dict, body)`. If there is no front matter block,
    or it fails to parse as YAML, returns `({}, raw_text)` unchanged.
    """
    match = _FRONT_MATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text

    body = raw_text[match.end() :]
    try:
        front_matter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, raw_text

    if not isinstance(front_matter, dict):
        return {}, body

    return front_matter, body


def strip_shortcodes(text: str) -> str:
    """Remove Hugo shortcode tags, keeping their meaningful inner text.

    Paired shortcodes (`{{< note >}}...{{< /note >}}`) keep the text between
    the tags. Self-closing shortcodes with a `text="..."` attribute (e.g.
    `glossary_tooltip`) keep that attribute's value, since it's the word the
    page actually displays. Everything else is dropped.
    """
    text = _PAIRED_SHORTCODE_RE.sub(lambda m: m.group(2), text)
    text = _SELF_CLOSING_WITH_TEXT_RE.sub(lambda m: m.group(1), text)
    text = _SELF_CLOSING_SHORTCODE_RE.sub("", text)
    return text


def clean_body(text: str) -> str:
    """Strip shortcodes/HTML comments and normalize whitespace in a doc body."""
    text = strip_shortcodes(text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def path_to_url(relative_path: PurePosixPath, base_url: str) -> str:
    """Map a doc's path (relative to the docs content root) to its public URL.

    `_index.md` files map to their parent directory's URL; other files map
    to a same-named directory URL (matching how Hugo renders these pages).
    """
    if relative_path.name == "_index.md":
        segment = relative_path.parent.as_posix()
    else:
        segment = (relative_path.parent / relative_path.stem).as_posix()

    segment = "" if segment == "." else segment.strip("/")
    base = base_url.rstrip("/")
    return f"{base}/{segment}/" if segment else f"{base}/"


def extract_title(front_matter: dict, fallback_path: PurePosixPath) -> str:
    """Get a page's title from front matter, falling back to its filename."""
    title = front_matter.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    stem = fallback_path.stem
    name = fallback_path.parent.name if stem == "_index" else stem
    return name.replace("-", " ").replace("_", " ").strip().title()
