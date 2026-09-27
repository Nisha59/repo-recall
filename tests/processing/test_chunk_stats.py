from repo_recall.processing.chunk_model import Chunk
from repo_recall.processing.chunk_stats import bucket_label, compute_stats, format_stats


def make_chunk(source_type: str, tokens: int) -> Chunk:
    return Chunk(
        chunk_id=f"{source_type}:{tokens}",
        source_type=source_type,
        title="t",
        url="u",
        date=None,
        text="x",
        token_count=tokens,
    )


def test_bucket_label_caps_last_band_at_max_and_flags_overflow():
    assert bucket_label(50, 480) == "0-99"
    assert bucket_label(450, 480) == "400-480"
    assert bucket_label(480, 480) == "400-480"
    assert bucket_label(481, 480) == ">480"


def test_compute_stats_counts_per_source_type_and_percentiles():
    chunks = [make_chunk("doc", n) for n in (100, 200, 300, 400, 500)] + [make_chunk("issue", 50)]
    stats = compute_stats(chunks, max_tokens=480)
    assert stats["doc"].count == 5
    assert stats["issue"].count == 1
    assert stats["all"].count == 6
    assert stats["doc"].min == 100 and stats["doc"].max == 500
    assert stats["doc"].percentiles["median"] == 300
    assert stats["doc"].mean == 300
    assert stats["doc"].histogram[">480"] == 1
    assert stats["doc"].histogram["100-199"] == 1


def test_format_stats_mentions_every_group():
    report = format_stats(compute_stats([make_chunk("doc", 10), make_chunk("issue", 20)], 480))
    assert "doc" in report and "issue" in report and "all" in report
