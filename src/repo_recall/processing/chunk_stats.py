"""Summary statistics over a set of chunks: counts and token-length distribution."""

from collections import Counter
from dataclasses import dataclass, field

from repo_recall.processing.chunk_model import Chunk

BUCKET_WIDTH = 100
PERCENTILES = {"p25": 0.25, "median": 0.5, "p75": 0.75, "p90": 0.9}


@dataclass
class LengthStats:
    """Token-length distribution for one group of chunks."""

    count: int
    mean: float
    min: int
    max: int
    percentiles: dict[str, int]
    histogram: dict[str, int] = field(default_factory=dict)


def percentile(sorted_values: list[int], fraction: float) -> int:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    return sorted_values[round(fraction * (len(sorted_values) - 1))]


def bucket_label(tokens: int, max_tokens: int) -> str:
    """Histogram bucket for a length: 100-token bands up to the limit, then "over"."""
    if tokens > max_tokens:
        return f">{max_tokens}"
    low = min(tokens // BUCKET_WIDTH * BUCKET_WIDTH, max_tokens // BUCKET_WIDTH * BUCKET_WIDTH)
    high = min(low + BUCKET_WIDTH - 1, max_tokens)
    return f"{low}-{high}"


def bucket_labels(max_tokens: int) -> list[str]:
    """All histogram bucket labels in ascending order."""
    labels = [bucket_label(low, max_tokens) for low in range(0, max_tokens + 1, BUCKET_WIDTH)]
    return [*dict.fromkeys(labels), f">{max_tokens}"]


def length_stats(lengths: list[int], max_tokens: int) -> LengthStats:
    """Distribution summary for a non-empty list of token lengths."""
    ordered = sorted(lengths)
    counts = Counter(bucket_label(n, max_tokens) for n in ordered)
    return LengthStats(
        count=len(ordered),
        mean=sum(ordered) / len(ordered),
        min=ordered[0],
        max=ordered[-1],
        percentiles={name: percentile(ordered, f) for name, f in PERCENTILES.items()},
        histogram={label: counts.get(label, 0) for label in bucket_labels(max_tokens)},
    )


def compute_stats(chunks: list[Chunk], max_tokens: int) -> dict[str, LengthStats]:
    """Length stats per source type, plus an "all" group."""
    groups: dict[str, list[int]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.source_type, []).append(chunk.token_count)
    stats = {name: length_stats(lengths, max_tokens) for name, lengths in sorted(groups.items())}
    if chunks:
        stats["all"] = length_stats([c.token_count for c in chunks], max_tokens)
    return stats


def format_stats(stats: dict[str, LengthStats]) -> str:
    """Render stats as a plain-text report for the terminal."""
    lines = ["Chunks per source type:"]
    lines += [f"  {name:<6} {s.count:>7,}" for name, s in stats.items()]
    lines.append("")
    lines.append("Token length (embedding-model tokens):")
    header = f"  {'':<6} {'mean':>6} {'min':>5} " + " ".join(f"{p:>6}" for p in PERCENTILES)
    lines.append(header + f" {'max':>5}")
    for name, s in stats.items():
        pct = " ".join(f"{s.percentiles[p]:>6}" for p in PERCENTILES)
        lines.append(f"  {name:<6} {s.mean:>6.0f} {s.min:>5} {pct} {s.max:>5}")
    lines.append("")
    lines.append("Length histogram (chunks per token band):")
    names = list(stats)
    lines.append(f"  {'band':<9}" + "".join(f"{n:>8}" for n in names))
    for label in next(iter(stats.values())).histogram if stats else []:
        lines.append(f"  {label:<9}" + "".join(f"{stats[n].histogram[label]:>8,}" for n in names))
    return "\n".join(lines)
