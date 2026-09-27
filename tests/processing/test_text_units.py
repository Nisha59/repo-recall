from repo_recall.processing.text_units import pack_units, split_blocks, split_oversized


def words(n: int, word: str = "w") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def test_split_blocks_splits_on_blank_lines():
    assert split_blocks("one\ntwo\n\nthree\n\n\nfour") == ["one\ntwo", "three", "four"]


def test_split_blocks_keeps_fenced_code_with_blank_lines_whole():
    text = "Intro.\n\n```yaml\na: 1\n\nb: 2\n```\n\nOutro."
    assert split_blocks(text) == ["Intro.", "```yaml\na: 1\n\nb: 2\n```", "Outro."]


def test_split_oversized_leaves_small_text_alone(counter):
    assert split_oversized("a b c", counter, 10) == ["a b c"]


def test_split_oversized_prefers_line_boundaries(counter):
    text = "\n".join([words(4, "a"), words(4, "b"), words(4, "c")])
    pieces = split_oversized(text, counter, 8)
    assert pieces == [f"{words(4, 'a')}\n{words(4, 'b')}", words(4, "c")]


def test_split_oversized_falls_back_to_words(counter):
    pieces = split_oversized(words(25), counter, 10)
    assert [counter.count(p) for p in pieces] == [10, 10, 5]


def test_pack_units_respects_max_tokens(counter):
    units = [words(6, f"u{i}_") for i in range(10)]
    chunks = pack_units(units, counter, max_tokens=20, overlap_tokens=0)
    assert all(counter.count(c) <= 20 for c in chunks)
    assert "\n\n".join(chunks).count("u9_") > 0


def test_pack_units_overlap_repeats_trailing_whole_units(counter):
    units = ["a1 a2 a3 a4", "b1 b2 b3 b4", "c1 c2 c3 c4", "d1 d2 d3 d4"]
    chunks = pack_units(units, counter, max_tokens=8, overlap_tokens=4)
    assert chunks == [
        "a1 a2 a3 a4\n\nb1 b2 b3 b4",
        "b1 b2 b3 b4\n\nc1 c2 c3 c4",
        "c1 c2 c3 c4\n\nd1 d2 d3 d4",
    ]


def test_pack_units_splits_a_unit_larger_than_max(counter):
    chunks = pack_units([words(30)], counter, max_tokens=10, overlap_tokens=0)
    assert len(chunks) == 3
    assert all(counter.count(c) <= 10 for c in chunks)


def test_pack_units_never_emits_an_overlap_only_chunk(counter):
    chunks = pack_units(["a b", "c d e f g h i j"], counter, max_tokens=8, overlap_tokens=4)
    assert chunks == ["a b", "c d e f g h i j"]
