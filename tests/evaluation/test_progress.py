from repo_recall.evaluation.progress import ProgressLog, progress_path


def record(mode: str, n: int, **extra) -> dict:
    return {"id": f"o/r#{n}", "mode": mode, **extra}


def test_path_is_next_to_the_results_file(tmp_path):
    assert progress_path(tmp_path, "20260927-120000") == tmp_path / "20260927-120000.progress.jsonl"


def test_missing_log_loads_as_empty(tmp_path):
    assert ProgressLog(tmp_path / "none.progress.jsonl").load() == {}


def test_appended_records_load_by_mode_and_id(tmp_path):
    log = ProgressLog(tmp_path / "run.progress.jsonl")
    log.append(record("vector", 1, answer="a"))
    log.append(record("hybrid", 1, answer="b"))
    log.append(record("vector", 2, answer="c"))

    loaded = ProgressLog(log.path).load()
    assert set(loaded) == {"vector", "hybrid"}
    assert set(loaded["vector"]) == {"o/r#1", "o/r#2"}
    assert loaded["hybrid"]["o/r#1"]["answer"] == "b"


def test_later_lines_win_for_the_same_question(tmp_path):
    log = ProgressLog(tmp_path / "run.progress.jsonl")
    log.append(record("vector", 1, answer="first"))
    log.append(record("vector", 1, answer="second"))
    assert log.load()["vector"]["o/r#1"]["answer"] == "second"


def test_a_line_cut_off_by_a_crash_is_skipped_and_later_appends_survive(tmp_path):
    path = tmp_path / "run.progress.jsonl"
    ProgressLog(path).append(record("vector", 1, answer="kept"))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "o/r#2", "mode": "vec')  # the process died mid-write

    log = ProgressLog(path)  # a resumed run opens the log again
    log.append(record("vector", 3, answer="after"))

    assert set(log.load()["vector"]) == {"o/r#1", "o/r#3"}
