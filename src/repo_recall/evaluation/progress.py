"""Crash-safe, per-question progress log for evaluation runs.

Each finished question is appended as one JSON line to
`<timestamp>.progress.jsonl` and flushed to disk straight away, so a run
killed mid-mode keeps every question that had already finished. A line cut
off by a crash is skipped on load; everything before it stays readable.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def progress_path(results_dir: Path, timestamp: str) -> Path:
    """Where the progress log for the run `timestamp` lives."""
    return results_dir / f"{timestamp}.progress.jsonl"


class ProgressLog:
    """Append-only JSONL file of finished question records, keyed by mode and question id."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._end_partial_line()

    def _end_partial_line(self) -> None:
        """Terminate a line left half-written by a crash, so new lines don't join onto it."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")

    def append(self, record: dict) -> None:
        """Write one finished question record and force it to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load(self) -> dict[str, dict[str, dict]]:
        """Records logged so far as {mode: {question id: record}}; later lines win."""
        if not self.path.exists():
            return {}
        done: dict[str, dict[str, dict]] = {}
        with self.path.open(encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping unreadable line %d of %s", number, self.path.name)
                    continue
                done.setdefault(record["mode"], {})[record["id"]] = record
        return done
