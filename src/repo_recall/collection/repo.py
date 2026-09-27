"""Cloning/updating a git repository as a local data source."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def clone_or_update_repo(repo_url: str, dest: Path) -> Path:
    """Ensure a shallow clone of `repo_url` exists at `dest`, up to date.

    Clones with `--depth 1` if `dest` doesn't exist yet; otherwise fetches the
    latest commit and resets onto it. (`git pull --ff-only` can't be used: a
    depth-1 commit has no parent, so git always sees the new upstream commit
    as diverging.) `dest` is treated as a read-only mirror. Returns `dest`.
    """
    if (dest / ".git").exists():
        logger.info("Updating existing clone at %s", dest)
        for args in (["fetch", "--depth", "1", "origin", "HEAD"], ["reset", "--hard", "FETCH_HEAD"]):
            subprocess.run(
                ["git", "-C", str(dest), *args],
                check=True,
                capture_output=True,
                text=True,
            )
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning %s into %s (shallow)", repo_url, dest)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )

    return dest


def head_commit_date(repo_dir: Path) -> str:
    """Return the ISO-8601 commit date of the checked-out HEAD in `repo_dir`.

    With a shallow clone there is no per-file history, so this is the best
    available "as of" date for the whole docs snapshot.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "-1", "--format=%cI"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
