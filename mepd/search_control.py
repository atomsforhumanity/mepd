"""Stop a path search early and keep what it found ("Stop & analyze").

A run is told to stop by a file appearing at `$MEPD_STOP_FILE` (the web
app sets it per job; from a shell: `MEPD_STOP_FILE=stop mepd channels ...`,
then `touch stop`). From then on, as with `--search-budget`'s deadline:

- no new pair search starts and no search is split further: each becomes a
  leaf with the path it has (status "stopped");
- a path search already running ends at its next step with its current
  path (`SearchStopped`, raised from the per-step progress report while
  `interruptible()` is active, caught by MSMEP);

and whatever comes after the searches -- TS optimization, IRC, channel
classification -- runs on what was found, so stopping early loses nothing
but the searches that had not finished.

A file is used so that every worker process (pairs run in forked workers)
and every step sees the request without any plumbing, and only the job it
was meant for (each job has its own file).
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any, Optional

STOP_ENV = "MEPD_STOP_FILE"

_interruptible = 0


class SearchStopped(Exception):
    """A path search stopped on request; `chain` is its path at that step."""

    def __init__(self, chain: Any = None):
        super().__init__("path search stopped on request")
        self.chain = chain


def stop_file() -> Optional[Path]:
    value = os.environ.get(STOP_ENV)
    return Path(value) if value else None


def stop_requested() -> bool:
    fp = stop_file()
    return fp is not None and fp.exists()


def search_over(path_min_inputs: Any = None) -> Optional[str]:
    """Why no more search work should start: "stopped" (asked to), "time_budget"
    (--search-budget's deadline has passed) or "path_search_budget"
    (--max-path-searches have run); None to go on."""
    if stop_requested():
        return "stopped"
    deadline = getattr(path_min_inputs, "recursive_split_deadline", None)
    if deadline is not None and time.time() > deadline:
        return "time_budget"
    cap = getattr(path_min_inputs, "max_path_searches", None)
    if cap and path_searches(path_min_inputs) >= cap:
        return "path_search_budget"
    return None


def start_path_search_count(path_min_inputs: Any, counter: Path, cap: int) -> None:
    """Count path searches in `counter` (a file, shared by forked workers) and
    stop new work after `cap` of them (`search_over`). A resumed run keeps
    counting from the file."""
    counter = Path(counter)
    if not counter.exists():
        counter.write_text("0\n")
    path_min_inputs.path_search_counter = str(counter)
    path_min_inputs.max_path_searches = int(cap)


def path_searches(path_min_inputs: Any) -> int:
    """How many path searches have started under `start_path_search_count`."""
    fp = getattr(path_min_inputs, "path_search_counter", None)
    if not fp or not Path(fp).exists():
        return 0
    return int(Path(fp).read_text().strip() or 0)


def count_path_search(path_min_inputs: Any) -> None:
    """One more path search (called as each starts); no-op without a counter."""
    import fcntl

    fp = getattr(path_min_inputs, "path_search_counter", None)
    if not fp:
        return
    with open(fp, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        n = int(fh.read().strip() or 0) + 1
        fh.seek(0)
        fh.write(f"{n}\n")
        fh.truncate()


@contextlib.contextmanager
def interruptible():
    """A path search that `check` may end (MSMEP around optimize_chain)."""
    global _interruptible
    _interruptible += 1
    try:
        yield
    finally:
        _interruptible -= 1


def check(chain: Any = None) -> None:
    """At a path-search step: raise SearchStopped (with that step's path)
    if a stop was requested. Elsewhere (TS optimization, IRC) a no-op."""
    if _interruptible and stop_requested():
        raise SearchStopped(chain)
