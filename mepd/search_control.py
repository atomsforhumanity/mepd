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
    """Why no more search work should start: "stopped" (asked to), or
    "time_budget" (--search-budget's deadline has passed); None to go on."""
    if stop_requested():
        return "stopped"
    deadline = getattr(path_min_inputs, "recursive_split_deadline", None)
    if deadline is not None and time.time() > deadline:
        return "time_budget"
    return None


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
