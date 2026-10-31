"""Regression tests for the macOS fork hazard that killed `--workers > 1`.

`mepd.cli_common._fork_map` forks its workers, and each worker's first act is
to fork again to launch the engine's subprocess. If Tcl's `pthread_atfork`
handlers are registered in the parent (which `import tkinter` does, and which
`openbabel.pybel` used to pull in transitively), that second fork hits a
`notifierInitLock` owned by a thread that does not exist in the child and the
child is SIGKILLed -- surfacing only as `BrokenProcessPool`, with every pair in
the run lost. `mepd/__init__.py` keeps Tk out of the interpreter to prevent it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run(script: str, **env_extra) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, env=env, timeout=300,
    )


darwin_only = pytest.mark.skipif(
    sys.platform != "darwin", reason="the Tk/atfork hazard is macOS-specific"
)


@darwin_only
def test_importing_the_cli_does_not_load_tcl():
    """The import used to arrive via `openbabel.pybel`; pybel itself must
    still import, just without Tk."""
    proc = _run(
        """
        import sys
        import mepd.cli_channels  # noqa: F401  (pulls in openbabel.pybel)
        from openbabel import pybel

        assert sys.modules.get("_tkinter") is None, "Tcl got loaded"
        assert pybel.tk is None, "pybel kept a live Tk handle"
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_fork_map_children_can_launch_subprocesses():
    """The actual failure: every child died the moment it spawned the engine,
    so `_fork_map` raised `BrokenProcessPool` instead of returning."""
    proc = _run(
        """
        import subprocess
        from mepd.cli_common import _fork_map

        def job(i):
            # stands in for an engine call: a subprocess, launched with a cwd
            # (which is what pushes CPython onto its fork/exec path)
            return subprocess.run(
                ["/bin/echo", str(i)], capture_output=True, cwd="/tmp"
            ).returncode

        assert _fork_map(job, [0, 1, 2, 3], workers=4) == [0, 0, 0, 0]
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


@darwin_only
def test_fork_map_falls_back_to_serial_when_tcl_is_loaded():
    """MEPD_ALLOW_TKINTER re-enables Tk; `_fork_map` must then notice that
    forking is unsafe and run serially rather than lose the run."""
    proc = _run(
        """
        import subprocess, sys
        import mepd.cli_channels  # noqa: F401
        import tkinter  # noqa: F401  (registers Tcl's atfork handlers)
        from mepd.cli_common import _fork_map, _unsafe_to_fork

        assert _unsafe_to_fork() is not None

        def job(i):
            return subprocess.run(
                ["/bin/echo", str(i)], capture_output=True, cwd="/tmp"
            ).returncode

        assert _fork_map(job, [0, 1, 2], workers=3) == [0, 0, 0]
        print("OK")
        """,
        MEPD_ALLOW_TKINTER="1",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "serially" in proc.stdout


def test_fork_map_retries_serially_when_a_worker_dies():
    """A child killed outright surfaces only as `BrokenProcessPool` for the
    whole pool, which used to abort the run. Workers that keep dying end in
    a serial run."""
    proc = _run(
        """
        import os
        from mepd.cli_common import _fork_map

        PARENT = os.getpid()

        def job(i):
            if os.getpid() != PARENT:
                os._exit(1)   # die like a SIGKILLed worker: no exception
            return i

        assert _fork_map(job, [0, 1, 2], workers=3) == [0, 1, 2]
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "died without reporting an error" in proc.stdout + proc.stderr   # a warning: stderr without the CLI


def test_a_worker_that_dies_once_is_retried_in_a_new_pool(tmp_path):
    """One worker killed once (the OOM killer at a peak): the unfinished items
    run again across workers, not serially -- a serial retry of a step's
    searches once took two hours where the pool needed one."""
    proc = _run(
        f"""
        import os
        from mepd.cli_common import _fork_map

        PARENT = os.getpid()
        MARK = {str(tmp_path / "killed")!r}

        def job(i):
            if i == 1 and not os.path.exists(MARK):
                open(MARK, "w").close()
                os._exit(1)
            return i, os.getpid() != PARENT

        assert _fork_map(job, list(range(4)), workers=2) == [(i, True) for i in range(4)]
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "in a new pool" in proc.stdout + proc.stderr


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="workers die with their parent on Linux only")
def test_a_broken_pool_leaves_no_orphans_and_the_process_exits(tmp_path):
    """One worker killed (the OOM killer) while the others run pools of their
    own, in a process with a spawn-context pool (whose resource tracker Python
    waits for at exit): the others' workers used to be orphaned for good, and
    the process hung after its last line until something killed it."""
    import time

    script = f"""
        import os, signal, time, multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from mepd.cli_common import _fork_map

        MARK = {str(tmp_path / "killed")!r}

        def inner(x):
            time.sleep(1)
            return x

        def outer(i):
            if i == 1 and not os.path.exists(MARK):
                open(MARK, "w").close()
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGKILL)
            return sum(_fork_map(inner, list(range(4)), 2))

        if __name__ == "__main__":
            spawned = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
            assert list(spawned.map(abs, [-1])) == [1]
            assert _fork_map(outer, list(range(3)), 3) == [6, 6, 6]
            spawned.shutdown()
            print("OK", flush=True)
        """
    fp = tmp_path / "broken_pool.py"
    fp.write_text(textwrap.dedent(script))
    try:
        proc = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=60)
    finally:
        time.sleep(2)
        orphans = subprocess.run(["pgrep", "-f", str(fp)], capture_output=True, text=True).stdout.split()
        for pid in orphans:
            subprocess.run(["kill", "-9", pid])
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr
    assert orphans == [], "workers outlived the broken pool"


@darwin_only
def test_fork_probe_says_yes_on_a_guarded_process():
    """The probe is consulted on every parallel stage, so a false alarm would
    silently drop the run to serial."""
    proc = _run(
        """
        import mepd.cli_common as cc
        assert cc._fork_probe_ok() is True, "probe cried wolf on a clean process"
        assert cc._unsafe_to_fork() is None
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


@darwin_only
def test_fork_probe_detects_a_parent_whose_children_would_be_killed():
    """The probe is what catches hazards the `sys.modules` check cannot see,
    so it has to stand on its own against a parent that really does kill its
    children -- here, one with Tk let back in. (This deliberately kills one
    child, so macOS writes one crash report.)"""
    proc = _run(
        """
        import sys
        import mepd.cli_common as cc

        # MEPD_ALLOW_TKINTER let pybel pull Tcl in on the way here.
        assert sys.modules.get("_tkinter") is not None
        assert cc._fork_probe_ok() is False, "the probe missed a lethal parent"
        print("OK")
        """,
        MEPD_ALLOW_TKINTER="1",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_a_worker_that_runs_its_own_fork_map_still_runs_the_next_item():
    """A worker taking several items (fewer workers than items) whose item runs
    a `_fork_map` of its own (the explorer's channel steps: a step's searches)
    lost the outer map's function when the inner one ended: KeyError on its
    next item, and the whole expansion failed."""
    proc = _run(
        """
        from mepd.cli_common import _fork_map

        def inner(x):
            return x * 10

        def outer(i):
            return sum(_fork_map(inner, [i, i + 1], workers=2))

        assert _fork_map(outer, list(range(6)), workers=2) == [10 * (2 * i + 1) for i in range(6)]
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_a_search_share_lets_a_step_run_more_searches_than_its_workers(tmp_path):
    """Channel steps run their searches in forked processes of their own, `workers` each. With a share of the
    run's lanes, a step left alone (the others finished) uses the lanes they freed instead of its own few."""
    proc = _run(
        f"""
        import multiprocessing, os, time
        from pathlib import Path
        from mepd import cli_common
        from mepd.cli_common import PairFollowUp, _fork_pairs_with_follow_up

        OUT = Path({str(tmp_path)!r})
        searches = multiprocessing.get_context("fork").Value("i", 0)
        cli_common._SEARCH_SHARE = (4, searches)

        def one(pair):
            (OUT / f"{{pair[0]}}.start").write_text(str(time.time()))
            time.sleep(1.0)
            (OUT / f"{{pair[0]}}.end").write_text(str(time.time()))

        follow = PairFollowUp(tasks=lambda d: [], run=lambda t: None)
        _fork_pairs_with_follow_up(one, [(k, k + 1) for k in range(8)], OUT, 1, follow)
        spans = [(float((OUT / f"{{k}}.start").read_text()), float((OUT / f"{{k}}.end").read_text())) for k in range(8)]
        most = max(sum(a <= t < b for a, b in spans) for t, _ in spans)
        assert 3 <= most <= 4, most
        assert searches.value == 0
        print("OK", most)
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
