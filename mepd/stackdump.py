"""Where a long run's processes are, without a profiler that has to attach to them: with $MEPD_STACK_DUMP set
to a folder, `start()` makes this process write every thread's Python stack to <folder>/<pid>.txt every
$MEPD_STACK_DUMP_SECONDS (default 15) seconds, each dump headed by a "== <unix time>" line. Nothing to install,
no ptrace. A thread of mepd's own rather than faulthandler's watchdog, which a forked child cannot re-arm (it
waits for the parent's watchdog thread, which the child does not have)."""
from __future__ import annotations

import os

_STARTED: set = set()


def start() -> None:
    folder = os.environ.get("MEPD_STACK_DUMP", "").strip()
    if not folder or os.getpid() in _STARTED:
        return
    import faulthandler
    import threading
    import time

    try:
        fh = open(os.path.join(folder, f"{os.getpid()}.txt"), "a")
    except OSError:
        return
    _STARTED.add(os.getpid())
    every = float(os.environ.get("MEPD_STACK_DUMP_SECONDS", "15"))
    pid = os.getpid()

    def loop():
        while os.getpid() == pid:
            time.sleep(every)
            try:
                fh.write(f"== {time.time():.0f}\n")
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
                fh.flush()
            except Exception:
                return

    threading.Thread(target=loop, daemon=True, name="mepd-stackdump").start()
