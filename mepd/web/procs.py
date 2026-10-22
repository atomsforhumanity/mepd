"""Helpers for processes the web server spawns."""
import os
import sys


def die_with_parent(parent_pid: int) -> None:
    """Call first thing in a spawned process: exit when the server does. A
    server killed outright (SIGKILL, `fuser -k`) never shuts its children
    down, and they would otherwise linger, holding memory (or, for a busy
    loop such as the interactive reactor's MD, a whole CPU) indefinitely."""
    import signal

    if sys.platform.startswith("linux"):
        try:
            import ctypes

            PR_SET_PDEATHSIG = 1
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception:
            pass
    if os.getppid() != parent_pid:  # the server died before we got here
        os._exit(0)
