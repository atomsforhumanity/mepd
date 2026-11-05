"""How many CPUs this process may use."""
from __future__ import annotations

import functools
import math
import os


@functools.lru_cache(maxsize=1)
def available_cpus() -> int:
    """The CPUs this process may run on, within its container's CPU limit.

    `os.cpu_count()` is the machine's: in a Kubernetes pod limited to 48 CPUs on a 128-thread node it says
    128, and work sized from it runs ~2.7x oversubscribed (throttled, and with memory for 128 workers)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = None
    try:   # cgroup v2: "max 100000" or "<quota> <period>"
        q, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:   # cgroup v1
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0 and period > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, max(1, math.ceil(quota)))
    return max(1, n)
