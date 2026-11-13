"""How crowded the cluster was (with my own tasks) while each arm ran: from the pulled result.json files (start,
end) of every listed run, the number of my tasks running at each moment; per run and configuration: the window,
the median concurrent tasks over its task-seconds, and the share of its task-seconds spent while more than CAP
tasks (default 8 = 256 threads) of mine ran.

    python benchmarks/cluster/contention.py [--cap 8] RUN [RUN ...]
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

AFH = Path(os.environ.get("MEPD_SCOREBOARD") or Path(__file__).resolve().parents[2] / "scoreboard") / "afh"   # as bench.py


def main(argv: list) -> int:
    cap = int(argv[argv.index("--cap") + 1]) if "--cap" in argv else 8
    runs = [a for a in argv if not a.startswith("--") and not a.isdigit()]
    tasks = []
    for run in runs:
        for fp in sorted((AFH / run / "output").glob("*/result.json")):
            r = json.loads(fp.read_text())
            if r.get("start") and r.get("end"):
                tasks.append((run, r.get("config"), float(r["start"]), float(r["end"])))
    edges = sorted({t for _, _, s, e in tasks for t in (s, e)})
    out = {}
    for run, cfg, s, e in tasks:
        g = out.setdefault(f"{run}:{cfg}", {"n": 0, "start": s, "end": e, "secs": 0.0, "over": 0.0, "conc": []})
        g["n"] += 1
        g["start"], g["end"] = min(g["start"], s), max(g["end"], e)
        for a, b in zip(edges, edges[1:]):
            lo, hi = max(a, s), min(b, e)
            if hi <= lo:
                continue
            c = sum(1 for _, _, s2, e2 in tasks if s2 < hi and e2 > lo)
            g["secs"] += hi - lo
            g["over"] += (hi - lo) if c > cap else 0.0
            g["conc"].append((c, hi - lo))
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")
    for k, g in out.items():
        cs = sorted(g.pop("conc"))
        half, acc, med = g["secs"] / 2, 0.0, None
        for c, w in cs:
            acc += w
            if acc >= half:
                med = c
                break
        print(json.dumps({"arm": k, "tasks": g["n"], "window_utc": [fmt(g["start"]), fmt(g["end"])],
                          "median_concurrent_tasks": med, "share_over_cap": round(g["over"] / g["secs"], 3)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
