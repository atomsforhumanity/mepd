"""A run's job log (network_aware.jsonl) by job kind: jobs, worker seconds, and the seconds the main process held
the queue (`main_s`), beside read_summary.py (which reads summary.json only).

    python benchmarks/staged/read_jobs.py RUN_DIR [RUN_DIR ...] [--json OUT.json]
"""
import collections
import json
import sys
from pathlib import Path


def read(run: Path) -> dict:
    jobs, secs, main_s, cancelled = collections.Counter(), collections.Counter(), 0.0, 0
    for line in (run / "network_aware.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("event") == "job":
            if e.get("cancelled"):
                cancelled += 1
                continue
            jobs[e["kind"]] += 1
            secs[e["kind"]] += float(e.get("run_s") or 0.0)
        elif e.get("event") == "horizontal_connect":
            main_s += float(e.get("main_s") or 0.0)
    return {"run": str(run), "jobs": dict(jobs), "worker_s": {k: round(v, 1) for k, v in secs.items()},
            "worker_s_total": round(sum(secs.values()), 1), "main_s": round(main_s, 1), "cancelled": cancelled}


def main(argv):
    out = None
    if "--json" in argv:
        out = argv[argv.index("--json") + 1]
        argv = [a for a in argv if a not in ("--json", out)]
    rows = [read(Path(a)) for a in argv]
    for r in rows:
        print(json.dumps(r))
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
