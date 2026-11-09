"""Whether every nanoreactor MD schedule of a run records its forces and pulls: run/reactor/run_*/md/schedule.json
inside each task's run.tgz (bench.py's unpack leaves run/reactor/ in the archive), per task: how many schedules,
how many have the keys "forces" and "pulls", and the distinct values of "forces".

    python benchmarks/cluster/schedule_check.py AFH_RUN [AFH_RUN ...]
"""
import json
import re
import sys
import tarfile
from pathlib import Path

AFH = Path("/home/user/astera/data/runs/explorer-team/scoreboard/afh")


def main(argv: list) -> int:
    out, bad = {}, []
    for run in argv:
        for d in sorted((AFH / run / "output").iterdir()):
            if not (d / "run.tgz").is_file():
                continue
            n = has = 0
            forces = set()
            with tarfile.open(d / "run.tgz") as tf:
                for m in tf:
                    if re.fullmatch(r"run/reactor/run_[^/]+/md/schedule\.json", m.name):
                        n += 1
                        s = json.load(tf.extractfile(m))
                        ok = "forces" in s and "pulls" in s
                        has += ok
                        forces.add(json.dumps(s.get("forces"), sort_keys=True))
            out[d.name] = {"schedules": n, "with_forces_and_pulls": has, "forces": sorted(forces)}
            if n == 0 or has < n:
                bad.append(d.name)
    print(json.dumps({"tasks": len(out), "all_complete": not bad, "incomplete": bad, "per_task": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
