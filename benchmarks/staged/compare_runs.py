"""Several runs of one case (identical.py's OUT_DIR/run-*/summary.json, of one commit or two): how many species,
steps and removed TSs each has, and every step whose ends or TS energy are not the same in all of them, with each
run's value -- to tell a difference between two commits from the difference between two runs of one.

    python benchmarks/staged/compare_runs.py OUT_DIR [OUT_DIR ...] [--json OUT.json]
"""
import json
import sys
from pathlib import Path


def main(argv):
    out = None
    if "--json" in argv:
        out = argv[argv.index("--json") + 1]
        argv = [a for a in argv if a not in ("--json", out)]
    runs = {}
    for d in argv:
        for fp in sorted(Path(d).glob("run-*/summary.json")):
            s = json.loads(fp.read_text())
            mol = ((s.get("rounds") or [{}])[-1].get("kinetics") or {}).get("molecules") or {}
            runs[f"{Path(d).name}/{fp.parent.name[4:]}"] = {
                "species": len(s["species"]), "steps": len(s["steps"]), "removed": len(s.get("steps_removed") or []),
                "stopped": (s.get("rounds") or [{}])[-1].get("stopped"),
                "final": sorted((round(float(c), 6) for c in mol.get("final") or []), reverse=True)[:4],
                "by_label": {st["label"]: (st["a"], st["b"], round(st["ts_energy"], 6)) for st in s["steps"]}}
    labels = sorted({k for r in runs.values() for k in r["by_label"]})
    differ = {k: {name: r["by_label"].get(k) for name, r in runs.items()} for k in labels
              if len({r["by_label"].get(k) for r in runs.values()}) > 1}
    table = {name: {k: v for k, v in r.items() if k != "by_label"} for name, r in runs.items()}
    for name, r in table.items():
        print(name, r)
    for k, v in differ.items():
        print(k, v)
    if out:
        Path(out).write_text(json.dumps({"runs": table, "steps_that_differ": differ}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
