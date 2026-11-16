"""What place_network.py's steps.jsonl files say: statuses, spreads, shifts, and one setting against another.

    python benchmarks/diagnostics/addmol/place_network_report.py DIR [OTHER_DIR]
"""
import json
import statistics as S
import sys
from pathlib import Path


def load(d):
    return {r["label"]: r for r in map(json.loads, (Path(d) / "steps.jsonl").read_text().splitlines())}


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))]


def main():
    a = load(sys.argv[1])
    run = json.loads((Path(sys.argv[1]) / "run.json").read_text())
    print({k: run[k] for k in ("steps", "workers", "wall_s", "first_result_s", "tenth_result_s", "core_s", "status")})
    for key in ("barrier_shift", "reaction_shift"):
        el = [r[key]["electronic"] for r in a.values() if r.get(key, {}).get("electronic") is not None]
        sp = [r[key]["spread"] for r in a.values() if r.get(key, {}).get("spread") is not None]
        fr = [r[key]["free"] for r in a.values() if r.get(key, {}).get("free") is not None]
        print(key, "n", len(el), "electronic: median %.1f, 10th %.1f, 90th %.1f, min %.1f, max %.1f" % (S.median(el), q(el, .1), q(el, .9), min(el), max(el)))
        if sp:
            print("  spread (range over repeats): median %.1f, 75th %.1f, 90th %.1f, max %.1f; above 2 kcal/mol: %d of %d"
                  % (S.median(sp), q(sp, .75), q(sp, .9), max(sp), sum(x > 2 for x in sp), len(sp)))
        if fr and el:
            d = [r[key]["free"] - r[key]["electronic"] for r in a.values() if r.get(key, {}).get("free") is not None]
            print("  free minus electronic: median %.2f, largest %.2f" % (S.median(d), max(d, key=abs)))
    for f in ("r", "ts", "p"):
        b = [r["binding"][f]["electronic"] for r in a.values() if r.get("binding", {}).get(f, {}).get("electronic") is not None]
        g = [r["binding"][f]["free"] for r in a.values() if r.get("binding", {}).get(f, {}).get("free") is not None]
        print("binding", f, "electronic median %.1f (%.1f to %.1f); free at 1 M median %.1f (%.1f to %.1f)"
              % (S.median(b), min(b), max(b), S.median(g), min(g), max(g)))
    w = [r["wall_s"] for r in a.values()]
    print("seconds of one core per step: median %.0f, 90th %.0f, max %.0f" % (S.median(w), q(w, .9), max(w)))
    if len(sys.argv) > 2:
        b = load(sys.argv[2])
        d = [a[k]["barrier_shift"]["electronic"] - b[k]["barrier_shift"]["electronic"] for k in a
             if k in b and a[k]["barrier_shift"]["electronic"] is not None and b[k]["barrier_shift"]["electronic"] is not None]
        print("barrier shift, first minus second folder: n %d, mean %.2f, rms %.2f, |d| > 2: %d, same sign: %d"
              % (len(d), S.mean(d), (sum(x * x for x in d) / len(d)) ** 0.5, sum(abs(x) > 2 for x in d),
                 sum((a[k]["barrier_shift"]["electronic"] > 0) == (b[k]["barrier_shift"]["electronic"] > 0) for k in a
                     if k in b and a[k]["barrier_shift"]["electronic"] is not None and b[k]["barrier_shift"]["electronic"] is not None)))


if __name__ == "__main__":
    main()
