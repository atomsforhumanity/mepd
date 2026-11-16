"""Tables from place_measure.py's tasks.jsonl files: spread of the bystander shifts over repeats against cost.

    python benchmarks/diagnostics/addmol/place_analyze.py OUT.json MEAS_DIR [MEAS_DIR ...] [--final]

Per step, placer, with / without carrying, and number of placed poses n per frame: each repeat's barrier and
reaction-energy shift from the lowest eligible pose of each frame among its first n (plus the poses carried at
that n); their standard deviation and range over the repeats; the seconds one repeat took on its one thread;
how far each repeat's lowest complex lies above the lowest one any placer and repeat found for that frame
(the pooled lowest), and the pooled shift; and, without carrying, after how many placed poses a repeat's
own lowest complex of a frame is within 0.5 kcal/mol of what its 24 poses reach. --final: the same on the second level's single points
(lowest pose chosen at that level). Prints markdown tables; everything goes to OUT.json.
"""
import argparse
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

H = 627.509474
UNMOVED, CONTACT = 1e-3, 1.3


def ok(p):
    return "failed" not in p and not p["reacted"] and p["moved"] > UNMOVED and p["contact"] <= CONTACT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    key = "final_energy" if a.final else "energy"
    tasks = []
    for d in a.dirs:
        for line in (Path(d) / "tasks.jsonl").read_text().splitlines():
            t = json.loads(line)
            t["system"] = Path(d).name
            tasks.append(t)
    rows, pooled = [], defaultdict(lambda: 9e9)

    def ref(t):
        if a.final:
            return t["final"]["step_energies"], t["final"]["molecule_energy"]
        return t["step_energies"], t["molecule_energy"]

    def bind(t, f, e):
        ex, ec = ref(t)
        return (e - ex[f] - ec) * H

    for t in tasks:                                   # the pooled lowest binding per step and frame
        for f, ps in t["poses"].items():
            groups = [ps] + [c["poses"][f] for c in t["carried"].values()]
            for g in groups:
                for p in g:
                    if ok(p) and key in p:
                        k = (t["system"], t["label"], f)
                        pooled[k] = min(pooled[k], bind(t, f, p[key]))
    prefixes = sorted(int(n) for n in tasks[0]["carried"])
    per = defaultdict(list)
    for t in tasks:
        dts = {}
        last = 0.0
        for f in ("r", "ts", "p"):
            dts[f] = []
            for p in t["poses"][f]:
                dts[f].append(p["t"] - last)
                last = p["t"]
        for n in prefixes:
            for carry in (0, 1):
                b, wall = {}, t["setup_s"]
                for f in ("r", "ts", "p"):
                    ps = [p for p in t["poses"][f][:n]]
                    wall += sum(dts[f][:n])
                    if carry:
                        ps = ps + t["carried"][str(n)]["poses"][f]
                    es = [p[key] for p in ps if ok(p) and key in p]
                    if es:
                        b[f] = bind(t, f, min(es))
                if carry:
                    wall += t["carried"][str(n)]["wall_s"]
                per[(t["system"], t["label"], t["placer"], carry, n)].append({"binding": b, "wall_s": wall,
                                                                                "repeat": t["repeat"]})
    for (system, label, placer, carry, n), reps in sorted(per.items()):
        full = [r for r in reps if len(r["binding"]) == 3]
        bs = [r["binding"]["ts"] - r["binding"]["r"] for r in full]
        rs = [r["binding"]["p"] - r["binding"]["r"] for r in full]
        pl = {f: pooled[(system, label, f)] for f in ("r", "ts", "p")}
        above = [r["binding"][f] - pl[f] for r in full for f in ("r", "ts", "p")]
        rows.append({"system": system, "label": label, "placer": placer, "carry": carry, "poses": n,
                     "repeats": len(reps), "complete": len(full),
                     "barrier_shift": bs, "reaction_shift": rs,
                     "barrier_shift_std": S.pstdev(bs) if len(bs) > 1 else None,
                     "barrier_shift_range": max(bs) - min(bs) if bs else None,
                     "reaction_shift_std": S.pstdev(rs) if len(rs) > 1 else None,
                     "barrier_shift_mean": S.mean(bs) if bs else None,
                     "pooled_barrier_shift": pl["ts"] - pl["r"], "pooled_reaction_shift": pl["p"] - pl["r"],
                     "pooled_binding": pl,
                     "barrier_shift_rms_from_pooled": (sum((x - (pl["ts"] - pl["r"])) ** 2 for x in bs) / len(bs)) ** 0.5 if bs else None,
                     "within_0.5_of_pooled": sum(x <= 0.5 for x in above) / len(above) if above else None,
                     "mean_above_pooled": S.mean(above) if above else None,
                     "wall_s": S.mean(r["wall_s"] for r in reps)})
    Path(a.out).write_text(json.dumps({"rows": rows, "pooled": {"|".join(k): v for k, v in pooled.items()}}, indent=1))
    # summary per system: over its steps
    print("| system | placer | carry | poses | s per repeat (1 thread) | barrier-shift std over 5 repeats: median / max over steps | rms from pooled shift | frames within 0.5 of pooled lowest |")
    print("|---|---|---|---|---|---|---|---|")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["system"], r["placer"], r["carry"], r["poses"])].append(r)
    for (system, placer, carry, n), rs in sorted(agg.items()):
        sd = [r["barrier_shift_std"] for r in rs if r["barrier_shift_std"] is not None]
        rm = [r["barrier_shift_rms_from_pooled"] for r in rs if r["barrier_shift_rms_from_pooled"] is not None]
        w5 = [r["within_0.5_of_pooled"] for r in rs if r["within_0.5_of_pooled"] is not None]
        if not sd:
            continue
        print(f"| {system} | {placer} | {'yes' if carry else 'no'} | {n} | {S.mean(r['wall_s'] for r in rs):.1f} | "
              f"{S.median(sd):.2f} / {max(sd):.2f} | {S.median(rm):.2f} / {max(rm):.2f} | {100 * S.mean(w5):.0f} % |")
    stops_changing(tasks, key)



def stops_changing(tasks, key="energy"):
    """Per system and placer: the number of placed poses after which a repeat's lowest eligible pose of a
    frame is within 0.5 kcal/mol of the lowest of all its placed poses (median, 90th percentile, over
    repeats, frames and steps)."""
    out = defaultdict(list)
    for t in tasks:
        for f, ps in t["poses"].items():
            es = [p[key] if ok(p) and key in p else None for p in ps]
            have = [e for e in es if e is not None]
            if not have:
                continue
            low, best = min(have), None
            for n, e in enumerate(es, 1):
                best = e if best is None or (e is not None and e < best) else best
                if best is not None and (best - low) * H <= 0.5:
                    out[(t["system"], t["placer"])].append(n)
                    break
    print("\n| system | placer | placed poses until the repeat's lowest complex of a frame stops changing by more than 0.5 "
          "kcal/mol (of 24): median / 90th percentile |")
    print("|---|---|---|")
    for (system, placer), ns in sorted(out.items()):
        ns.sort()
        print(f"| {system} | {placer} | {S.median(ns):.0f} / {ns[int(0.9 * (len(ns) - 1))]} |")


if __name__ == "__main__":
    main()
