"""Task-by-task comparison of two configurations run with the same seeds: per (system, replicate), whether the
final selectivities and the recall (found / connected per channel) are equal, and if not how far apart:
|RT ln(sel_cand / sel_base)| per target (kcal/mol, at the system's temperature), the channels whose recall
differs, and the species / step counts. For telling noise in a change that should not move results (e.g. live
emitters) from a real change.

    python benchmarks/cluster/pair_compare.py RESULT BASE_CONFIG CAND_CONFIG [--only a,b]
(RESULT: a scored result set under the scoreboard results folder, as bench.py score ... --as RESULT makes)
"""
import json
import math
import sys
from pathlib import Path

RESULTS = Path("/home/user/astera/data/runs/explorer-team/scoreboard/results")
R_KCAL = 1.987204e-3


def scores(result: str, config: str) -> dict:
    agg = json.loads((RESULTS / result / "aggregate.json").read_text())
    by = {}
    for run in agg.get("runs", [result]):
        for fp in sorted((RESULTS / run).glob("*/score.json")):
            o = json.loads(fp.read_text())
            if o["config"] == config and not o.get("control_of"):
                by[(o["system"], o["replicate"])] = o
    return by


def _dev(a, b, temp):
    if a == b:
        return 0.0
    if a is None or b is None or not (0 < a < math.inf and 0 < b < math.inf):
        return None   # one of them missing, zero or infinite: reported as unequal without a distance
    return abs(R_KCAL * temp * math.log(b / a))


def main(argv: list) -> int:
    only = None
    if "--only" in argv:
        k = argv.index("--only")
        only, argv = set(argv[k + 1].split(",")), argv[:k] + argv[k + 2:]
    result, base, cand = argv[:3]
    sa, sb = scores(result, base), scores(result, cand)
    rows = []
    for key in sorted(set(sa) & set(sb)):
        if only and key[0] not in only:
            continue
        a, b = sa[key], sb[key]
        row = {"system": key[0], "replicate": key[1], "ok": [a.get("ok"), b.get("ok")]}
        if a.get("ok") and b.get("ok"):
            temp = float(a.get("kinetics_T") or b.get("kinetics_T") or 298.15)
            sel_a = [t["selectivity"] for t in a["targets"]]
            sel_b = [t["selectivity"] for t in b["targets"]]
            devs = [_dev(x, y, temp) for x, y in zip(sel_a, sel_b)]
            rec = [c for c in a["recall"]["found"]
                   if (a["recall"]["found"][c], a["recall"]["connected"][c])
                   != (b["recall"]["found"].get(c), b["recall"]["connected"].get(c))]
            row.update(sel_equal=sel_a == sel_b, recall_equal=not rec, sel=[sel_a, sel_b], dev_kcal=devs,
                       recall_differs=rec, species=[a["species"], b["species"]], steps=[a["steps"], b["steps"]],
                       worst_abs_error_kcal=[a.get("worst_abs_error_kcal"), b.get("worst_abs_error_kcal")])
        rows.append(row)
    both = [r for r in rows if "sel_equal" in r]
    finite = [d for r in both for d in r["dev_kcal"] if d is not None]
    out = {"base": base, "candidate": cand, "pairs": len(rows), "both_ok": len(both),
           "sel_equal": sum(r["sel_equal"] for r in both), "recall_equal": sum(r["recall_equal"] for r in both),
           "same_network_size": sum(r["species"][0] == r["species"][1] and r["steps"][0] == r["steps"][1] for r in both),
           "dev_kcal_max": max(finite) if finite else None,
           "dev_kcal_median": sorted(finite)[len(finite) // 2] if finite else None,
           "dev_undefined": sum(d is None for r in both for d in r["dev_kcal"]), "rows": rows}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
