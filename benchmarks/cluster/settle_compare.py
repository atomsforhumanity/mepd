"""Settle-time distributions of two configurations of one system, side by side: n, median, IQR, min/max, the values,
and a two-sided Mann-Whitney U test (scipy; runs that never settled count as +inf, i.e. rank last).

    python benchmarks/cluster/settle_compare.py RESULT SYSTEM BASE_CONFIG CAND_CONFIG
(RESULT: a scored result set under the scoreboard results folder, e.g. one made by bench.py score ... --as RESULT)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

RESULTS = Path("/home/user/astera/data/runs/explorer-team/scoreboard/results")


def settles(result: str, system: str, config: str) -> list:
    agg = json.loads((RESULTS / result / "aggregate.json").read_text())
    by_id = {}
    for run in agg.get("runs", [result]):
        for fp in sorted((RESULTS / run).glob("*/score.json")):
            o = json.loads(fp.read_text())
            if o["system"] == system and o["config"] == config and not o.get("control_of"):
                by_id[o["id"]] = o
    return [((o.get("settle") or {}).get("settle_s") if o.get("ok") else None) for o in by_id.values()]


def summary(xs: list) -> dict:
    v = np.array([math.inf if x is None else x for x in xs], dtype=float)
    fin = v[np.isfinite(v)]
    q1, q3 = (np.percentile(fin, 25), np.percentile(fin, 75)) if len(fin) else (None, None)
    return {"n": len(v), "never": int((~np.isfinite(v)).sum()), "median": float(np.median(v)),
            "iqr": None if q1 is None else float(q3 - q1), "min": float(fin.min()) if len(fin) else None,
            "max": float(fin.max()) if len(fin) else None, "values": sorted(round(float(x)) if np.isfinite(x) else "inf"
                                                                           for x in v)}


def main(argv: list) -> int:
    result, system, base, cand = argv[:4]
    a, b = settles(result, system, base), settles(result, system, cand)
    fa = [math.inf if x is None else x for x in a]
    fb = [math.inf if x is None else x for x in b]
    u = mannwhitneyu(fa, fb, alternative="two-sided")
    out = {"system": system, "base": {"config": base, **summary(a)}, "candidate": {"config": cand, **summary(b)},
           "mann_whitney_u": float(u.statistic), "p_two_sided": float(u.pvalue)}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
