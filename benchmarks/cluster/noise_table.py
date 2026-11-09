"""Two arms of the same commit and seeds (a rerun) side by side, per system: each metric the gate checks, its
difference between the arms, and the tolerance the gate applies to it -- how far a 5-run pass can be trusted.

    python benchmarks/cluster/noise_table.py RESULT_A:CONFIG RESULT_B:CONFIG [--spread-from RESULT] [--settle-basis partial]

Per system: median worst |error| (kcal/mol) A, B, |diff|, tolerance; median settle (s) A, B, |diff|, tolerance and a
two-sided Mann-Whitney p on the settle times (never settled = +inf); full recall and each required channel's
connected count A/B; runs stopped at max_species A/B; median species A/B; and gate.compare's status both ways
(A as base and B as base).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate  # noqa: E402
from recall_noise import runs_of  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402


def main(argv: list) -> int:
    spread_from = argv[argv.index("--spread-from") + 1] if "--spread-from" in argv else "sb-base-axtb"
    partial = "--settle-basis" in argv and argv[argv.index("--settle-basis") + 1] == "partial"
    a_spec, b_spec = [x for x in argv if ":" in x][:2]
    A, ai = gate._load(a_spec)
    B, bi = gate._load(b_spec)
    if partial:
        for side in (A, B):
            for g in side.values():
                g["settle_s"] = g["settle_s_partial_basis"]
    spread = json.loads((gate.RESULTS / spread_from / "aggregate.json").read_text()).get("control_settle_spread_s") or {}
    ab, ba = gate.compare(A, B, spread), gate.compare(B, A, spread)
    ra, rb = runs_of(*a_spec.split(":", 1)), runs_of(*b_spec.split(":", 1))
    key = "settle_s_partial_basis" if partial else "settle_s"
    out = {}
    for sy in sorted(set(A) & set(B)):
        acc, st = ab[sy]["checks"]["accuracy"], ab[sy]["checks"]["settle"]
        sa = [((o.get("settle") or {}).get(key) or math.inf) if o.get("ok") else math.inf
              for (s, _), o in ra.items() if s == sy]
        sb = [((o.get("settle") or {}).get(key) or math.inf) if o.get("ok") else math.inf
              for (s, _), o in rb.items() if s == sy]
        cap = lambda rs: sum("max species" in str(o.get("stopped")) for (s, _), o in rs.items() if s == sy)
        spc = lambda rs: sorted(o.get("species") or 0 for (s, _), o in rs.items() if s == sy)
        med = lambda xs: xs[len(xs) // 2] if xs else None
        d = lambda x, y: None if x is None or y is None or math.isinf(x) or math.isinf(y) else abs(x - y)
        out[sy] = {
            "n": [A[sy]["n"], B[sy]["n"]],
            "accuracy": {"a": acc["base_median"], "b": acc["cand_median"], "diff": d(acc["base_median"], acc["cand_median"]),
                         "tol": acc["tolerance"]},
            "settle": {"a": st["base_median_s"], "b": st["cand_median_s"], "diff": d(st["base_median_s"], st["cand_median_s"]),
                       "tol": st["tolerance_s"],
                       "mann_whitney_p": float(mannwhitneyu(sa, sb, alternative="two-sided").pvalue) if sa and sb else None},
            "recall_counts": ab[sy]["checks"]["recall"]["counts"],
            "cap_stops": [cap(ra), cap(rb)], "species_median": [med(spc(ra)), med(spc(rb))],
            "status_a_base": ab[sy]["status"], "status_b_base": ba[sy]["status"]}
    print(json.dumps({"a": a_spec, "b": b_spec, "settle_basis": "partial" if partial else "as scored",
                      "systems": out}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
