"""Gate: does a candidate (commit, configuration) do at least as well as a base one, per system?

    P=/home/user/astera/mepd/.venv/bin/python
    $P benchmarks/cluster/gate.py BASE_RUN:CONFIG CAND_RUN:CONFIG [--only sys1,sys2] [--json OUT]

Each side is a scored cluster run (`bench.py score RUN` wrote results/RUN/aggregate.json) and one of
its configurations. Systems in both are compared; held-out systems count like the others (they are
marked); a system run on one side only fails ("not run"). Per system, all three must hold:

1. accuracy: median(|worst error|) of the candidate <= the base's + max(0.5 kcal/mol, spread), where
   spread = the larger interquartile range of |worst error| over the two sides' replicates (finite
   values; a replicate with a missing channel counts as +inf in the median);
2. recall: the fraction of replicates with every channel found and connected does not drop, and no
   channel's connected fraction drops (strict: 5 replicates resolve 20 % steps, so one replicate less
   fails the gate; its counts are printed);
3. speed: median settle time of the candidate <= the base's + max(20 % of it, 2 x the same-seed rerun spread
   of the settle time (the base's controls, per system; else the largest), 60 s) (coordinator, 2026-11-05;
   a replicate that never settled counts as +inf; both +inf passes, base finite and candidate +inf fails).
The verdicts follow these rules as the brief states them; a failure that is a single replicate (recall) or
within the larger interquartile range of the two sides (settle) is marked so, for a human to judge.

Exit 0 when every system passes, 1 otherwise.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RESULTS = Path("/home/user/astera/data/runs/explorer-team/scoreboard/results")
FLOOR_KCAL = 0.5
SETTLE_GROWTH_FRAC = 0.2     # settle may grow by max(20 %, 2 x the same-seed rerun spread, 60 s) (coordinator)
SETTLE_FLOOR_S = 60.0


def _load(spec: str) -> tuple[dict, dict]:
    run, _, config = spec.partition(":")
    agg = json.loads((RESULTS / run / "aggregate.json").read_text())
    per = {g["system"]: g for g in agg.values() if isinstance(g, dict) and g.get("config") == config}
    if not per:
        raise SystemExit(f"{spec}: no configuration {config!r} in results/{run}/aggregate.json")
    return per, {"run": run, "config": config, "commit": agg.get("commit"),
                 "control_settle_spread_s": agg.get("control_settle_spread_s") or {}}


def _fmt(x, f=".2f"):
    return "-" if x is None else ("inf" if isinstance(x, float) and math.isinf(x) else format(x, f))


def compare(base: dict, cand: dict, rerun_spread: dict | None = None) -> dict:
    """{system: verdict} for two {system: aggregate entry} maps; `rerun_spread`: {system: same-seed rerun spread of
    the settle time, s} (the controls'; a system without one takes the largest)."""
    rerun_spread = rerun_spread or {}
    out = {}
    for system in sorted(set(base) | set(cand)):
        b, c = base.get(system), cand.get(system)
        if b is None or c is None:
            out[system] = {"pass": False, "why": [f"not run on the {'base' if b is None else 'candidate'} side"]}
            continue
        why, checks = [], {}
        # 1. accuracy
        bw, cw = b["worst_abs_error_kcal"], c["worst_abs_error_kcal"]
        spreads = [x["iqr"] for x in (bw, cw) if x.get("iqr") is not None]
        tol = max([FLOOR_KCAL] + spreads)
        bm, cm = bw["median"], cw["median"]
        if bm is None or cm is None:
            ok = cm is not None or bm is None
        elif math.isinf(bm):
            ok = True
        else:
            ok = cm <= bm + tol
        checks["accuracy"] = {"pass": ok, "base_median": bm, "cand_median": cm, "tolerance": tol}
        if not ok:
            why.append(f"worst |error| median {_fmt(cm)} > base {_fmt(bm)} + {tol:.2f} kcal/mol")
        # 2. recall
        ok_r = c["full_recall_fraction"] >= b["full_recall_fraction"] - 1e-9
        drops = [ch for ch in b["recall"] if c["recall"].get(ch, {}).get("connected", 0.0)
                 < b["recall"][ch]["connected"] - 1e-9]
        checks["recall"] = {"pass": ok_r and not drops, "base_full": b["full_recall_fraction"],
                            "cand_full": c["full_recall_fraction"], "channels_dropped": drops,
                            "base": b["recall"], "cand": c["recall"]}
        if not ok_r:
            why.append(f"full recall {c['full_recall_fraction']:.0%} of {c['n']} < base {b['full_recall_fraction']:.0%} "
                       f"of {b['n']}")
        for ch in drops:
            why.append(f"{ch} connected in {c['recall'][ch]['connected']:.0%} < base {b['recall'][ch]['connected']:.0%}"
                       + (" (one replicate)" if b['recall'][ch]['connected'] - c['recall'][ch]['connected']
                          <= 1.0 / max(1, c['n']) + 1e-9 else ""))
        # 3. speed
        bs, cs = b["settle_s"]["median"], c["settle_s"]["median"]
        bs = math.inf if bs is None else bs
        cs = math.inf if cs is None else cs
        rs = rerun_spread.get(system, max(rerun_spread.values()) if rerun_spread else 0.0)
        s_tol = max(SETTLE_GROWTH_FRAC * bs if math.isfinite(bs) else 0.0, 2.0 * rs, SETTLE_FLOOR_S)
        ok_s = cs <= bs + s_tol if math.isfinite(bs) else True
        checks["settle"] = {"pass": ok_s, "base_median_s": bs, "cand_median_s": cs, "tolerance_s": s_tol,
                            "rerun_spread_s": rs}
        if not ok_s:
            why.append(f"settle median {_fmt(cs, '.0f')} s > base {_fmt(bs, '.0f')} s + {s_tol:.0f} s")
        out[system] = {"pass": not why, "held_out": bool(b.get("held_out") or c.get("held_out")), "why": why,
                       "checks": checks, "n": [b["n"], c["n"]]}
    return out


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    base, bi = _load(argv[0])
    cand, ci = _load(argv[1])
    if "--only" in argv:
        keep = set(argv[argv.index("--only") + 1].split(","))
        base = {k: v for k, v in base.items() if k in keep}
        cand = {k: v for k, v in cand.items() if k in keep}
    verdict = compare(base, cand, bi.get("control_settle_spread_s"))
    print(f"gate: base {bi['run']}:{bi['config']} ({(bi['commit'] or '')[:7]}) vs candidate "
          f"{ci['run']}:{ci['config']} ({(ci['commit'] or '')[:7]})")
    for system, v in verdict.items():
        mark = "PASS" if v["pass"] else "FAIL"
        held = " (held out)" if v.get("held_out") else ""
        ch = v.get("checks", {})
        detail = ""
        if ch:
            a, r, s = ch["accuracy"], ch["recall"], ch["settle"]
            detail = (f" | worst|err| {_fmt(a['base_median'])} -> {_fmt(a['cand_median'])} (tol {a['tolerance']:.2f})"
                      f" | full recall {r['base_full']:.0%} -> {r['cand_full']:.0%}"
                      f" | settle {_fmt(s['base_median_s'], '.0f')} -> {_fmt(s['cand_median_s'], '.0f')} s"
                      f" (tol {s['tolerance_s']:.0f})")
        print(f"  {system}{held}: {mark}{detail}")
        for w in v["why"]:
            print(f"      {w}")
    ok = all(v["pass"] for v in verdict.values())
    print("overall:", "PASS" if ok else "FAIL")
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(
            {"base": bi, "candidate": ci, "systems": verdict, "pass": ok}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
