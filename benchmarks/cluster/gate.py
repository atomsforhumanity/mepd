"""Gate: does a candidate (commit, configuration) do at least as well as a base one, per system?

    P=python
    $P benchmarks/cluster/gate.py BASE_RUN:CONFIG CAND_RUN:CONFIG [--only sys1,sys2] [--spread-from RESULT] [--json OUT]
        [--settle-basis partial]   (settle on the partial basis for both sides: a run without the live network
                                    against one with it; score.py settle)

Each side is a scored cluster run (`bench.py score RUN` wrote results/RUN/aggregate.json) and one of
its configurations. Systems in both are compared; held-out systems count like the others (they are
marked); a system run on one side only fails ("not run"). Per system, all three must hold:

1. accuracy: median(|worst error|) of the candidate <= the base's + max(0.5 kcal/mol, spread), where
   spread = the larger interquartile range of |worst error| over the two sides' replicates (finite
   values; a replicate with a missing channel counts as +inf in the median);
2. recall (coordinator 2026-11-07, two-stage; recall_noise.py: the strict rule failed 85-88 % of gates between
   arms of the same run code): only the channels systems.yaml marks as required count, and the full-recall
   count. A drop with a one-sided Fisher exact p < 0.05 (connected counts, base vs candidate) fails; a smaller
   drop is inconclusive -- the system needs replicates (10 more per arm, seeds 1000 x r for r = 5..14) -- until
   both sides have >= 15 replicates, when only p < 0.05 fails;
3. speed: median settle time of the candidate <= the base's + max(20 % of it, 2 x the same-seed rerun spread
   of the settle time (the base's controls, per system; else the largest), 60 s) (coordinator, 2026-11-05;
   a replicate that never settled counts as +inf; base +inf passes). Two-stage (coordinator 2026-11-09, from
   the sb-n85 same-commit rerun: isoprene's 5-run median moved 355 s > its tolerance between identical runs):
   a median over that fails only if a one-sided Mann-Whitney test on the runs' settle times (never settled =
   +inf) also says the candidate is slower, p < 0.05; otherwise it is inconclusive (needs replicates) until
   both sides have >= 15 replicates, when only p < 0.05 fails. Without per-run settle times (an aggregate
   alone) the median rule decides as before.
The verdicts follow these rules as the brief states them; a failure that is a single replicate (recall) or
within the larger interquartile range of the two sides (settle) is marked so, for a human to judge.

Exit 0 when every system passes, 1 when one fails, 3 when none fails but one is inconclusive.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

RESULTS = Path(os.environ.get("MEPD_SCOREBOARD") or Path(__file__).resolve().parents[2] / "scoreboard") / "results"   # as bench.py
FLOOR_KCAL = 0.5
SETTLE_GROWTH_FRAC = 0.2     # settle may grow by max(20 %, 2 x the same-seed rerun spread, 60 s) (coordinator)
SETTLE_FLOOR_S = 60.0
RECALL_P = 0.05              # one-sided Fisher exact test on connected counts
STAGE2_N = 15                # replicates per side from which a non-significant recall drop / settle rise passes
SETTLE_P = 0.05              # one-sided Mann-Whitney test on the settle times (candidate slower)


def _required(system: str) -> list | None:
    """The system's required channels (systems.yaml, through score.load_systems)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import score

    sd = score.load_systems().get(system) or {}
    return sd.get("required") or (list(sd["channels"]) if sd.get("channels") else None)


def recall_drop(kb: int, nb: int, kc: int, nc: int) -> str | None:
    """None (no drop), "fail" (one-sided Fisher p < RECALL_P) or "inconclusive" (a smaller drop, before both
    sides have STAGE2_N replicates; after that a smaller drop is None)."""
    from scipy.stats import fisher_exact

    if kc * nb >= kb * nc:
        return None
    p = fisher_exact([[kb, nb - kb], [kc, nc - kc]], alternative="greater")[1]
    if p < RECALL_P:
        return "fail"
    return None if min(nb, nc) >= STAGE2_N else "inconclusive"


def _load(spec: str) -> tuple[dict, dict]:
    run, _, config = spec.partition(":")
    agg = json.loads((RESULTS / run / "aggregate.json").read_text())
    per = {g["system"]: g for g in agg.values() if isinstance(g, dict) and g.get("config") == config}
    if not per:
        raise SystemExit(f"{spec}: no configuration {config!r} in results/{run}/aggregate.json")
    # the runs' settle times per system (for the settle check's rank test), as scored and on the partial basis
    by_id = {}
    for r in agg.get("runs", [run]):
        for fp in sorted((RESULTS / r).glob("*/score.json")):
            o = json.loads(fp.read_text())
            if o.get("config") == config and not o.get("control_of"):
                by_id[o["id"]] = o
    for o in by_id.values():
        g = per.get(o["system"])
        if g is None:
            continue
        st = o.get("settle") or {}
        for key, field in (("settle_values", "settle_s"), ("settle_values_partial", "settle_s_partial_basis")):
            v = st.get(field) if o.get("ok") else None
            g.setdefault(key, []).append(math.inf if v is None else float(v))
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
        # 2. recall: required channels and full recall, two-stage
        nb, nc = b["n"], c["n"]
        req = b.get("required") or c.get("required") or _required(system) or list(b["recall"])
        counts = {"full recall": (round(b["full_recall_fraction"] * nb), round(c["full_recall_fraction"] * nc))}
        counts.update({ch: (round(b["recall"][ch]["connected"] * nb),
                            round(c["recall"].get(ch, {}).get("connected", 0.0) * nc)) for ch in req if ch in b["recall"]})
        outcome = {k: recall_drop(kb, nb, kc, nc) for k, (kb, kc) in counts.items()}
        failed = [k for k, o in outcome.items() if o == "fail"]
        unsure = [k for k, o in outcome.items() if o == "inconclusive"]
        checks["recall"] = {"pass": not failed and not unsure, "status": "fail" if failed else
                            ("inconclusive" if unsure else "pass"), "required": req, "counts": counts,
                            "failed": failed, "inconclusive": unsure, "base_full": b["full_recall_fraction"],
                            "cand_full": c["full_recall_fraction"], "base": b["recall"], "cand": c["recall"]}
        for k in failed + unsure:
            kb, kc = counts[k]
            why.append(f"{k}: {kc}/{nc} < base {kb}/{nb} ({'Fisher p < %.2f' % RECALL_P if k in failed else 'not significant: inconclusive, needs replicates'})")
        # 3. speed
        bs, cs = b["settle_s"]["median"], c["settle_s"]["median"]
        bs = math.inf if bs is None else bs
        cs = math.inf if cs is None else cs
        rs = rerun_spread.get(system, max(rerun_spread.values()) if rerun_spread else 0.0)
        s_tol = max(SETTLE_GROWTH_FRAC * bs if math.isfinite(bs) else 0.0, 2.0 * rs, SETTLE_FLOOR_S)
        over = (cs > bs + s_tol) if math.isfinite(bs) else False
        s_status, p_s = ("pass" if not over else "fail"), None
        bv, cv = b.get("settle_values"), c.get("settle_values")
        if over and bv and cv:
            from scipy.stats import mannwhitneyu

            p_s = float(mannwhitneyu(cv, bv, alternative="greater").pvalue)
            s_status = "fail" if p_s < SETTLE_P else ("pass" if min(len(bv), len(cv)) >= STAGE2_N else "inconclusive")
        checks["settle"] = {"pass": s_status == "pass", "status": s_status, "base_median_s": bs, "cand_median_s": cs,
                            "tolerance_s": s_tol, "rerun_spread_s": rs, "mann_whitney_p_slower": p_s}
        if over:
            why.append(f"settle median {_fmt(cs, '.0f')} s > base {_fmt(bs, '.0f')} s + {s_tol:.0f} s"
                       + ("" if p_s is None else f" (Mann-Whitney p {p_s:.3f}: "
                          + {"fail": "slower", "inconclusive": "not significant: inconclusive, needs replicates",
                             "pass": "not significant at >= 15 a side: passes"}[s_status] + ")"))
        hard = not checks["accuracy"]["pass"] or s_status == "fail" or checks["recall"]["status"] == "fail"
        unsure = checks["recall"]["status"] == "inconclusive" or s_status == "inconclusive"
        status = "fail" if hard else ("inconclusive" if unsure else "pass")
        out[system] = {"pass": status == "pass", "status": status,
                       "held_out": bool(b.get("held_out") or c.get("held_out")), "why": why,
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
    spread = bi.get("control_settle_spread_s")
    if "--spread-from" in argv:     # the same-seed rerun spreads of another result set (one with controls)
        spread = json.loads((RESULTS / argv[argv.index("--spread-from") + 1] / "aggregate.json").read_text()
                            ).get("control_settle_spread_s") or {}
    if "--settle-basis" in argv and argv[argv.index("--settle-basis") + 1] == "partial":
        for side in (base, cand):
            for g in side.values():
                g["settle_s"] = g["settle_s_partial_basis"]
                g["settle_values"] = g.get("settle_values_partial")
    verdict = compare(base, cand, spread)
    print(f"gate: base {bi['run']}:{bi['config']} ({(bi['commit'] or '')[:7]}) vs candidate "
          f"{ci['run']}:{ci['config']} ({(ci['commit'] or '')[:7]})")
    for system, v in verdict.items():
        mark = v.get("status", "pass" if v["pass"] else "fail").upper()
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
    statuses = {v.get("status", "pass" if v["pass"] else "fail") for v in verdict.values()}
    overall = "fail" if "fail" in statuses else ("inconclusive" if "inconclusive" in statuses else "pass")
    need = [sy for sy, v in verdict.items() if v.get("status") == "inconclusive"]
    print("overall:", overall.upper())
    if need:
        print("needs replicates (10 more per arm, seeds 5000-14000):", ", ".join(need))
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(
            {"base": bi, "candidate": ci, "systems": verdict, "pass": ok, "overall": overall,
             "needs_replicates": need}, indent=1, default=str))
    return {"pass": 0, "fail": 1, "inconclusive": 3}[overall]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
