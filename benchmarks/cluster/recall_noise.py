"""How often the gate's recall check fails between arms whose run code is identical or nearly so (a false fail),
and how candidate recall rules trade that against catching a real loss.

    python benchmarks/cluster/recall_noise.py NAME=RESULT:CONFIG,RESULT:CONFIG [NAME=...] [--boot N] [--seed S]

Each NAME is a null group: two arms of the same (or nearly the same) run code and seeds. Systems where either arm
hit max_species are left out of a group (the cap is then a real difference). Per group:
- empirical: gate.compare's recall check both ways (A as base, B as candidate, and back): per system pass/fail and
  which channels dropped;
- per task (same seed in both arms): how many (task, channel) pairs disagree on `connected`;
- bootstrap: the runs of both arms pooled per system (same code: one population), 5 base and 5 candidate runs drawn
  with replacement N times; the probability that each rule fails the system, and that it fails at least one system
  of the group (the gate's false-fail rate for those systems).
Rules (recall only; accuracy and settle unchanged):
  strict          the gate now: full recall fraction and every channel's connected fraction may not drop
  required        the same, required channels only (systems.yaml `required`; full recall already is)
  required_drop2  required channels; fails only on a drop of >= 2 replicates (of 5)
  required_fisher required channels; fails when a one-sided Fisher exact test on connected counts gives p < 0.05
  two_stage       required channels; at 5 vs 5 a Fisher p < 0.05 fails, any other drop is inconclusive and adds 10
                  replicates per arm for that system, then Fisher p < 0.05 on 15 vs 15 decides (`rerun` = how often
                  the extra replicates are needed)
Power: for each rule, the chance it fails a system whose one required channel truly drops from connected in every
run to a rate q (binomial draws, 5 vs 5), q = 0.8, 0.6, 0.4, 0.2, 0.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gate  # noqa: E402
import score  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

RESULTS = gate.RESULTS


def runs_of(result: str, config: str) -> dict:
    """{(system, replicate): score} of one arm, a later run replacing an earlier one's same id (as bench.py)."""
    agg = json.loads((RESULTS / result / "aggregate.json").read_text())
    by_id = {}
    for run in agg.get("runs", [result]):
        for fp in sorted((RESULTS / run).glob("*/score.json")):
            o = json.loads(fp.read_text())
            if o["config"] == config and not o.get("control_of"):
                by_id[o["id"]] = o
    return {(o["system"], o["replicate"]): o for o in by_id.values()}


def conn(o: dict, channels: list) -> dict:
    """{channel: connected} of one run; a run that failed connects nothing."""
    if not o.get("ok"):
        return {c: False for c in channels}
    return {c: bool(o["recall"]["connected"].get(c)) for c in channels}


def capped(o: dict) -> bool:
    return "max species" in str(o.get("stopped") or "").lower() or "max_species" in str(o.get("stopped") or "")


def rule_fails(rule: str, base: list, cand: list, channels: list, required: list) -> bool:
    """base, cand: lists of {channel: connected} (one per run)."""
    return bool(_rule_fails(rule, base, cand, channels, required))


def _rule_fails(rule: str, base: list, cand: list, channels: list, required: list):
    chans = channels if rule == "strict" else required
    full = lambda rs: sum(all(r[c] for c in required) for r in rs)
    if rule == "strict":
        if full(cand) * len(base) < full(base) * len(cand):
            return True
        return any(sum(r[c] for r in cand) * len(base) < sum(r[c] for r in base) * len(cand) for c in chans)
    if rule == "required":
        if full(cand) * len(base) < full(base) * len(cand):
            return True
        return any(sum(r[c] for r in cand) * len(base) < sum(r[c] for r in base) * len(cand) for c in chans)
    if rule == "required_drop2":
        return any(sum(r[c] for r in base) - sum(r[c] for r in cand) >= 2 for c in chans) or \
            full(base) - full(cand) >= 2
    if rule == "required_fisher":
        def worse(kb, kc):
            if kc >= kb:
                return False
            p = fisher_exact([[kb, len(base) - kb], [kc, len(cand) - kc]], alternative="greater")[1]
            return p < 0.05
        return any(worse(sum(r[c] for r in base), sum(r[c] for r in cand)) for c in chans) or \
            worse(full(base), full(cand))
    raise ValueError(rule)


RULES = ("strict", "required", "required_drop2", "required_fisher")


def two_stage(base15: list, cand15: list, channels: list, required: list) -> tuple[bool, bool]:
    """(fails, needed the extra replicates) of the two-stage rule on 15 runs a side (the first 5 are stage 1)."""
    if rule_fails("required_fisher", base15[:5], cand15[:5], channels, required):
        return True, False
    if not rule_fails("required", base15[:5], cand15[:5], channels, required):
        return False, False
    return rule_fails("required_fisher", base15, cand15, channels, required), True


def main(argv: list) -> int:
    boot = int(argv[argv.index("--boot") + 1]) if "--boot" in argv else 20000
    rng = random.Random(int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0)
    specs = [a for a in argv if "=" in a and not a.startswith("--")]
    suite = score.load_systems()
    out = {"groups": {}, "power": {}}
    for spec in specs:
        name, _, arms = spec.partition("=")
        (ra, ca), (rb, cb) = (a.split(":", 1) for a in arms.split(","))
        A, B = runs_of(ra, ca), runs_of(rb, cb)
        systems = sorted({s for s, _ in A} & {s for s, _ in B})
        grp = {"arms": [f"{ra}:{ca}", f"{rb}:{cb}"], "systems": {}, "excluded_capped": []}
        # empirical: gate.compare both ways on aggregates rebuilt from these runs
        for sy in systems:
            ra_ = [o for (s, _), o in A.items() if s == sy]
            rb_ = [o for (s, _), o in B.items() if s == sy]
            if any(capped(o) for o in ra_ + rb_):
                grp["excluded_capped"].append(sy)
                continue
            chans = list(suite[sy]["channels"])
            req = suite[sy].get("required") or chans
            ca_ = [conn(o, chans) for o in ra_]
            cb_ = [conn(o, chans) for o in rb_]
            emp = {r: [rule_fails(r, ca_, cb_, chans, req), rule_fails(r, cb_, ca_, chans, req)] for r in RULES}
            # gate.compare itself (strict), as a check that rule_fails("strict") is the gate's recall check
            agg = score.aggregate(ra_ + rb_, suite)
            ga = next(g for g in agg.values() if isinstance(g, dict) and g.get("config") == ca and g.get("system") == sy)
            gb = next(g for g in agg.values() if isinstance(g, dict) and g.get("config") == cb and g.get("system") == sy) \
                if ca != cb else None
            gate_fail = None
            if gb is not None:
                v1 = gate.compare({sy: ga}, {sy: gb})[sy]["checks"]["recall"]["pass"]
                v2 = gate.compare({sy: gb}, {sy: ga})[sy]["checks"]["recall"]["pass"]
                gate_fail = [not v1, not v2]
            pairs = sorted({rep for (s, rep) in A if s == sy} & {rep for (s, rep) in B if s == sy})
            dis = sum(conn(A[(sy, rep)], chans)[c] != conn(B[(sy, rep)], chans)[c] for rep in pairs for c in chans)
            pool = ca_ + cb_
            fails = {r: 0 for r in RULES + ("two_stage", "rerun")}
            for _ in range(boot):
                bs = [rng.choice(pool) for _ in range(15)]
                cs = [rng.choice(pool) for _ in range(15)]
                for r in RULES:
                    fails[r] += rule_fails(r, bs[:5], cs[:5], chans, req)
                f, rerun = two_stage(bs, cs, chans, req)
                fails["two_stage"] += f
                fails["rerun"] += rerun
            miss = {c: round(1 - sum(r[c] for r in pool) / len(pool), 3) for c in chans}
            grp["systems"][sy] = {"n": [len(ca_), len(cb_)], "miss_rate": miss, "required": req,
                                  "empirical_fail_AB_BA": emp, "gate_recall_fail_AB_BA": gate_fail,
                                  "same_seed_pairs": len(pairs), "same_seed_channel_disagreements": dis,
                                  "boot_fail": {r: fails[r] / boot for r in fails}}
        p_any = {r: 1 - math.prod(1 - v["boot_fail"][r] for v in grp["systems"].values())
                 for r in RULES + ("two_stage", "rerun")}
        grp["boot_fail_any_system"] = p_any
        out["groups"][name] = grp
    # power: one required channel, connected in every base run, true rate q in the candidate
    for r in RULES + ("two_stage",):
        out["power"][r] = {}
        for q in (0.8, 0.6, 0.4, 0.2, 0.0):
            hit = 0
            for _ in range(boot):
                base = [{"x": True} for _ in range(15)]
                cand = [{"x": rng.random() < q} for _ in range(15)]
                hit += two_stage(base, cand, ["x"], ["x"])[0] if r == "two_stage" else \
                    rule_fails(r, base[:5], cand[:5], ["x"], ["x"])
            out["power"][r][str(q)] = hit / boot
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
