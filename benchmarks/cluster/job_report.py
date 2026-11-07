"""Per arm and system: TS-search and IRC job failures, negative-barrier flags, how much of the final product the
scored channels get, and the open steps without an IRC check or with a TS below the seed.

    python benchmarks/cluster/job_report.py RESULT:CONFIG [RESULT:CONFIG ...] [--only a,b]

- jobs (network_aware.jsonl, score.queue_jobs; jobs the time budget cancelled are left out): TS search = kinds
  fast, ts, rotamer, channels; IRC = kind irc; done / failed, summed over the arm's runs.
- negative_barrier: the audit's flags (summed).
- product: final amounts (score.folded_amounts): converted = 1 - the seed's amount; channel share = the scored
  channels' species' amount / converted (median and min over runs).
- open steps (a kinetics barrier): `unverified` = no IRC check (files.verified not true); `below_seed` = TS below
  the seed on the energies the kinetics used (summary access: ts_energies vs energies[0]; else ts_energy vs
  seed_energy); `both`; summed over runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402
from recall_noise import RESULTS, runs_of  # noqa: E402

TS_KINDS = {"fast", "ts", "rotamer", "channels"}


def run_facts(o: dict) -> dict | None:
    d = RESULTS / o["run"] / o["id"]
    if not o.get("ok") or not (d / "run" / "summary.json").is_file():
        return None
    s = json.loads((d / "run" / "summary.json").read_text())
    events = []
    na = d / "network_aware.jsonl"
    if na.is_file():
        for line in na.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    failed, cancelled = score.queue_jobs(na, o.get("budget_s"))
    cancelled_ids = {id(e) for e in cancelled}
    jobs = [e for e in events if e.get("event") == "job"]
    done = {"ts": sum(e.get("kind") in TS_KINDS for e in jobs), "irc": sum(e.get("kind") == "irc" for e in jobs)}
    fail = {"ts": sum(e.get("kind") in TS_KINDS for e in failed), "irc": sum(e.get("kind") == "irc" for e in failed)}
    am = score.folded_amounts(s) or {}
    conv = 1.0 - am.get(0, (0.0, 0.0))[0] if am else None
    chan = {k for ks in (o["recall"].get("species_of") or {}).values() for k in ks}
    share = (sum(am.get(k, (0.0, 0.0))[0] for k in chan) / conv) if conv and conv > 1e-12 else None
    acc = s.get("access")
    steps = s.get("steps") or []
    if acc:
        e0, ts = acc["energies"][0], acc["ts_energies"]
    else:
        e0, ts = s.get("seed_energy"), [st.get("ts_energy") for st in steps]
    unver = below = both = 0
    for m, st in enumerate(steps):
        if st.get("kinetics_barrier_kcal") is None:
            continue
        u = not (st.get("files") or {}).get("verified")
        b = e0 is not None and ts[m] is not None and float(ts[m]) < float(e0)
        unver, below, both = unver + u, below + b, both + (u and b)
    return {"done": done, "failed": fail, "neg": len(((o.get("audit") or {}).get("flags") or {}).get("negative_barrier") or []),
            "converted": conv, "share": share, "unverified": unver, "below_seed": below, "both": both}


def main(argv: list) -> int:
    only = None
    if "--only" in argv:
        k = argv.index("--only")
        only, argv = set(argv[k + 1].split(",")), argv[:k] + argv[k + 2:]
    out = {}
    for spec in argv:
        res, cfg = spec.split(":", 1)
        arm = {}
        for (sy, rep), o in sorted(runs_of(res, cfg).items()):
            if only and sy not in only:
                continue
            f = run_facts({**o, "run": o.get("run") or res})
            g = arm.setdefault(sy, {"n": 0, "ts_jobs": 0, "ts_failed": 0, "irc_jobs": 0, "irc_failed": 0,
                                    "negative_barrier": 0, "shares": [], "unverified_open": 0,
                                    "below_seed_open": 0, "unverified_and_below_seed_open": 0})
            g["n"] += 1
            if f is None:
                continue
            g["ts_jobs"] += f["done"]["ts"]
            g["ts_failed"] += f["failed"]["ts"]
            g["irc_jobs"] += f["done"]["irc"]
            g["irc_failed"] += f["failed"]["irc"]
            g["negative_barrier"] += f["neg"]
            g["shares"].append(None if f["share"] is None else round(f["share"], 4))
            g["unverified_open"] += f["unverified"]
            g["below_seed_open"] += f["below_seed"]
            g["unverified_and_below_seed_open"] += f["both"]
        for g in arm.values():
            xs = sorted(x for x in g["shares"] if x is not None)
            g["channel_share_median"] = xs[len(xs) // 2] if xs else None
            g["channel_share_min"] = xs[0] if xs else None
        out[spec] = arm
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
