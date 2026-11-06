"""Per (system, arm): max_species hits and when the queue stopped, wall vs the 1800 s budget (overruns > 2 % split by
stop kind: a budget stop = bug 4's steering points after the deadline, a max_species stop), the seed's proposal job
seconds, and the audit's closed_with_barrier / negative_barrier counts.

    python benchmarks/cluster/capreport.py LABEL=RESULTS_GLOB ...   (globs under the scoreboard results folder)
"""
import glob, json, os, statistics as st, sys
from collections import defaultdict

RES = "/home/user/astera/data/runs/explorer-team/scoreboard/results"
arms = [a.split("=") for a in sys.argv[1:]]          # label=run/glob-pattern-of-task-dirs
med = lambda xs: round(st.median(xs)) if xs else None
for label, pat in arms:
    print(f"== {label}")
    by = defaultdict(list)
    for d in sorted(glob.glob(f"{RES}/{pat}")):
        if not os.path.exists(d + "/score.json"):
            continue
        by[json.load(open(d + "/score.json"))["system"]].append(d)
    for system, dirs in sorted(by.items()):
        cap, walls, over_t, over_cap, prop, crash, closed, negb, fr = [], [], 0, 0, [], 0, 0, 0, 0
        for d in dirs:
            o = json.load(open(d + "/score.json"))
            ev = [json.loads(l) for l in open(d + "/network_aware.jsonl")] if os.path.exists(d + "/network_aware.jsonl") else []
            t0 = next((e["t"] for e in ev if e.get("phase") == "cycle_start"), None)
            stop = next((e["t"] for e in ev if e.get("event") == "horizontal_connect"), None)
            p = [e["run_s"] for e in ev if e.get("event") == "job" and e.get("kind") == "propose" and e.get("what") == "species 0"]
            prop += p
            if not o.get("ok"):
                crash += 1
            fr += bool(o.get("full_recall"))
            w = o.get("wall_s")
            if w:
                walls.append(w)
            if o.get("stopped") == "max species":
                cap.append(round(stop - t0) if (stop and t0) else None)
                over_cap += bool(w and w > 1.02 * 1800)
            elif w and w > 1.02 * 1800:
                over_t += 1
            fl = (o.get("audit") or {}).get("n_flags") or {}
            closed += fl.get("closed_with_barrier", 0)
            negb += fl.get("negative_barrier", 0)
        print(f"  {system:16s} n={len(dirs)} full={fr} crash={crash} cap_hits={len(cap)} at_s={sorted(x for x in cap if x)} "
              f"wall_med={med(walls)} max={round(max(walls)) if walls else None} overrun(>2%): budget-stop {over_t}, cap-stop {over_cap} "
              f"seed_propose_s med={med(prop)} closed_with_barrier={closed} negative_barrier={negb}")
