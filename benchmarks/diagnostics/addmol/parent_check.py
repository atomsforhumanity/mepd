"""Does mepd.discovery.parent.network give the run's own molecule kinetics back?

    parent_check.py RUN_DIR [PROFILE]

For a finished `mepd discovery expand` run whose kinetics had open steps: its recorded molecule network
(summary.json rounds[-1].kinetics.molecules: names, g_kcal, reactions with g_ts_kcal and the step each came
from) beside the network mepd addmol builds from the run's files. Prints, for the molecules and for the
reactions both have, the largest difference of a barrier (forward or reverse); and the reactions only one of them has (the parent
network also holds the run's closed IRC-verified steps, so it has more).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(run_dir, profile=None):
    from mepd.discovery import parent

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    rec = summary["rounds"][-1]["kinetics"]["molecules"]
    labels = [st["label"] for st in summary["steps"]]
    engine = None
    if profile:                            # (else only what the run recorded)
        from mepd.cli_common import _open_run_inputs

        engine = _open_run_inputs(Path(profile)).engine
    net = parent.network(run_dir, engine, workers=4)
    g = {s["id"]: s["g"] for s in net["species"]}
    mine = {st["id"]: st for st in net["steps"]}
    both = [n for n in rec["names"] if n in g]
    rows, only_run = [], []
    for x in rec["reactions"]:
        sid = labels[x["step"]] + ("~mirror" if x["twin"] else "")
        st = mine.get(sid)
        if st is None:
            only_run.append(sid)
            continue
        same = sorted(st["reactants"]) == sorted(rec["names"][i] for i in x["r"]) and \
            sorted(st["products"]) == sorted(rec["names"][i] for i in x["p"])
        # barriers, not the TS's own number: a molecule's free energy is fixed only up to what no reaction of
        # the network changes (the fit may share a box's free energy among its molecules either way)
        fwd = (st["g_ts"] - sum(g[i] for i in st["reactants"])) - (x["g_ts_kcal"] - sum(rec["g_kcal"][i] for i in x["r"]))
        rev = (st["g_ts"] - sum(g[i] for i in st["products"])) - (x["g_ts_kcal"] - sum(rec["g_kcal"][i] for i in x["p"]))
        if sorted(st["reactants"]) != sorted(rec["names"][i] for i in x["r"]):
            fwd, rev = rev, fwd
        rows.append((sid, max(abs(fwd), abs(rev)), same, fwd, rev))
    ran = {r[0] for r in rows}
    print(json.dumps({
        "run": {"molecules": len(rec["names"]), "reactions": len(rec["reactions"])},
        "parent": {"molecules": len(net["species"]), "reactions": len(net["steps"])},
        "molecules_in_both": len(both),
        "reactions_in_both": len(rows), "same_reactants_and_products": sum(1 for r in rows if r[2]),
        "largest_barrier_difference_kcal": max((r[1] for r in rows), default=None),
        "barrier_differences_over_0.01": [(r[0], round(r[3], 3), round(r[4], 3)) for r in rows if r[1] > 0.01][:40],
        "only_in_run": only_run[:20], "only_in_parent": len([s for s in mine if s not in ran]),
        "parent_notes": net["notes"], "fit": {"run": rec.get("fit"), "parent": net["fit"]}}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
