"""Two `mepd addmol test` results of one candidate on one parent (two targets, so different reactions scanned):
the kinetics with the rows of both, each TS once (mepd.cli_addmol._same_ts), and the amounts of named species.

    merged_amounts.py QUERY.json RESULT_A.json RESULT_B.json SPECIES [SPECIES ...]
"""
import json
import sys


def main(query_fp, a_fp, b_fp, *names):
    from mepd import addmol
    from mepd.cli_addmol import _same_ts

    q = json.load(open(query_fp))
    a, b = json.load(open(a_fp)), json.load(open(b_fp))
    est = json.loads(json.dumps(a["estimates"]))
    rows = [n for n in est["new_steps"]]
    added = []
    for n in b["estimates"]["new_steps"]:
        if n.get("mirror_image_of"):
            continue
        if any(n["id"] == m["id"] for m in rows) or _same_ts([m for m in rows if not m.get("mirror_image_of")],
                                                               {**n, "layer": "other"}):
            continue
        rows.append(n)
        added.append(n["id"])
        rows += [m for m in b["estimates"]["new_steps"] if m.get("mirror_image_of") == n["id"]]
    est["new_steps"] = rows
    cond = q["conditions"]
    net = {"temperature": q["temperature"], "energies": q["energies"],
           "species": [{k: s.get(k) for k in ("id", "name", "g", "smiles")} for s in q["species"]],
           "steps": [{k: st.get(k) for k in ("id", "label", "reactants", "products", "g_ts", "status", "level")}
                     for st in q["steps"]]}
    out = {}
    for side, e in (("without", None), ("with", est)):
        network = net if e is None else addmol.added(net, {"id": addmol.ADDED_ID, "name": a["candidate"]["name"]}, e)
        start = cond["initial"] if e is None else {**cond["initial"], addmol.ADDED_ID: float(a["candidate"]["amount"])}
        k = addmol.run(network, start, cond["time_s"], cond.get("held") or ())
        out[side] = {x: float(k["res"].c[-1, k["pos"][x]]) for x in names}
    print(json.dumps({"a": a_fp, "b": b_fp, "rows_of_b_added": added, **out}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
