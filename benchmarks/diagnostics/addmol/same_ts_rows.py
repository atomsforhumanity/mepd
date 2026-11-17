"""Rows of a `mepd addmol test` result that are one TS found by two layers (mepd.cli_addmol._same_ts), and the
read-out with each counted once (the later one dropped, with its mirror copy).

    same_ts_rows.py QUERY.json RESULT.json [SPECIES ...]
"""
import json
import sys


def main(query_fp, result_fp, *names):
    from mepd import addmol
    from mepd.cli_addmol import _same_ts

    q = json.load(open(query_fp))
    d = json.load(open(result_fp))
    rows = sorted(d["estimates"]["new_steps"], key=lambda n: n.get("t_s") or 0.0)
    kept, dropped = [], []
    for n in rows:
        if n.get("mirror_image_of"):
            continue
        again = _same_ts([k for k in kept if k.get("layer") != n.get("layer")], n)
        (dropped if again else kept).append(n if not again else {"row": n["id"], "same_as": again,
                                                                 "barrier": n["barrier"]["free"]})
    gone = {x["row"] for x in dropped} | {x["row"] + "~mirror" for x in dropped}
    est = {**d["estimates"], "new_steps": [n for n in d["estimates"]["new_steps"] if n["id"] not in gone]}
    out = {}
    for side, e in (("as_written", d["estimates"]), ("each_ts_once", est)):
        r = addmol.result(q, d["candidate"], e)["readout"]
        out[side] = {"target": r["target"]["with"]["final"], "route_peak": r["peak_routes"]["with"][0]["peak"],
                     "fate": r["added"]["in_words"]}
        if names:
            cond = q["conditions"]
            net = {"temperature": q["temperature"], "energies": q["energies"],
                   "species": [{k: s.get(k) for k in ("id", "name", "g", "smiles")} for s in q["species"]],
                   "steps": [{k: st.get(k) for k in ("id", "label", "reactants", "products", "g_ts", "status", "level")}
                             for st in q["steps"]]}
            v = addmol.added(net, {"id": addmol.ADDED_ID, "name": d["candidate"]["name"]}, e)
            k = addmol.run(v, {**cond["initial"], addmol.ADDED_ID: float(d["candidate"]["amount"])}, cond["time_s"])
            out[side]["amounts"] = {x: float(k["res"].c[-1, k["pos"][x]]) for x in names}
    print(json.dumps({"result": result_fp, "candidate": d["candidate"]["name"], "rows": len(rows),
                      "found_twice": dropped, **out}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
