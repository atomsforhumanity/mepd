"""The amounts of named species at the end of a `mepd addmol test` result's run, without and with the molecule,
from the result's own rows on its query (the kinetics only; nothing searched).

    amounts_of.py QUERY.json RESULT.json SPECIES [SPECIES ...]
"""
import json
import sys


def main(query_fp, result_fp, *names):
    from mepd import addmol

    q = json.load(open(query_fp))
    d = json.load(open(result_fp))
    cond = q["conditions"]
    net = {"temperature": q["temperature"], "energies": q["energies"],
           "species": [{k: s.get(k) for k in ("id", "name", "g", "smiles")} for s in q["species"]],
           "steps": [{k: st.get(k) for k in ("id", "label", "reactants", "products", "g_ts", "status", "level")}
                     for st in q["steps"]]}
    variant = addmol.added(net, {"id": addmol.ADDED_ID, "name": d["candidate"]["name"]}, d["estimates"])
    out = {}
    for side, network, start in (("without", net, cond["initial"]),
                                 ("with", variant, {**cond["initial"], addmol.ADDED_ID: float(d["candidate"]["amount"])})):
        k = addmol.run(network, start, cond["time_s"], cond.get("held") or ())
        out[side] = {x: float(k["res"].c[-1, k["pos"][x]]) for x in names}
    print(json.dumps({"query": query_fp, "result": result_fp, "candidate": d["candidate"]["name"], **out}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
