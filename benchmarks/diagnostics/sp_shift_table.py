"""Single-point rungs of a level ladder at the explorer's (g-xTB) geometries: each level's electronic barrier replaces
the steering level's (wB97X-D3BJ/def2-TZVP) in the ceiling's free barriers, everything else as the explorer has it.

    python sp_shift_table.py DEF.json OUT.json

DEF: {"ceiling": ceiling.py output, "cache": the ceiling's cache, "sp_dir": dft_ladder workdir of "sp" records at the
      g-xTB geometries, "seed_xyz": the ceiling's seed species file (its cache entry is the reference, molecules apart),
      "seed": [sp record names whose energies add up to the seed's, molecules apart], "steps": {ceiling step label:
      sp record name}, "ratio": [num, den | [dens]], "degeneracy": {step label: mirror count (default 2 if the ceiling
      added a twin, else 1)}}
Selectivity from sum_channel L exp(-dG/RT) with each step's channel as in the ceiling.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from explorer_level import HARTREE, R_KCAL, geo_key, load  # noqa: E402


def main(def_path: str, out_path: str):
    D = json.loads(Path(def_path).read_text())
    base = Path(def_path).resolve().parent
    cv = json.loads((base / D["ceiling"]).read_text())["variants"]["steered+mirrors"]
    cache = json.loads((base / D["cache"]).read_text())
    T = cv["temperature"]
    rt = R_KCAL * T
    twins = {}
    for t in cv.get("mirrors", {}).get("added_twins", []):
        twins[t["step"]] = twins.get(t["step"], 0) + 1
    steps = {s["label"]: s for s in cv["steps"]}
    L = {lab: D.get("degeneracy", {}).get(lab, 1 + twins.get(s["index"], 0)) for lab, s in steps.items()}
    ch = {lab: s["channel"] for lab, s in steps.items()}
    num, den = D["ratio"]
    dens = den if isinstance(den, list) else [den]
    recs = {}
    for f in (base / D["sp_dir"]).glob("*.json"):
        r = json.loads(f.read_text())
        recs[r["name"]] = r
    ref_seed = cache["sp:" + geo_key(load(base / D["seed_xyz"]), (False,))]["e_high"]

    def ts_ref(lab):
        st = load(base / [x for x in json.loads((base / D["spec"]).read_text())["steps"] if x["label"] == lab][0]["xyz"])
        return cache["sp:" + geo_key(st, (True,))]["e_high"]

    rows = []

    def add(name, b):
        k = {}
        for lab, g in b.items():
            k[ch[lab]] = k.get(ch[lab], 0.0) + L[lab] * math.exp(-g / rt)
        r = k.get(num, 0.0) / sum(k.get(d, 0.0) for d in dens)
        rows.append({"row": name, "barriers": b, "selectivity": r, "rtln_kcal": rt * math.log(r)})

    b0 = {lab: s["dG_kcal"] for lab, s in steps.items()}
    add("R0 explorer level (ceiling)", b0)
    levels = sorted({lv for r in recs.values() for lv in r.get("sp", {})})
    for lev in levels:
        names = list(D["seed"]) + list(D["steps"].values())
        if any(lev not in recs.get(n, {}).get("sp", {}) for n in names):
            continue
        e_seed = sum(recs[n]["sp"][lev]["energy"] for n in D["seed"])
        b = {}
        for lab, n in D["steps"].items():
            shift = ((recs[n]["sp"][lev]["energy"] - e_seed) - (ts_ref(lab) - ref_seed)) * HARTREE
            b[lab] = b0[lab] + shift
        add(f"R0 + ({lev} - wB97X-D3BJ) electronic barriers at g-xTB geometries", b)
    Path(out_path).write_text(json.dumps({"temperature": T, "rows": rows, "degeneracy": L}, indent=1))
    for r in rows:
        print(f"{r['row']:<88} " + " ".join(f"{k[3:]} {v:6.2f}" for k, v in r["barriers"].items())
              + f"  sel {r['selectivity']:8.4g}  RTln {r['rtln_kcal']:+.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
