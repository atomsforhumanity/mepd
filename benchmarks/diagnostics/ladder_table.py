"""Level and model ladder for a unimolecular selectivity (channels out of one reactant), one ingredient at a time.

    python ladder_table.py DEF.json OUT.json

DEF: {"temperature": K, "standard_state": "1atm", "ladder_dir": dft_ladder workdir, "extra_dirs": [dirs with more
      single points per structure (cluster runs: <name>.json records, their "sp" merged)],
      "ceiling": ceiling.py output (the explorer-level corrections and steering deltas),
      "reactant": ladder name of the floor conformer, "reactant_ceiling": label of the seed species in the ceiling,
      "channels": {product: [{"ts": ladder name, "ceiling": ceiling step label, "degeneracy": mirror count}]},
      "ratio": [num, den], "cc_levels": [single-point level names to use as the electronic energy, best last]}

Rows (kcal/mol barriers from the reactant floor; selectivity = sum_ch L exp(-dG/RT) ratio; RT ln of it):
 R0 explorer level: E(DFT SP @ g-xTB geometry) + g-xTB qRRHO (both from the ceiling's own numbers)
 R1 + DFT geometry: E(DFT @ DFT geometry), thermal still g-xTB's
 R2 + DFT thermal: qRRHO from DFT frequencies (FreeEnergies._one with the ORCA engine)
 R3.. electronic energy at each `cc_levels` level at the DFT geometry, DFT thermal
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HARTREE = 627.5094740631
R_KCAL = 1.987204259e-3


def load_records(d: Path, extra: list) -> dict:
    recs = {}
    for f in sorted(Path(d).glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(r, dict) and "name" in r:
            recs[r["name"]] = r
    for e in extra:
        for f in sorted(Path(e).rglob("*.json")):
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(r, dict) or "name" not in r:
                continue
            base = recs.setdefault(r["name"], {})
            for k, v in r.items():                 # fill what the main record lacks (a structure finished elsewhere)
                if k == "sp":
                    base.setdefault("sp", {}).update(v)
                elif k not in base:
                    base[k] = v
    return recs


def selectivity(barriers: dict, defs: dict, T: float) -> dict:
    rt = R_KCAL * T
    k = {}
    for prod, items in defs.items():
        k[prod] = sum(it.get("degeneracy", 1) * math.exp(-barriers[it["ts"]] / rt) for it in items)
    return k


def main(def_path: str, out_path: str):
    D = json.loads(Path(def_path).read_text())
    base = Path(def_path).resolve().parent
    T = float(D["temperature"])
    rt = R_KCAL * T
    recs = load_records(base / D["ladder_dir"], [base / e for e in D.get("extra_dirs", [])])
    ceil = json.loads((base / D["ceiling"]).read_text())["variants"]["steered+mirrors"]
    c_steps = {s["label"]: s for s in ceil["steps"]}
    c_seed = next(s for s in ceil["species"] if s["label"] == D["reactant_ceiling"])
    num, den = D["ratio"]
    R = recs[D["reactant"]]
    ts_names = [it["ts"] for items in D["channels"].values() for it in items]
    rows = []

    def row(name, barrier_of):
        b = {t: barrier_of(t) for t in ts_names}
        k = selectivity(b, D["channels"], T)
        r = k[num] / sum(k[x] for x in (den if isinstance(den, list) else [den]))
        rows.append({"row": name, "barriers": b, "selectivity": r, "rtln_kcal": rt * math.log(r)})

    # R0: the ceiling's own numbers (explorer level)
    def r0(t):
        lab = next(it["ceiling"] for items in D["channels"].values() for it in items if it["ts"] == t)
        return c_steps[lab]["dG_kcal"]

    row("R0 explorer level (g-xTB geometry + g-xTB qRRHO + wB97X-D3BJ/def2-TZVP SP)", r0)

    def corr_gxtb(t):
        lab = next(it["ceiling"] for items in D["channels"].values() for it in items if it["ts"] == t)
        return c_steps[lab]["corr_kcal"] - c_seed["corr_kcal"]

    def e_dft_opt(t):
        return (recs[t]["e_opt"] - R["e_opt"]) * HARTREE

    def corr_dft(t):
        return recs[t]["corr_kcal"] - R["corr_kcal"]

    row("R1 + DFT-optimized geometry (wB97X-D3BJ/def2-TZVP), g-xTB thermal", lambda t: e_dft_opt(t) + corr_gxtb(t))
    row("R2 + DFT frequencies (qRRHO from wB97X-D3BJ/def2-TZVP Hessians)", lambda t: e_dft_opt(t) + corr_dft(t))
    for lev in D.get("cc_levels", []):
        if lev not in R.get("sp", {}) or any(lev not in recs[t].get("sp", {}) for t in ts_names):
            rows.append({"row": f"R3 E = {lev} @ DFT geometry", "missing": True})
            continue
        row(f"R3 E = {lev} @ DFT geometry, DFT thermal",
            lambda t, lev=lev: (recs[t]["sp"][lev]["energy"] - R["sp"][lev]["energy"]) * HARTREE + corr_dft(t))
    out = {"temperature": T, "ratio": [num, den], "rows": rows,
           "components": {t: {"dE_dft_at_gxtb_geom": (c_steps[next(it["ceiling"] for items in D["channels"].values()
                                                                       for it in items if it["ts"] == t)]["dG_kcal"]
                                                      - corr_gxtb(t)),
                              "dE_dft_opt": e_dft_opt(t), "corr_gxtb": corr_gxtb(t), "corr_dft": corr_dft(t),
                              "imag_cm": recs[t].get("freqs_cm", [None])[0],
                              "sp": {lev: (v["energy"] - R["sp"][lev]["energy"]) * HARTREE
                                     for lev, v in recs[t].get("sp", {}).items() if lev in R.get("sp", {})}}
                          for t in ts_names}}
    Path(out_path).write_text(json.dumps(out, indent=1))
    prev = None
    for r in rows:
        if r.get("missing"):
            print(f"{r['row']:<85} (missing)")
            continue
        shift = "" if prev is None else f"  shift {r['rtln_kcal'] - prev:+.2f}"
        dl = "+".join(den) if isinstance(den, list) else den
        print(f"{r['row']:<85} " + " ".join(f"{t} {v:6.2f}" for t, v in r["barriers"].items())
              + f"  {num}:{dl} = {r['selectivity']:7.3g}  RTln {r['rtln_kcal']:5.2f}{shift}")
        prev = r["rtln_kcal"] if r["row"].startswith(("R0", "R1", "R2")) else prev
    for t, c in out["components"].items():
        print(t, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items() if k != "sp"},
              {k: round(v, 2) for k, v in c["sp"].items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
