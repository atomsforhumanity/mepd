"""F0007 (Cp + methyl acrylate, endo:exo) level ladder, one ingredient at a time, from the ceiling and the F0007
Finding's DFT re-optimization.

    python f0007_table.py TRACING_F0007_DIR OUT.json

Rows: free barriers (kcal/mol, 353.15 K, 1 M, from Cp + MA apart) of the four Diels-Alder TS conformers
(endo/exo x s-cis/s-trans; every one chiral, so mirror images double each channel alike) and
endo:exo = sum_endo exp(-dG/RT) / sum_exo exp(-dG/RT).
 R0  explorer level (ceiling: g-xTB geometry, g-xTB qRRHO, wB97X-D3BJ/def2-TZVP SP), all four TSs
 R0e the same with the three TSs the explorer's network had (no exo s-trans)
 R1  + DFT-optimized geometries (the Finding's wB97X-D3BJ/def2-TZVP re-optimization: dE(opt) - dE(SP at g-xTB))
 R2  + DFT thermal corrections (the Finding's qRRHO from DFT frequencies - g-xTB's)
 S*  each single-point level's shift of the electronic barrier, at the g-xTB geometries (sp_gxtb_geom), on top of R2:
     CPCM(benzene) solvation, other functionals, DLPNO-CCSD(T) (when present)
The exo s-trans TS is not in the Finding: its R1/R2 shifts are taken as exo s-cis's (said in the output).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

HARTREE = 627.5094740631
R_KCAL = 1.987204259e-3
# the F0007 Finding's data/orca_free_energies.csv
FINDING = Path(os.environ.get("MEPD_F0007_FREE_ENERGIES") or "orca_free_energies.csv")
TS = ["ts_endo_scis", "ts_endo_strans", "ts_exo_scis", "ts_exo_strans"]
FIND_ROW = {"ts_endo_scis": "0->34", "ts_endo_strans": "58->0", "ts_exo_scis": "61->0"}


def ratio(b: dict, T: float, which=TS) -> float:
    rt = R_KCAL * T
    endo = sum(math.exp(-b[t] / rt) for t in which if "endo" in t)
    exo = sum(math.exp(-b[t] / rt) for t in which if "exo" in t)
    return endo / exo


def main(d: str, out: str):
    d = Path(d)
    ceil = json.loads((d / "ceiling_explorer_level.json").read_text())["variants"]["steered+mirrors"]
    T = ceil["temperature"]
    rt = R_KCAL * T
    r0 = {s["label"].replace("TS_", "ts_"): s["dG_kcal"] for s in ceil["steps"]}
    rows = []

    def add(name, b, which=TS, note=""):
        r = ratio(b, T, which)
        rows.append({"row": name, "barriers": {t: b[t] for t in which}, "endo:exo": r,
                     "endo_percent": 100 * r / (1 + r), "rtln_kcal": rt * math.log(r), "note": note})

    add("R0 explorer level, 4 TS conformers", r0)
    add("R0e explorer level, the 3 TSs the explorer found (no exo s-trans)", r0, TS[:3])
    fr = {r["what"]: r for r in csv.DictReader(FINDING.open()) if r["kind"] == "ts"}
    geo, th = {}, {}
    for t in TS:
        f = fr[FIND_ROW.get(t, FIND_ROW["ts_exo_scis"])]
        geo[t] = float(f["de_orca_opt_kcal"]) - float(f["de_orca_sp_kcal"])
        th[t] = float(f["thermal_dft_kcal"]) - float(f["thermal_gxtb_kcal"])
    r1 = {t: r0[t] + geo[t] for t in TS}
    add("R1 + DFT-optimized geometry (Finding)", r1, note="exo s-trans: exo s-cis's shift")
    r2 = {t: r1[t] + th[t] for t in TS}
    add("R2 + DFT frequencies (Finding)", r2, note="exo s-trans: exo s-cis's shift")
    # single-point shifts at g-xTB geometries
    recs = {}
    for f in (d / "sp_gxtb_geom").glob("*.json"):
        r = json.loads(f.read_text())
        recs[r["name"]] = r
    cache = json.loads((d / "ceiling_cache.json").read_text())
    import sys as _s

    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from explorer_level import geo_key, load

    def e_ref(name, path, ts):
        st = load(d / path)
        hit = cache.get("sp:" + geo_key(st, (ts,)))
        return None if hit is None else hit["e_high"]

    spec = json.loads((d / "sp_gxtb_spec.json").read_text())
    base = {n: e_ref(n, s["xyz"], True) for n, s in spec["structures"].items() if n.startswith("ts")}
    seed = e_ref("seed", "structs/seed_gxtb.xyz", False)     # Cp + MA apart, each at its geometry in the complex
    base["cp"], base["ma"] = seed, 0.0
    levels = [lv["name"] for lv in spec["sp"]] + ["DLPNO-CCSD(T)/def2-TZVP"]
    shifts = {}
    for lev in levels:
        if any(lev not in recs.get(n, {}).get("sp", {}) for n in spec["structures"]) or None in base.values():
            rows.append({"row": f"S {lev}", "missing": True})
            continue
        e = {n: recs[n]["sp"][lev]["energy"] for n in spec["structures"]}
        sh = {t: ((e[t] - e["cp"] - e["ma"]) - (base[t] - base["cp"] - base["ma"])) * HARTREE for t in TS}
        shifts[lev] = sh
        add(f"S R2 + ({lev} - wB97X-D3BJ) at g-xTB geometries", {t: r2[t] + sh[t] for t in TS})
    out_d = {"temperature": T, "rows": rows, "geometry_shift": geo, "thermal_shift": th, "sp_shifts": shifts,
             "experiment": {"endo:exo": 77 / 23, "rtln_kcal": rt * math.log(77 / 23)}}
    Path(out).write_text(json.dumps(out_d, indent=1))
    for r in rows:
        if r.get("missing"):
            print(f"{r['row']:<70} (missing)")
            continue
        print(f"{r['row']:<70} " + " ".join(f"{t[3:]} {v:6.2f}" for t, v in r["barriers"].items())
              + f"  endo:exo {r['endo_percent']:5.1f}:{100 - r['endo_percent']:4.1f}  RTln {r['rtln_kcal']:+.2f}")
    print(f"experiment 77:23 RTln {rt * math.log(77 / 23):+.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
