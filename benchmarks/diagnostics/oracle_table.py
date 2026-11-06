"""Oracle test table: the ceiling with g-xTB, GFN2-xTB and UMA as the geometry/Hessian oracle, steering level fixed.

    python oracle_table.py SYSTEM_DIR WINDOW_LO WINDOW_HI|none [OUT.json]

Reads SYSTEM_DIR/ceiling_explorer_level.json (g-xTB oracle) and oracle_{gfn2,uma}/ceiling_explorer_level.json plus the
per-TS records oracle_*/<label>.json. Per TS: free barrier, its DFT electronic part (oracle barrier + steering shift)
and thermal part (qRRHO from the oracle's Hessian). Selectivity (RT ln, k∞ rate sums with each run's mirror twins) for
  full      the oracle's geometries and Hessians,
  geometry  the oracle's geometries for the DFT energies, g-xTB's thermal corrections,
  thermal   g-xTB's geometries for the DFT energies, the oracle's thermal corrections,
and against the g-xTB ceiling and the experimental window. Also survival (TS found, one imaginary mode, same IRC ends),
RMSD to the g-xTB TS, mirror-twin agreement at the oracle level, and the geometry-work cost.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

R_KCAL = 1.987204259e-3


def parts(cv):
    s0 = cv["species"][0]
    twins = {}
    for t in cv.get("mirrors", {}).get("added_twins", []):
        twins[t["step"]] = twins.get(t["step"], 0) + 1
    out = {}
    for s in cv["steps"]:
        el = s["dE_kcal"] + s["steer_delta_kcal"] - s0["steer_delta_kcal"]
        th = s["corr_kcal"] - s0["corr_kcal"]
        out[s["label"]] = {"dG": s["dG_kcal"], "el": el, "th": th, "L": 1 + twins.get(s["index"], 0), "ch": s["channel"]}
    return out


def rtln(p: dict, num, den, T, el_from=None, th_from=None):
    rt = R_KCAL * T
    dens = den if isinstance(den, list) else [den]
    k = {}
    for lab, x in p.items():
        el = (el_from or p)[lab]["el"] if lab in (el_from or p) else None
        th = (th_from or p)[lab]["th"] if lab in (th_from or p) else None
        if el is None or th is None:
            continue
        k[x["ch"]] = k.get(x["ch"], 0.0) + x["L"] * math.exp(-(el + th) / rt)
    a, b = k.get(num, 0.0), sum(k.get(d, 0.0) for d in dens)
    if a <= 0 or b <= 0:
        return None
    return rt * math.log(a / b)


def err(v, lo, hi):
    if v is None:
        return None
    if lo is not None and v < lo:
        return v - lo
    if hi is not None and v > hi:
        return v - hi
    return 0.0


def main(argv):
    d = Path(argv[0])
    lo = float(argv[1])
    hi = None if argv[2] == "none" else float(argv[2])
    base = json.loads((d / "ceiling_explorer_level.json").read_text())
    cv0 = base["variants"]["steered+mirrors"]
    T = cv0["temperature"]
    num, den = base["spec"]["ratio"]
    p0 = parts(cv0)
    out = {"system": d.name, "T": T, "window_rtln": [lo, hi], "oracles": {}}
    g0 = rtln(p0, num, den, T)
    out["oracles"]["gxtb"] = {"rtln": g0, "err_vs_expt": err(g0, lo, hi)}
    lines = [f"{d.name}: T {T} K, window {lo}..{hi}; g-xTB ceiling {g0:+.2f} (expt err {err(g0, lo, hi):+.2f})"]
    for K in ("gfn2", "uma"):
        f = d / f"oracle_{K}" / "ceiling_explorer_level.json"
        if not f.exists():
            continue
        cv = json.loads(f.read_text())["variants"]["steered+mirrors"]
        p = parts(cv)
        full = rtln(p, num, den, T)
        geo = rtln(p, num, den, T, el_from=p, th_from=p0)
        th = rtln(p, num, den, T, el_from=p0, th_from=p)
        recs = {}
        for r in (d / f"oracle_{K}").glob("*.json"):
            try:
                x = json.loads(r.read_text())
            except Exception:
                continue
            if isinstance(x, dict) and "label" in x and K in x:
                recs[x["label"]] = x
        ts_rows = {}
        for lab, x in recs.items():
            o = x[K]
            ratio = (abs(o.get("imag_cm") or 0) / abs(x.get("gxtb_imag_cm") or 1)) if x.get("gxtb_imag_cm") else None
            same_saddle = bool(x.get("survives") and ratio is not None and 0.5 < ratio < 1.5
                               and (x.get("rmsd_A") or 9) < 0.3)
            ts_rows[lab] = {"survives": x.get("survives"), "same_saddle": same_saddle, "n_imag": o.get("n_imag"), "imag_cm": o.get("imag_cm"),
                            "gxtb_imag_cm": x.get("gxtb_imag_cm"), "rmsd_A": x.get("rmsd_A"),
                            "same_connectivity": x.get("same_connectivity"), "energy": o.get("energy"),
                            "t_ts_irc_s": o.get("t_total"),
                            "gxtb_t_ts_irc_s": (x.get("gxtb_from_gfn2") or {}).get("t_total"),
                            "call": x.get(f"{K}_call"), "gxtb_call": x.get("gxtb_call"),
                            "dG": p.get(lab, {}).get("dG"), "dG_gxtb": p0.get(lab, {}).get("dG"),
                            "d_el": (p[lab]["el"] - p0[lab]["el"]) if lab in p and lab in p0 else None,
                            "d_th": (p[lab]["th"] - p0[lab]["th"]) if lab in p and lab in p0 else None}
        twin_spread = []
        for lab, x in recs.items():
            if "__twin" in lab and lab.split("__twin")[0] in recs:
                a, b = x[K].get("energy"), recs[lab.split("__twin")[0]][K].get("energy")
                if a is not None and b is not None:
                    twin_spread.append(abs(a - b) * 627.5094740631)
        cost = sum(r["t_ts_irc_s"] or 0 for r in ts_rows.values())
        cost_g = sum(r["gxtb_t_ts_irc_s"] or 0 for r in ts_rows.values())
        out["oracles"][K] = {"rtln": full, "rtln_geometry_only": geo, "rtln_thermal_only": th,
                             "vs_gxtb": None if full is None else full - g0, "err_vs_expt": err(full, lo, hi),
                             "n_ts": len(recs), "survived": sum(bool(r["survives"]) for r in ts_rows.values()),
                             "same_saddle": sum(bool(r["same_saddle"]) for r in ts_rows.values()),
                             "twin_spread_kcal_max": max(twin_spread) if twin_spread else None,
                             "ts_search_irc_s": cost, "gxtb_ts_search_irc_s_same_displacement": cost_g or None,
                             "ts": ts_rows}
        fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"
        lines.append(f"  {K}: full {fmt(full)} (vs g-xTB {fmt(None if full is None else full - g0)}, expt err "
                     f"{fmt(err(full, lo, hi))}); geometry only {fmt(geo)}, thermal only {fmt(th)}; "
                     f"TS survived {out['oracles'][K]['survived']}/{len(recs)}, same saddle {out['oracles'][K]['same_saddle']}/{len(recs)}; twin spread max "
                     f"{fmt(out['oracles'][K]['twin_spread_kcal_max'])}; TS+IRC wall {cost:.0f} s"
                     + (f" (g-xTB same displacement {cost_g:.0f} s)" if cost_g else ""))
        for lab in sorted(ts_rows):
            r = ts_rows[lab]
            lines.append(f"     {lab:<26} surv {r['survives']} same {r['same_saddle']} imag {r['imag_cm'] or 0:8.1f} (g-xTB {r['gxtb_imag_cm'] or 0:8.1f}) "
                         f"rmsd {r['rmsd_A'] if r['rmsd_A'] is not None else float('nan'):.3f}  dG {fmt(r['dG'])} "
                         f"(g-xTB {fmt(r['dG_gxtb'])}) d_el {fmt(r['d_el'])} d_th {fmt(r['d_th'])}")
    print("\n".join(lines))
    if len(argv) > 3:
        Path(argv[3]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
