"""Across systems: oracle_table.json of each -> one table (RT ln vs experiment and vs the g-xTB ceiling, survival,
geometry-work cost per oracle), and the per-call costs (median over the TSs).

    python oracle_summary.py TRACING_DIR SYSTEM [SYSTEM ...]
"""
import json
import statistics
import sys
from pathlib import Path


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main(argv):
    T = Path(argv[0])
    cpu = json.loads((T / "uma_cpu_timing.json").read_text()) if (T / "uma_cpu_timing.json").exists() else {}
    rows = []
    for s in argv[1:]:
        t = json.loads((T / s / "oracle_table.json").read_text())
        g = t["oracles"]["gxtb"]
        line = {"system": s, "window": t["window_rtln"], "gxtb": g["rtln"], "gxtb_err": g["err_vs_expt"]}
        for K in ("gfn2", "uma"):
            o = t["oracles"].get(K)
            if not o:
                continue
            ts = o["ts"].values()
            line[K] = {"rtln": o["rtln"], "vs_gxtb": o["vs_gxtb"], "err": o["err_vs_expt"],
                       "geometry_only_vs_gxtb": None if o["rtln_geometry_only"] is None else o["rtln_geometry_only"] - g["rtln"],
                       "thermal_only_vs_gxtb": None if o["rtln_thermal_only"] is None else o["rtln_thermal_only"] - g["rtln"],
                       "survived": f"{o['survived']}/{o['n_ts']}", "same_saddle": f"{o['same_saddle']}/{o['n_ts']}",
                       "ts_search_irc_s": o["ts_search_irc_s"], "rmsd_median_A": med([r["rmsd_A"] for r in ts]),
                       "twin_spread_max": o["twin_spread_kcal_max"],
                       "egrad_s": med([(r["call"] or {}).get("energy_gradient_s") for r in ts]),
                       "hess_s": med([(r["call"] or {}).get("hessian_s") for r in ts]),
                       "gxtb_egrad_s": med([(r["gxtb_call"] or {}).get("energy_gradient_s") for r in ts]),
                       "gxtb_hess_s": med([(r["gxtb_call"] or {}).get("hessian_s") for r in ts])}
        c = next((v for k, v in cpu.items() if k.split(":")[0] == s), None)
        if c:
            line["uma_cpu"] = {"ts_search_irc_s": c["search"].get("t_total"), "hess_s": c["call"]["hessian_s"]}
        rows.append(line)
    print(json.dumps(rows, indent=1))
    (T / "oracle_summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
