"""How reproducible are the recorded thermal corrections across runs?

    python benchmarks/diagnostics/thermo/spread.py --out DIR  RUNSET_OUTPUT_DIR ...

Reads every <run>/summary.json under the given folders (run folders named <system>-<config>-r<k>). Per system,
the TSs (steps with a recorded `ts_energy_terms.thermal_correction_kcal`) and the species (`energy_terms`) are
grouped by what they are (a TS: its step's reactants, products and spectators as the run recorded them, either
direction; a species: its SMILES) and then by electronic energy: values chained while neighbours are within
--tolerance (1e-6 Eh). Energy alone is not enough: among thousands of TSs two different ones agree to 1e-6 Eh by
chance. A group seen in two or more runs is "the same structure found again"; its spread is max - min of the
recorded corrections. Mirror images share an energy and a correction, so they fall in one group. Reads records
only; computes nothing.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def groups(rows, tol):
    rows = sorted(rows, key=lambda r: r["e"])
    out, cur = [], []
    for r in rows:
        if cur and r["e"] - cur[-1]["e"] > tol:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def what(step, smiles):
    """A step's identity as the run recorded it, the same for both directions."""
    r = step.get("reacting")
    if r:
        ends = sorted([".".join(sorted(r.get("reactants") or [])), ".".join(sorted(r.get("products") or []))])
        return " <-> ".join(ends) + " | " + ".".join(sorted(s["key"] for s in r.get("spectators") or []))
    return " <-> ".join(sorted([str(smiles.get(step["a"])), str(smiles.get(step["b"]))]))


def table(rows, tol):
    by = defaultdict(list)
    for r in rows:
        by[(r["system"], r["temperature"], r["standard_state"], r["what"])].append(r)
    out = []
    for (system, T, ss, _), rs in sorted(by.items(), key=repr):
        for g in groups(rs, tol):
            runs = sorted({r["run"] for r in g})
            c = [r["corr"] for r in g]
            lo, hi = min(g, key=lambda r: r["corr"]), max(g, key=lambda r: r["corr"])
            out.append({"system": system, "temperature": T, "energy": sum(r["e"] for r in g) / len(g),
                        "energy_range_Eh": g[-1]["e"] - g[0]["e"], "n": len(g), "n_runs": len(runs),
                        "spread_kcal": max(c) - min(c), "min": min(c), "max": max(c),
                        "min_at": f"{lo['run']}:{lo['label']}", "max_at": f"{hi['run']}:{hi['label']}",
                        "open_in": sum(1 for r in g if r.get("open")),
                        "in_window": sum(1 for r in g if r.get("window")), "what": g[0].get("what"),
                        "members": [{k: r.get(k) for k in ("run", "dir", "label", "a", "b", "e", "corr", "open", "window", "reacting_atoms",
                                              "verified")}
                                    for r in g]})
    return out


def digest(tab, name):
    multi = [t for t in tab if t["n_runs"] >= 2]
    lines = [f"## {name}: {len(tab)} groups, {len(multi)} seen in >= 2 runs"]
    edges = [0.0, 0.01, 0.05, 0.2, 0.5, 1.0, 1e9]
    for lo, hi in zip(edges, edges[1:]):
        k = [t for t in multi if lo <= t["spread_kcal"] < hi]
        lines.append(f"- spread [{lo:g}, {hi:g}) kcal/mol: {len(k)}" +
                     (f" (open in a run's kinetics: {sum(1 for t in k if t['open_in'])}; in a deciding window: "
                      f"{sum(1 for t in k if t['in_window'])})" if name == "TSs" else ""))
    lines.append("")
    lines.append("| system | what | E (Eh) | entries | runs | spread | min | max | lowest at | highest at |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(multi, key=lambda t: -t["spread_kcal"])[:40]:
        lines.append(f"| {t['system']} | {t['what']} | {t['energy']:.6f} | {t['n']} | {t['n_runs']} | {t['spread_kcal']:.3f} | "
                     f"{t['min']:.3f} | {t['max']:.3f} | {t['min_at']} | {t['max_at']} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ts_rows, sp_rows, runs = [], [], []
    for folder in a.folders:
        for fp in sorted(Path(folder).glob("*/summary.json")):
            run = fp.parent.name
            m = re.match(r"(.+?)-(lin2.*)-r(\d+)$", run)
            system = m.group(1) if m else run
            s = json.loads(fp.read_text())
            st = s.get("steering") or {}
            T, ss = st.get("temperature_K"), st.get("standard_state")
            n_ts = n_sp = 0
            smiles = {sp["index"]: sp.get("smiles") for sp in s.get("species", [])}
            for step in s.get("steps", []):
                c = (step.get("ts_energy_terms") or {}).get("thermal_correction_kcal")
                if c is None or step.get("ts_energy") is None:
                    continue
                n_ts += 1
                ts_rows.append({"system": system, "run": run, "dir": str(fp.parent), "label": step["label"],
                                "a": step["a"], "b": step["b"], "e": float(step["ts_energy"]),
                                "reacting_atoms": (step.get("reacting") or {}).get("atoms"),
                                "verified": (step.get("files") or {}).get("verified"),
                                "corr": float(c), "temperature": T, "standard_state": ss,
                                "open": step.get("closed") is None and step.get("kinetics_barrier_kcal") is not None,
                                "window": bool(step.get("deciding_window")),
                                "what": what(step, smiles)})
            for sp in s.get("species", []):
                c = (sp.get("energy_terms") or {}).get("thermal_correction_kcal")
                if c is None or sp.get("energy") is None:
                    continue
                n_sp += 1
                sp_rows.append({"system": system, "run": run, "label": f"species_{sp['index']}",
                                "e": float(sp["energy"]), "corr": float(c), "temperature": T, "standard_state": ss,
                                "what": sp.get("smiles")})
            runs.append({"run": run, "system": system, "ts_with_correction": n_ts, "species_with_correction": n_sp})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tabs = {"TSs": table(ts_rows, a.tolerance), "species": table(sp_rows, a.tolerance)}
    (out / "spread.json").write_text(json.dumps({"tolerance_Eh": a.tolerance, "runs": runs, **tabs}, indent=1))
    text = f"{len(runs)} runs read.\n\n" + "\n\n".join(digest(t, n) for n, t in tabs.items()) + "\n"
    (out / "spread.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
