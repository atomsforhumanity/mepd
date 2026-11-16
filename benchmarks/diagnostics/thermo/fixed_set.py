"""What converging a TS before its Hessian changes in the energetics fixed set (a diagnostic of the change).

    python benchmarks/diagnostics/thermo/fixed_set.py PROFILE SET_DIR OUT_DIR [WORKERS]

SET_DIR: an energetics fixed set (a folder with set.json and its structures; the set is not in this repository yet).
For every structure of SET_DIR/set.json, at its system's temperature and standard state, the thermal correction of
the molecules it counts as (`parts`; a TS's first part carries its reaction mode), by the explorer's own
FreeEnergies._one: `given` with the Hessian at the geometry in the set (FreeEnergies._saddle switched off, the
code before) and `converged` as the code is. Also each TS's largest atomic force, how far converging moved it and
its lowest real mode at both points. Judges nothing: the difference is what the change moves.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

BOHR = 1.8897259886
_PROFILE = None


def one(job):
    sid, kind, T, ss, symbols, xyz, parts = job
    import mepd
    from qcdata import Structure

    import mepd.discovery.free_energy as fe
    from mepd.nodes.node import StructureNode

    out = {"id": sid, "kind": kind, "temperature_K": T, "molecules": [len(p) for p in parts]}
    try:
        saddle = fe.FreeEnergies._saddle
        for name in ("given", "converged"):
            fe.FreeEnergies._saddle = (lambda self, node: (node, 0.0)) if name == "given" else saddle
            free = fe.FreeEnergies(mepd.engine_from_profile(_PROFILE), T, ss)
            out[name] = sum(free._one([symbols[i] for i in p], xyz[p], 0, 1, kind == "ts" and k == 0,
                                      kind == "ts" and k == 0)
                            for k, p in enumerate(parts))
            out[name + "_notes"] = list(free.notes)
        if kind == "ts":
            p = parts[0]
            node = StructureNode(structure=Structure(symbols=[symbols[i] for i in p], geometry=xyz[p] * BOHR, charge=0,
                                                     multiplicity=1))
            free = fe.FreeEnergies(mepd.engine_from_profile(_PROFILE), T, ss)
            g = np.asarray(free.engine.compute_gradients([node])[0], dtype=float).reshape(-1, 3)
            out["fmax_eV_A"] = float(np.linalg.norm(g, axis=1).max()) * fe.EV_PER_HARTREE * BOHR
            tight, moved = free._saddle(node)
            out["energy_change_kcal"] = moved
            out["moved_max_A"] = float(np.linalg.norm((np.asarray(tight.structure.geometry) -
                                                       np.asarray(node.structure.geometry)).reshape(-1, 3),
                                                      axis=1).max()) / BOHR
            for name, nd in (("given", node), ("converged", tight)):
                w = sorted(mepd.frequencies(nd, free.engine).wavenumbers_cm.tolist())
                out[f"lowest_cm_{name}"] = w[:3]
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def read_xyz(fp):
    lines = Path(fp).read_text().splitlines()
    n = int(lines[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def main(profile, set_dir, out_dir, workers="1"):
    global _PROFILE
    _PROFILE = profile
    set_dir, out_dir = Path(set_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = json.loads((set_dir / "set.json").read_text())
    jobs = []
    for s in sets["systems"].values():
        for kind, ent in [("reactant", s["reactant"])] + [("ts", t) for t in s["ts"]]:
            sym, X = read_xyz(set_dir / ent["file"])
            jobs.append((ent["id"], kind, float(s["temperature_K"]), s["standard_state"], sym, X, ent["parts"]))
    with get_context("fork").Pool(int(workers)) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    (out_dir / "fixed_set.json").write_text(json.dumps({"set": sets.get("version"), "rows": rows}, indent=1))
    lines = ["| structure | T (K) | molecules | fmax (eV/A) | given | converged | change | of which energy | moved (A) | "
             "lowest real mode given -> converged (cm-1) | notes |", "|" + "---|" * 11]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['id']} | {r['temperature_K']:g} | | | | | | | | | {r['error']} |")
            continue
        ts = r["kind"] == "ts"

        def low(w):
            return next((f"{x:.1f}" for x in w[1:]), "-")
        lines.append(f"| {r['id']} | {r['temperature_K']:g} | {'+'.join(map(str, r['molecules']))} | "
                     f"{(f'{r['fmax_eV_A']:.5f}') if ts else ''} | {r['given']:.4f} | {r['converged']:.4f} | "
                     f"{r['converged'] - r['given']:+.4f} | {(f'{r['energy_change_kcal']:+.4f}') if ts else ''} | "
                     f"{(f'{r['moved_max_A']:.4f}') if ts else ''} | "
                     f"{(low(r['lowest_cm_given']) + ' -> ' + low(r['lowest_cm_converged'])) if ts else ''} | "
                     f"{'; '.join(r['converged_notes'])} |")
    ok = [r for r in rows if "error" not in r]
    d = [abs(r["converged"] - r["given"]) for r in ok]
    head = [f"Set {sets.get('version')}: {len(rows)} structures ({sum(1 for r in rows if r['kind'] == 'ts')} TSs), "
            f"{len(rows) - len(ok)} failed. Change of the thermal correction (kcal/mol): identical (< 1e-6) in "
            f"{sum(1 for x in d if x < 1e-6)}, < 0.01 in {sum(1 for x in d if x < 0.01)}, < 0.05 in "
            f"{sum(1 for x in d if x < 0.05)}, largest {max(d):.4f}.", ""]
    (out_dir / "fixed_set.md").write_text("\n".join(head + lines) + "\n")
    print("\n".join(head + lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
