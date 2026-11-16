"""The TSs the saddle step moved: is where they ended the same saddle?

    python benchmarks/diagnostics/thermo/moved.py VERIFIED.json RUNS_ROOT PROFILES_DIR OUT_DIR [WORKERS]

VERIFIED.json: verified.py's output. For each of its TSs the saddle step moved by more than 0.01 Angstrom (RMSD)
or 1e-5 Hartree, from the run's own file: the structure converged to HESSIAN_FMAX with the engine's TS optimizer
whatever the rule (as free_energy.FreeEnergies._saddle does it), its energy change, both reaction frequencies
and the overlap of the two reaction modes (|cosine|, each as the rates of change of all interatomic distances
along it: for information, no rule uses it), and the thermal correction of the
first counted molecule three ways: Hessian at the given geometry, at the converged one (its energy change
counted), and as the code decides.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np


def rmsd(a, b) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(b.T @ a)
    if np.linalg.det(u @ vt) < 0:
        s[-1] = -s[-1]
    return float(np.sqrt(max(0.0, (a ** 2).sum() + (b ** 2).sum() - 2 * s.sum()) / len(a)))


def one(job):
    folder, profile, T, ss, r = job
    import mepd
    from qcdata import Structure

    import mepd.discovery.free_energy as fe
    from mepd.nodes.node import StructureNode

    out = {k: r[k] for k in ("run", "label", "molecules")}
    try:
        s = json.loads((Path(folder) / "run/summary.json").read_text())
        step = next(x for x in s["steps"] if x["label"] == r["label"])
        fp = Path(folder) / step["files"]["ts"]
        st = Structure.open(str(fp))
        irc = step["files"].get("irc")
        sp = {x["index"]: x["file"] for x in s["species"]}
        species = tuple(Structure.open(str(Path(folder) / sp[k])) for k in (step["a"], step["b"]))
        ends = fe.step_ends({"ts": str(fp), "irc": str(Path(folder) / irc) if irc else None}, species)
        engine = mepd.engine_from_profile(profile)
        free = fe.FreeEnergies(engine, T, ss)
        atoms, q, m = free.parts(st, True, ends)[0]
        sym = [str(x) for x in st.symbols]
        xyz = np.asarray(st.geometry, dtype=float).reshape(-1, 3)
        node = StructureNode(structure=Structure(symbols=[sym[k] for k in atoms], geometry=xyz[atoms], charge=q,
                                                 multiplicity=m))
        tight = mepd.find_ts(node, engine, fmax=fe.HESSIAN_FMAX, maxiter=200, continue_unconverged=0,
                             check_frequencies=False)
        e0, e1 = mepd.energies([node, tight], engine)
        out["rmsd_A"] = rmsd(np.asarray(node.structure.geometry).reshape(-1, 3),
                             np.asarray(tight.structure.geometry).reshape(-1, 3)) / fe.BOHR_PER_ANGSTROM
        out["dE_kcal"] = float(e1 - e0) * fe.HARTREE_TO_KCAL
        rates, out["reaction_cm"] = [], []
        for n in (node, tight):
            f = mepd.frequencies(n, engine)
            k = int(np.argmin(f.wavenumbers_cm))
            out["reaction_cm"].append(float(f.wavenumbers_cm[k]))
            x = np.asarray(n.structure.geometry, dtype=float).reshape(-1, 3)
            v = np.asarray(f.modes[k], dtype=float).reshape(-1, 3)
            i, j = np.triu_indices(len(x), 1)
            d = x[i] - x[j]
            rates.append(((v[i] - v[j]) * d).sum(axis=1) / np.linalg.norm(d, axis=1))
        out["overlap"] = float(abs(rates[0] @ rates[1]) / (np.linalg.norm(rates[0]) * np.linalg.norm(rates[1])))
        args = ([sym[k] for k in atoms], xyz[atoms] / fe.BOHR_PER_ANGSTROM, q, m, True, True)
        saddle = fe.FreeEnergies._saddle
        fe.FreeEnergies._saddle = lambda self, n: (n, 0.0)
        out["given"] = free._one(*args)
        fe.FreeEnergies._saddle = lambda self, n: (tight, out["dE_kcal"])
        out["converged"] = free._one(*args)
        fe.FreeEnergies._saddle = saddle
        free = fe.FreeEnergies(engine, T, ss)
        out["code"] = free._one(*args)
        out["notes"] = list(free.notes)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def main(verified, root, profiles, out_dir, workers="1"):
    rows = [r for r in json.loads(Path(verified).read_text())["rows"] if r.get("applied") and
            (r["saddle"]["rmsd_A"] > 0.01 or abs(r["saddle"]["dE_Eh"]) > 1e-5)]
    jobs = []
    for r in rows:
        folder = Path(root) / r["run"]
        s = json.loads((folder / "run/summary.json").read_text())
        jobs.append((str(folder), str(Path(profiles) / Path(s["inputs"]).name), s["steering"]["temperature_K"],
                     s["steering"]["standard_state"], r))
    with get_context("fork").Pool(int(workers)) as pool:
        res = pool.map(one, jobs, chunksize=1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "moved.json").write_text(json.dumps(res, indent=1))
    ok = sorted((r for r in res if "error" not in r), key=lambda r: -abs(r["dE_kcal"]))
    lines = [f"{len(res)} TSs moved by the saddle step ({len(res) - len(ok)} failed here); kept at the given geometry "
             f"by the rule: {sum(1 for r in ok if r['notes'])}.", "",
             "| run | TS | molecules | RMSD A | dE kcal/mol | reaction mode given -> converged, cm-1 | overlap | "
             "correction: given | converged | code | rule |", "|" + "---|" * 11]
    for r in ok:
        lines.append(f"| {r['run']} | {r['label']} | {'+'.join(map(str, r['molecules']))} | {r['rmsd_A']:.3f} | "
                     f"{r['dE_kcal']:+.4f} | {r['reaction_cm'][0]:.0f} -> {r['reaction_cm'][1]:.0f} | "
                     f"{'-' if r['overlap'] is None else '%.3f' % r['overlap']} | {r['given']:.3f} | "
                     f"{r['converged']:.3f} | {r['code']:.3f} | {'kept given' if r['notes'] else 'converged'} |")
    lines += [f"| {r['run']} | {r['label']} | {r['error']} |" for r in res if "error" in r]
    (out_dir / "moved.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
