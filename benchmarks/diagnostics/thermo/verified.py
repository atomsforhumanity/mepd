"""Every IRC-verified TS of a set of runs: what the thermal correction's saddle step did to it, and, for the
groups of one TS found again that still spread, why.

    python benchmarks/diagnostics/thermo/verified.py --runs ROOT --profiles DIR --spread SPREAD.json \\
        --min-spread 0.05 --workers 8 --out DIR

ROOT/<run>/run/summary.json and the files it names (the runs unpacked); DIR holds the runs' profiles (the file
name each summary's `inputs` gives). For every step with files.verified true and a recorded thermal correction,
with the explorer's own code (mepd.discovery.free_energy) and nothing built:
  - the molecules its TS counts as, and of the first (the one that carries the reaction mode): the largest atomic
    force and the frequencies at the geometry in the file;
  - whether FreeEnergies._saddle applies (force between HESSIAN_FMAX and twice the searches' threshold), and
    then: how far it moved the structure (RMSD after a best fit, Angstrom), its energy change (Hartree), the
    force and the frequencies where it ended.
For the members of the groups of SPREAD.json (spread.py; verified members of two or more runs, spread of the
recorded corrections >= --min-spread) also: the correction with the Hessian at the geometry given and as the code
computes it (against the recorded one), xtb's rotational number and linearity, chirality (mirror.is_chiral), and
what the file is a stationary point of (the whole box; the subsystem its ends file gives).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BOHR = 1.8897259886
EV_A = 27.211386245988 * BOHR


def rmsd(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, s, vt = np.linalg.svd(p.T @ q)
    if np.linalg.det(u @ vt) < 0:
        s[-1] = -s[-1]
    return float(np.sqrt(max(0.0, (p ** 2).sum() + (q ** 2).sum() - 2 * s.sum()) / len(p)))


def one(job):
    folder, profile, T, ss, step, files_a, files_b, detail = job
    import mepd
    from qcdata import Structure

    import mepd.discovery.free_energy as fe
    from mepd.nodes.node import StructureNode

    out = {"run": Path(folder).name, "label": step["label"], "e": step["ts_energy"], "detail": detail,
           "corr": step["ts_energy_terms"]["thermal_correction_kcal"]}
    try:
        fp = Path(folder) / step["files"]["ts"]
        st = Structure.open(str(fp))
        irc = step["files"].get("irc")
        species = tuple(Structure.open(str(Path(folder) / f)) for f in (files_a, files_b))
        ends = fe.step_ends({"ts": str(fp), "irc": str(Path(folder) / irc) if irc else None}, species)
        engine = mepd.engine_from_profile(profile)
        free = fe.FreeEnergies(engine, T, ss)
        parts = free.parts(st, True, ends)
        out["molecules"] = [len(p[0]) for p in parts]
        out["notes"] = list(free.notes)
        sym = [str(s) for s in st.symbols]
        xyz = np.asarray(st.geometry, dtype=float).reshape(-1, 3)

        def node_of(atoms, q=None, m=None):
            return StructureNode(structure=Structure(
                symbols=[sym[k] for k in atoms], geometry=xyz[atoms], charge=st.charge if q is None else q,
                multiplicity=st.multiplicity if m is None else m))

        def force(node):
            g = np.asarray(engine.compute_gradients([node])[0], dtype=float).reshape(-1, 3)
            return float(np.linalg.norm(g, axis=1).max()) * EV_A

        def freqs(node):
            return sorted(mepd.frequencies(node, engine).wavenumbers_cm.tolist())

        node = node_of(*parts[0])
        out["fmax"] = force(node)
        w0 = freqs(node)
        out["given"] = {"lowest6": w0[:6], "negative": sum(1 for x in w0 if x < 0),
                        "below_-10": sum(1 for x in w0 if x < -10)}
        from mepd.inputs import TSInputs

        out["applied"] = bool(getattr(engine, "analytic_hessian", False)
                              and fe.HESSIAN_FMAX < out["fmax"] <= 2 * TSInputs.fmax)
        if out["applied"]:
            tight, moved = free._saddle(node)
            w1 = freqs(tight)
            out["saddle"] = {"rmsd_A": rmsd(np.asarray(tight.structure.geometry).reshape(-1, 3) / BOHR,
                                            np.asarray(node.structure.geometry).reshape(-1, 3) / BOHR),
                             "dE_Eh": moved / fe.HARTREE_TO_KCAL, "fmax": force(tight), "lowest6": w1[:6],
                             "negative": sum(1 for x in w1 if x < 0), "below_-10": sum(1 for x in w1 if x < -10),
                             "notes": list(free.notes)[len(out["notes"]):]}
        if detail:
            from mepd.discovery.mirror import is_chiral
            from mepd.thermo import frequencies as xtb_analysis

            saddle = fe.FreeEnergies._saddle
            fe.FreeEnergies._saddle = lambda self, node: (node, 0.0)
            out["corr_given"] = fe.FreeEnergies(engine, T, ss)._correction(st, True, ends, step["ts_energy"])
            fe.FreeEnergies._saddle = saddle
            out["corr_code"] = fe.FreeEnergies(engine, T, ss)._correction(st, True, ends, step["ts_energy"])
            p0 = parts[0][0]
            try:
                shape = xtb_analysis([sym[k] for k in p0], xyz[p0] / BOHR, charge=parts[0][1], multiplicity=parts[0][2])
                out["rotational_number"], out["linear"] = shape["rotational_number"], shape["linear"]
            except Exception as exc:
                out["rotational_number"] = f"failed ({type(exc).__name__})"
            out["chiral"] = is_chiral(st)
            out["fmax_whole"] = out["fmax"] if len(parts) == 1 else force(node_of(list(range(len(sym)))))
            ef = fp.with_name(f"{fp.stem}_ends.xyz")
            out["ends_file"] = ef.exists()
            if ef.exists():
                fr = Structure.open_multi(str(ef))
                e2 = [np.asarray(k.geometry, dtype=float).reshape(-1, 3) / BOHR for k in (fr[0], fr[-1])]
                cut = fe.ts_groups(sym, e2[0], e2)
                if cut and len(cut) > 1:
                    out["ends_file_subsystem"] = len(cut[0])
                    out["fmax_ends_file_subsystem"] = force(node_of(cut[0]))
            ex = [np.asarray(e.geometry, dtype=float).reshape(-1, 3) / BOHR for e in ends]
            b0, b1, bts = fe.bonds(sym, ex[0]), fe.bonds(sym, ex[1]), fe.bonds(sym, xyz / BOHR)

            def named(pairs):
                return [f"{sym[i]}{i}-{sym[j]}{j} {np.linalg.norm(xyz[i] - xyz[j]) / BOHR:.2f}" for i, j in sorted(pairs)]
            out["ends_bonds"] = {"both_ends_missing_at_ts": named((b0 & b1) - bts),
                                 "ts_in_neither_end": named(bts - (b0 | b1)), "changed": named(b0 ^ b1)}
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--spread", required=True)
    ap.add_argument("--min-spread", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import multiprocessing as mp

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = []
    for g in json.loads(Path(a.spread).read_text())["TSs"]:
        ms = [m for m in g["members"] if m.get("verified") is True]
        c = [m["corr"] for m in ms]
        if len({m["run"] for m in ms}) >= 2 and max(c) - min(c) >= a.min_spread:
            groups.append({**g, "members": ms, "spread_kcal": max(c) - min(c)})
    wanted = {(m["run"], m["label"]) for g in groups for m in g["members"]}
    jobs = []
    for fp in sorted(Path(a.runs).glob("*/run/summary.json")):
        folder = fp.parent.parent
        s = json.loads(fp.read_text())
        st = s["steering"]
        sp = {x["index"]: x["file"] for x in s["species"]}
        profile = str(Path(a.profiles) / Path(s["inputs"]).name)
        for step in s["steps"]:
            if (step.get("files") or {}).get("verified") is True and \
                    (step.get("ts_energy_terms") or {}).get("thermal_correction_kcal") is not None:
                jobs.append((str(folder), profile, st["temperature_K"], st["standard_state"],
                             {k: step[k] for k in ("label", "ts_energy", "ts_energy_terms", "files")},
                             sp[step["a"]], sp[step["b"]], (folder.name, step["label"]) in wanted))
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    by = {(r["run"], r["label"]): r for r in rows}
    for g in groups:
        g["members"] = [{**m, **by.get((m["run"], m["label"]), {"error": "not among the verified steps"})}
                        for m in g["members"]]
    (out / "verified.json").write_text(json.dumps({"rows": rows, "groups": groups}, indent=1))
    ok = [r for r in rows if "error" not in r]
    ap_ = [r for r in ok if r["applied"]]
    per = defaultdict(lambda: [0, 0])
    for r in ok:
        per[r["run"]][0] += 1
        per[r["run"]][1] += r["applied"]
    lines = [f"{len(rows)} IRC-verified TSs with a recorded thermal correction; {len(rows) - len(ok)} failed here.", "",
             f"- _saddle applied (force between 1e-4 and 0.002 eV/A): {len(ap_)}; already below 1e-4: "
             f"{sum(1 for r in ok if r['fmax'] <= 1e-4)}; above 0.002 (not touched): {sum(1 for r in ok if r['fmax'] > 2e-3)}",
             f"- of those applied: moved by more than 0.01 A RMSD: {sum(1 for r in ap_ if r['saddle']['rmsd_A'] > 0.01)}; "
             f"energy change above 1e-5 Eh: {sum(1 for r in ap_ if abs(r['saddle']['dE_Eh']) > 1e-5)}; either: "
             f"{sum(1 for r in ap_ if r['saddle']['rmsd_A'] > 0.01 or abs(r['saddle']['dE_Eh']) > 1e-5)}; "
             f"geometry kept: optimizer failed {sum(1 for r in ap_ if any('could not' in n for n in r['saddle']['notes']))}, "
             f"refused as another structure {sum(1 for r in ap_ if any('another structure' in n for n in r['saddle']['notes']))} "
             f"({'; '.join(sorted(r['run'].split('-lin2')[0] + ' ' + r['label'] + ': ' + n.split('(')[1].split(')')[0] for r in ap_ for n in r['saddle']['notes'] if 'another structure' in n))})",
             f"- of those applied, negative modes where it ended: exactly one {sum(1 for r in ap_ if r['saddle']['negative'] == 1)}, "
             f"none {sum(1 for r in ap_ if r['saddle']['negative'] == 0)}, two or more "
             f"{sum(1 for r in ap_ if r['saddle']['negative'] > 1)} (below -10 cm-1: one "
             f"{sum(1 for r in ap_ if r['saddle']['below_-10'] == 1)}, none {sum(1 for r in ap_ if r['saddle']['below_-10'] == 0)}, "
             f"more {sum(1 for r in ap_ if r['saddle']['below_-10'] > 1)}); at the geometry given, the same TSs: one "
             f"{sum(1 for r in ap_ if r['given']['negative'] == 1)}, none {sum(1 for r in ap_ if r['given']['negative'] == 0)}, "
             f"more {sum(1 for r in ap_ if r['given']['negative'] > 1)}",
             f"- not applied, force above 0.002: negative modes at the geometry given: one "
             f"{sum(1 for r in ok if r['fmax'] > 2e-3 and r['given']['negative'] == 1)}, none "
             f"{sum(1 for r in ok if r['fmax'] > 2e-3 and r['given']['negative'] == 0)}, more "
             f"{sum(1 for r in ok if r['fmax'] > 2e-3 and r['given']['negative'] > 1)}", "",
             "| run | verified TSs | applied |", "|---|---|---|"]
    lines += [f"| {k} | {v[0]} | {v[1]} |" for k, v in sorted(per.items())]
    lines += ["", "Groups (recorded spread >= %g): one line per member." % a.min_spread, "",
              "| group | member | recorded | recomputed | Hessian at the given geometry | molecules | force eV/A (first "
              "molecule; whole box; ends-file subsystem) | saddle step: RMSD A, dE Eh | lowest six, given | "
              "lowest six, where it ended | sigma, linear, chiral | ends: both missing at TS / TS in neither |",
              "|" + "---|" * 12]

    def six(w):
        return " ".join(f"{x:.0f}" for x in w)
    for g in sorted(groups, key=lambda g: -g["spread_kcal"]):
        for m in g["members"]:
            name = f"{g['system']} {g['what']} {g['energy']:.6f} ({g['spread_kcal']:.3f})"
            if "error" in m:
                lines.append(f"| {name} | {m['run'][-2:]}:{m['label']} | {m['corr']:.3f} | {m['error']} |")
                continue
            sd = m.get("saddle")
            eb = m.get("ends_bonds", {})
            lines.append(
                f"| {name} | {m['run'][-2:]}:{m['label']} | {m['corr']:.3f} | {m.get('corr_code', float('nan')):.3f} | "
                f"{m.get('corr_given', float('nan')):.3f} | {'+'.join(map(str, m['molecules']))} | {m['fmax']:.4f}; "
                f"{m.get('fmax_whole', float('nan')):.4f}; "
                f"{('%.4f (%d atoms)' % (m['fmax_ends_file_subsystem'], m['ends_file_subsystem'])) if 'ends_file_subsystem' in m else '-'} | "
                f"{('%.4f, %.1e' % (sd['rmsd_A'], sd['dE_Eh'])) if sd else 'not applied'} | {six(m['given']['lowest6'])} | "
                f"{six(sd['lowest6']) if sd else '-'} | {m.get('rotational_number')}, {m.get('linear')}, {m.get('chiral')} | "
                f"{'; '.join(eb.get('both_ends_missing_at_ts', [])) or '-'} / {'; '.join(eb.get('ts_in_neither_end', [])) or '-'} |")
    (out / "verified.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
